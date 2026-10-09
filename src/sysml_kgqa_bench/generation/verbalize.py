"""Verbalize sampled candidates: generate paraphrases of each canonical question and filter them.

Paraphrases pass mechanical filters (no answer leakage, sufficient novelty, all bound entities kept) and an
LLM judge that rejects paraphrases changing the set of valid answers. The first accepted paraphrase becomes
the instance's `question`; the canonical question is used if none is accepted.
"""

import argparse
import asyncio
import difflib
import hashlib
import json
import os
import random
from collections import defaultdict
from datetime import datetime, timezone

from pydantic import BaseModel

from sysml_kgqa_bench import paths
from sysml_kgqa_bench.graph.backend import KuzuBackend
from sysml_kgqa_bench.llm import BACKENDS, build_async_llm_client, build_llm_client
from sysml_kgqa_bench.utils import TEMPLATE_FILES, load_jsonl, load_yaml, record_key

DEFAULT_MODEL_BY_BACKEND = {"openai": "gpt-4.1-mini", "private": "qwen-3-5-397b-a17b"}
DEFAULT_JUDGE_MODEL_BY_BACKEND = {"openai": "gpt-4.1", "private": "qwen3-next-80b-a3b-thinking"}
JUDGE_TEMPERATURE_DEFAULT = 0.0


class ParaphraseResponse(BaseModel):
    paraphrases: list[str]


class ConstraintJudgment(BaseModel):
    reason: str
    adds_qualifier_not_in_canonical: bool
    narrows_general_noun_to_specific: bool
    drops_qualifier_present_in_canonical: bool
    changes_logical_connective_or_comparison: bool


def build_clients(args):
    """Build the generation and judge clients (async if --concurrency > 1)."""
    gen_model = args.model or DEFAULT_MODEL_BY_BACKEND[args.backend]
    judge_model = args.judge_model or DEFAULT_JUDGE_MODEL_BY_BACKEND[args.judge_backend]
    extra_body = json.loads(args.extra_body_json) if args.extra_body_json else {}
    judge_extra_body = json.loads(args.judge_extra_body_json) if args.judge_extra_body_json else {}
    if args.concurrency > 1:
        gen = build_async_llm_client(
            args.backend, gen_model, args.timeout, args.max_tokens, extra_body, args.max_retries
        )
        judge = build_async_llm_client(
            args.judge_backend,
            judge_model,
            args.judge_timeout,
            args.judge_max_tokens,
            judge_extra_body,
            args.max_retries,
        )
    else:
        gen = build_llm_client(args.backend, gen_model, args.timeout, args.max_tokens, extra_body)
        judge = build_llm_client(
            args.judge_backend, judge_model, args.judge_timeout, args.judge_max_tokens, judge_extra_body
        )
    return gen, judge


TASK_CLASS_DESCRIPTION = {
    "impact_analysis": "Identify every element that would be affected, directly or "
    "transitively, if this element changes.",
    "negation": "Identify, within a defined cluster, which elements are not further "
    "refined by anything else in that cluster.",
    "aggregation": "Count how many active (non-deprecated) requirements exist within a given scope.",
    "tracing": "Determine whether one requirement can be derived from another through "
    "a chain of intermediate requirements, and name the chain.",
    "rationale_abstention": "State the rationale recorded for a requirement, or recognize that none is recorded.",
    "lookup": "Look up a specific attribute value of a named model element.",
}


def load_template_metas(templates_dir):
    metas = {}
    for fname in TEMPLATE_FILES:
        meta = load_yaml(f"{templates_dir}/{fname}")
        metas[meta["template_id"]] = meta
    return metas


def render_canonical_nl(record, meta):
    """Fill the template's canonical question with the record's entity names."""
    template_id = record["template_id"]
    dc = record["difficulty_coords"]

    if template_id == "impact_analysis.requirement_closure":
        text = meta["canonical_nl"]
        return text.replace("{X.name}", record["seed_name"])

    if template_id == "impact_analysis.structural_closure":
        seed_type = dc["seed_type"]
        text = meta.get("canonical_nl_by_seed_type", {}).get(seed_type, meta["canonical_nl"])
        return text.replace("{X.name}", record["seed_name"])

    if template_id == "negation.requirement_unrefined":
        text = meta["canonical_nl"]
        return text.replace("{cluster_anchor_name}", record["cluster_anchor_name"])

    if template_id == "aggregation.requirement_count":
        scope_unit_type = dc["scope_unit_type"]
        text = meta.get("canonical_nl_by_scope_unit_type", {}).get(scope_unit_type, meta["canonical_nl"])
        return text.replace("{scope_name}", record["scope_name"])

    if template_id == "tracing.requirement_derivation":
        text = meta["canonical_nl"] if record["target_is_requirement"] else meta["canonical_nl_cross_layer"]
        return text.replace("{source_name}", record["source_name"]).replace("{target_name}", record["target_name"])

    if template_id == "rationale_abstention":
        text = meta["canonical_nl"]
        return text.replace("{name}", record["seed_name"])

    if template_id == "lookup.value_property_attribute":
        text = meta["canonical_nl"]
        return text.replace("{name}", record["seed_name"]).replace("{owning_block_name}", dc["owning_block_name"])

    raise ValueError(f"no renderer for {template_id}")


def bound_entities(record, meta):
    """Entities (name and type) that every paraphrase must mention."""
    template_id = record["template_id"]
    dc = record["difficulty_coords"]

    if template_id == "impact_analysis.requirement_closure":
        return [{"name": record["seed_name"], "type": "Requirement"}]
    if template_id == "impact_analysis.structural_closure":
        return [{"name": record["seed_name"], "type": dc["seed_type"]}]
    if template_id == "negation.requirement_unrefined":
        return [{"name": record["cluster_anchor_name"], "type": "Requirement (cluster anchor)"}]
    if template_id == "aggregation.requirement_count":
        unit = "document section" if dc["scope_unit_type"] == "header" else "package"
        return [{"name": record["scope_name"], "type": unit}]
    if template_id == "tracing.requirement_derivation":
        return [
            {"name": record["source_name"], "type": "Requirement"},
            {"name": record["target_name"], "type": record.get("target_type", "Requirement")},
        ]
    if template_id == "rationale_abstention":
        return [{"name": record["seed_name"], "type": "Requirement"}]
    if template_id == "lookup.value_property_attribute":
        return [
            {"name": record["seed_name"], "type": "ValueProperty"},
            {"name": dc["owning_block_name"], "type": "Block"},
        ]
    raise ValueError(f"no bound_entities for {template_id}")


def gold_answer_surface_strings(record, name_lookup):
    """Strings from the gold answer that must not appear in a paraphrase."""
    template_id = record["template_id"]
    answer = record["answer"]

    if template_id in (
        "impact_analysis.requirement_closure",
        "impact_analysis.structural_closure",
        "negation.requirement_unrefined",
    ):
        return [name_lookup[i] for i in answer if name_lookup.get(i)]
    if template_id == "aggregation.requirement_count":
        return [str(record["difficulty_coords"]["strict_count"])]
    if template_id == "tracing.requirement_derivation":
        path = answer["witness_path"]
        return [name_lookup[i] for i in path[1:-1] if name_lookup.get(i)]
    if template_id == "rationale_abstention":
        return [answer] if answer else []
    if template_id == "lookup.value_property_attribute":
        return [str(answer)]
    raise ValueError(f"no gold_answer_surface_strings for {template_id}")


GENERATION_PROMPT = """You are helping build a question-answering benchmark over a SysML \
systems-engineering model (the Thirty Meter Telescope). You will be given a canonical \
question and must write natural paraphrases of it, as a systems/requirements engineer \
would actually ask them in conversation or a design review.

Task type: {task_class} -- {description}
Canonical question: "{canonical_nl}"
Entities referenced in this question:
{entity_list}

Write {n} paraphrases. Requirements, all of them mandatory:
1. Each paraphrase must ask for EXACTLY the same thing as the canonical question -- do \
not add, drop, narrow, or broaden any constraint.
2. Each paraphrase must be self-contained (understandable with no other context) and \
must refer to the same entities, by name, as the canonical question.
3. Do not mention "the model", "the graph", "the database", queries, or any \
implementation detail -- write as an engineer, not as someone querying a system.
4. Vary sentence structure and wording meaningfully. A synonym swap alone is not a \
useful paraphrase.
5. Do not include or hint at the answer.

Return strictly valid JSON, no other text: {{"paraphrases": ["...", "...", ...]}}
"""

CONSTRAINT_JUDGE_PROMPT = """You are checking whether a paraphrase silently changed what a question \
asks for -- not whether it sounds natural, only whether the SET OF VALID ANSWERS could differ.

Canonical question: "{canonical_nl}"
Candidate paraphrase: "{paraphrase}"

Answer FOUR separate yes/no checks -- do not weigh them holistically, answer each independently \
against ONLY the definition given, for ANY instance of this model (not just this one; a restriction \
that happens to match zero or one specific case here is still a restriction):

1. adds_qualifier_not_in_canonical: does the paraphrase ADD a restricting phrase not present in the \
canonical question (a location, package, subsystem, type, or category qualifier)? E.g. "elements \
affected" becoming "elements in the OAD package affected" -- restricts to a subset even though it \
reads naturally.
2. narrows_general_noun_to_specific: does the paraphrase narrow a general noun in the canonical \
question to a more specific category? E.g. "elements" -> "mechanical elements".
3. drops_qualifier_present_in_canonical: does the paraphrase DROP a qualifier the canonical question \
has? E.g. dropping "active (non-deprecated)" from a count question would silently include retired \
content.
4. changes_logical_connective_or_comparison: does the paraphrase change an "and"/"or", "any"/"all", \
or a comparison direction from the canonical question?

Return strictly valid JSON, no other text -- a brief one-sentence "reason" citing which check(s) \
fired (or "none" if all four are false), then the four booleans: \
{{"reason": "...", "adds_qualifier_not_in_canonical": true/false, \
"narrows_general_noun_to_specific": true/false, "drops_qualifier_present_in_canonical": true/false, \
"changes_logical_connective_or_comparison": true/false}}
"""


def generate_paraphrases(gen_client, meta, entities, canonical_nl, n, temperature):
    entity_list = "\n".join(f"- {e['name']} ({e['type']})" for e in entities)
    task_class = meta["task_class"]
    prompt = GENERATION_PROMPT.format(
        task_class=task_class,
        description=TASK_CLASS_DESCRIPTION[task_class],
        canonical_nl=canonical_nl,
        entity_list=entity_list,
        n=n,
    )
    result = gen_client.complete_json(prompt, temperature)
    return result.get("paraphrases", []), prompt


async def generate_paraphrases_async(gen_client, meta, entities, canonical_nl, n, temperature):
    entity_list = "\n".join(f"- {e['name']} ({e['type']})" for e in entities)
    task_class = meta["task_class"]
    prompt = GENERATION_PROMPT.format(
        task_class=task_class,
        description=TASK_CLASS_DESCRIPTION[task_class],
        canonical_nl=canonical_nl,
        entity_list=entity_list,
        n=n,
    )
    result = await gen_client.acomplete_json(prompt, temperature, response_model=ParaphraseResponse)
    return result.get("paraphrases", []), prompt


def filter_self_answering(paraphrase, surface_strings, canonical_nl):
    """Reject paraphrases that contain part of the gold answer."""
    low = paraphrase.lower()
    canonical_low = canonical_nl.lower()
    for s in surface_strings:
        if not s or len(s) < 3:
            continue
        s_low = s.lower()
        if s_low in canonical_low:
            continue
        if s_low in low:
            return False, f"contains gold-answer surface string {s!r}"
    return True, None


NOVELTY_MAX_RATIO = 0.85


def filter_novelty(paraphrase, canonical_nl):
    """Reject paraphrases that are too similar to the canonical question."""
    ratio = difflib.SequenceMatcher(None, paraphrase.lower(), canonical_nl.lower()).ratio()
    if ratio > NOVELTY_MAX_RATIO:
        return False, f"edit-distance ratio {ratio:.2f} too close to canonical form"
    return True, None


def filter_binding_preserved(paraphrase, entities):
    """Reject paraphrases that drop a bound entity name."""
    low = paraphrase.lower()
    for e in entities:
        if e["name"].lower() not in low:
            return False, f"dropped binding {e['name']!r}"
    return True, None


def filter_no_over_explicit_impact_enumeration(paraphrase):
    """Reject impact questions that spell out direct/indirect/cascading effects."""
    low = paraphrase.lower()
    if (
        "experience" in low
        and ("impact" in low or "effect" in low)
        and ("direct" in low or "cascad" in low or "indirect" in low)
    ):
        return (
            False,
            "over-explicit impact enumeration (\"experience direct/indirect/cascading impact/effect\" instead of a plain 'affected'/'impacted')",
        )
    return True, None


_CONSTRAINT_CHECK_FIELDS = [
    "adds_qualifier_not_in_canonical",
    "narrows_general_noun_to_specific",
    "drops_qualifier_present_in_canonical",
    "changes_logical_connective_or_comparison",
]


def _same_question_from_checks(result):
    return not any(result.get(f, True) for f in _CONSTRAINT_CHECK_FIELDS)


def filter_constraint_preserved(judge_client, paraphrase, canonical_nl, temperature=JUDGE_TEMPERATURE_DEFAULT):
    """LLM judge: reject paraphrases that add, narrow, drop or change a constraint of the question."""
    prompt = CONSTRAINT_JUDGE_PROMPT.format(canonical_nl=canonical_nl, paraphrase=paraphrase)
    result = judge_client.complete_json(prompt, temperature=temperature)
    if not _same_question_from_checks(result):
        return False, f"constraint-addition/scope judge: {result.get('reason')}"
    return True, None


async def filter_constraint_preserved_async(
    judge_client, paraphrase, canonical_nl, temperature=JUDGE_TEMPERATURE_DEFAULT
):
    prompt = CONSTRAINT_JUDGE_PROMPT.format(canonical_nl=canonical_nl, paraphrase=paraphrase)
    result = await judge_client.acomplete_json(prompt, temperature=temperature, response_model=ConstraintJudgment)
    if not _same_question_from_checks(result):
        return False, f"constraint-addition/scope judge: {result.get('reason')}"
    return True, None


def apply_filters(
    judge_client, paraphrase, canonical_nl, entities, surface_strings, judge_temperature=JUDGE_TEMPERATURE_DEFAULT
):
    for filt, args in [
        (filter_self_answering, (paraphrase, surface_strings, canonical_nl)),
        (filter_novelty, (paraphrase, canonical_nl)),
        (filter_binding_preserved, (paraphrase, entities)),
        (filter_no_over_explicit_impact_enumeration, (paraphrase,)),
    ]:
        ok, reason = filt(*args)
        if not ok:
            return False, reason
    return filter_constraint_preserved(judge_client, paraphrase, canonical_nl, judge_temperature)


async def apply_filters_async(
    judge_client, paraphrase, canonical_nl, entities, surface_strings, judge_temperature=JUDGE_TEMPERATURE_DEFAULT
):
    for filt, args in [
        (filter_self_answering, (paraphrase, surface_strings, canonical_nl)),
        (filter_novelty, (paraphrase, canonical_nl)),
        (filter_binding_preserved, (paraphrase, entities)),
        (filter_no_over_explicit_impact_enumeration, (paraphrase,)),
    ]:
        ok, reason = filt(*args)
        if not ok:
            return False, reason
    return await filter_constraint_preserved_async(judge_client, paraphrase, canonical_nl, judge_temperature)


def process_record(
    gen_client,
    judge_client,
    record,
    meta,
    name_lookup,
    n_generate,
    temperature,
    keep_max,
    judge_temperature=JUDGE_TEMPERATURE_DEFAULT,
):
    canonical_nl = render_canonical_nl(record, meta)
    entities = bound_entities(record, meta)
    surface_strings = gold_answer_surface_strings(record, name_lookup)

    raw_paraphrases, prompt = generate_paraphrases(gen_client, meta, entities, canonical_nl, n_generate, temperature)
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]

    accepted, rejected = [], []
    for p in raw_paraphrases:
        if len(accepted) >= keep_max:
            rejected.append({"paraphrase": p, "reason": "kept-max reached (judge not called)"})
            continue
        ok, reason = apply_filters(judge_client, p, canonical_nl, entities, surface_strings, judge_temperature)
        if ok:
            accepted.append(p)
        else:
            rejected.append({"paraphrase": p, "reason": reason})

    question_source = "paraphrase" if accepted else "canonical_fallback"

    provenance = {
        "generation": gen_client.provenance(),
        "judge": judge_client.provenance(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "temperature": temperature,
        "judge_temperature": judge_temperature,
        "prompt_hash": prompt_hash,
    }

    verbalization = {
        "canonical_question": canonical_nl,
        "question": accepted[0] if accepted else canonical_nl,
        "question_source": question_source,
        "accepted_paraphrases": accepted,
        "raw_generations": raw_paraphrases,
        "rejected": rejected,
        "provenance": provenance,
    }
    return {**record, "verbalization": verbalization}


async def process_record_async(
    gen_client,
    judge_client,
    record,
    meta,
    name_lookup,
    n_generate,
    temperature,
    keep_max,
    judge_temperature=JUDGE_TEMPERATURE_DEFAULT,
):
    canonical_nl = render_canonical_nl(record, meta)
    entities = bound_entities(record, meta)
    surface_strings = gold_answer_surface_strings(record, name_lookup)

    raw_paraphrases, prompt = await generate_paraphrases_async(
        gen_client, meta, entities, canonical_nl, n_generate, temperature
    )
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]

    accepted, rejected = [], []
    for p in raw_paraphrases:
        if len(accepted) >= keep_max:
            rejected.append({"paraphrase": p, "reason": "kept-max reached (judge not called)"})
            continue
        ok, reason = await apply_filters_async(
            judge_client, p, canonical_nl, entities, surface_strings, judge_temperature
        )
        if ok:
            accepted.append(p)
        else:
            rejected.append({"paraphrase": p, "reason": reason})

    question_source = "paraphrase" if accepted else "canonical_fallback"

    provenance = {
        "generation": gen_client.provenance(),
        "judge": judge_client.provenance(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "temperature": temperature,
        "judge_temperature": judge_temperature,
        "prompt_hash": prompt_hash,
    }

    verbalization = {
        "canonical_question": canonical_nl,
        "question": accepted[0] if accepted else canonical_nl,
        "question_source": question_source,
        "accepted_paraphrases": accepted,
        "raw_generations": raw_paraphrases,
        "rejected": rejected,
        "provenance": provenance,
    }
    return {**record, "verbalization": verbalization}


def load_done_keys(path):
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                done.add(record_key(json.loads(line)))
    return done


def verbalization_report(out_records, path):
    by_template = defaultdict(list)
    for r in out_records:
        by_template[r["template_id"]].append(r)

    lines = ["# Verbalization report\n"]
    total_n, total_fallback = 0, 0
    for template_id, recs in by_template.items():
        n = len(recs)
        n_fallback = sum(1 for r in recs if r["verbalization"]["question_source"] == "canonical_fallback")
        n_accepted_total = sum(len(r["verbalization"]["accepted_paraphrases"]) for r in recs)
        n_raw_total = sum(len(r["verbalization"]["raw_generations"]) for r in recs)
        total_n += n
        total_fallback += n_fallback
        lines.append(
            f"{template_id}: {n} instances, {n_fallback} canonical_fallback "
            f"({n_fallback / n:.1%}), {n_accepted_total}/{n_raw_total} paraphrases accepted"
        )
    if total_n:
        lines.append(
            f"\nCOMBINED: {total_n} instances, {total_fallback} canonical_fallback ({total_fallback / total_n:.1%})"
        )
    report = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    return report


def build_name_lookup(backend, records):
    from sysml_kgqa_bench.generation.leakage_audit import fetch_names

    ids = set()
    for r in records:
        t = r["template_id"]
        if t in (
            "impact_analysis.requirement_closure",
            "impact_analysis.structural_closure",
            "negation.requirement_unrefined",
        ):
            ids.update(r["answer"])
        elif t == "tracing.requirement_derivation":
            ids.update(r["answer"]["witness_path"])
    return fetch_names(backend, ids)


async def run_concurrent(gen_client, judge_client, todo, metas, name_lookup, args, out_path):
    sem = asyncio.Semaphore(args.concurrency)

    async def bound_process(record):
        async with sem:
            meta = metas[record["template_id"]]
            return await process_record_async(
                gen_client,
                judge_client,
                record,
                meta,
                name_lookup,
                args.n_generate,
                args.temperature,
                args.keep_max,
                args.judge_temperature,
            )

    tasks = [asyncio.create_task(bound_process(r)) for r in todo]
    n_fallback = 0
    n_done = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for coro in asyncio.as_completed(tasks):
            out = await coro
            if out["verbalization"]["question_source"] == "canonical_fallback":
                n_fallback += 1
            f.write(json.dumps(out) + "\n")
            f.flush()
            n_done += 1
            if n_done % 10 == 0 or n_done == len(todo):
                print(f"  {n_done}/{len(todo)} processed this run, {n_fallback} canonical_fallback so far", flush=True)
    return n_fallback


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", default=str(paths.CANDIDATES_SAMPLED))
    ap.add_argument("--templates-dir", default=str(paths.TEMPLATES))
    ap.add_argument("--db-path", default=str(paths.KUZU_DB))
    ap.add_argument("--outdir", default=str(paths.BENCHMARK))
    ap.add_argument("--backend", choices=BACKENDS, default="openai", help="backend for paraphrase generation")
    ap.add_argument("--model", default=None, help="generation model (default depends on backend)")
    ap.add_argument("--judge-backend", choices=BACKENDS, default=None, help="defaults to --backend")
    ap.add_argument("--judge-model", default=None, help="judge model (default depends on backend)")
    ap.add_argument("--timeout", type=float, default=120, help="seconds per generation request")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--judge-timeout", type=float, default=180, help="seconds per judge request")
    ap.add_argument("--judge-max-tokens", type=int, default=8192)
    ap.add_argument("--extra-body-json", default=None, help="JSON passed as extra_body to generation calls")
    ap.add_argument("--judge-extra-body-json", default=None, help="JSON passed as extra_body to judge calls")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--judge-temperature", type=float, default=JUDGE_TEMPERATURE_DEFAULT)
    ap.add_argument("--n-generate", type=int, default=5, help="paraphrases generated per instance")
    ap.add_argument("--keep-max", type=int, default=3, help="paraphrases accepted per instance at most")
    ap.add_argument("--limit-per-class", type=int, default=None)
    ap.add_argument("--only-templates", default=None, help="comma-separated template ids")
    ap.add_argument("--seed", type=int, default=20260724)
    ap.add_argument("--fresh", action="store_true", help="ignore existing output instead of resuming")
    ap.add_argument("--concurrency", type=int, default=1, help="records processed concurrently")
    ap.add_argument("--max-retries", type=int, default=6, help="retries per request when --concurrency > 1")
    args = ap.parse_args()
    if args.judge_backend is None:
        args.judge_backend = args.backend

    use_async = args.concurrency > 1
    backend = KuzuBackend(args.db_path)
    gen_client, judge_client = build_clients(args)
    metas = load_template_metas(args.templates_dir)

    records = load_jsonl(args.candidates)
    if args.only_templates:
        wanted = set(args.only_templates.split(","))
        records = [r for r in records if r["template_id"] in wanted]

    by_template = defaultdict(list)
    for r in records:
        by_template[r["template_id"]].append(r)

    rng = random.Random(args.seed)
    selected = []
    for template_id, recs in by_template.items():
        if args.limit_per_class is not None:
            recs = rng.sample(recs, min(args.limit_per_class, len(recs)))
        selected.extend(recs)
    rng.shuffle(selected)

    name_lookup = build_name_lookup(backend, selected)

    out_path = f"{args.outdir}/verbalized.jsonl"
    if args.fresh and os.path.exists(out_path):
        os.remove(out_path)
    done_keys = load_done_keys(out_path)
    todo = [r for r in selected if record_key(r) not in done_keys]

    gen_prov, judge_prov = gen_client.provenance(), judge_client.provenance()
    print(
        f"Verbalizing {len(selected)} records\n"
        f"  generation: backend={gen_prov['backend']} model={gen_prov['model']}\n"
        f"  judge:      backend={judge_prov['backend']} model={judge_prov['model']}\n"
        f"  n_generate={args.n_generate}, keep_max={args.keep_max}"
        f"{f', concurrency={args.concurrency}' if use_async else ''}"
    )
    if done_keys:
        print(f"  resuming: {len(done_keys)} already in {out_path}, {len(todo)} remaining")

    if use_async:
        asyncio.run(run_concurrent(gen_client, judge_client, todo, metas, name_lookup, args, out_path))
    else:
        n_fallback = 0
        with open(out_path, "a", encoding="utf-8") as f:
            for i, record in enumerate(todo):
                meta = metas[record["template_id"]]
                out = process_record(
                    gen_client,
                    judge_client,
                    record,
                    meta,
                    name_lookup,
                    args.n_generate,
                    args.temperature,
                    args.keep_max,
                    args.judge_temperature,
                )
                if out["verbalization"]["question_source"] == "canonical_fallback":
                    n_fallback += 1
                f.write(json.dumps(out) + "\n")
                f.flush()
                if (i + 1) % 10 == 0 or (i + 1) == len(todo):
                    print(f"  {i + 1}/{len(todo)} processed this run, {n_fallback} canonical_fallback so far")

    all_out_records = load_jsonl(out_path)
    report = verbalization_report(all_out_records, f"{args.outdir}/verbalization_report.md")
    print(f"\nWrote {len(all_out_records)} total verbalized records to {out_path}")
    print(report)


if __name__ == "__main__":
    main()
