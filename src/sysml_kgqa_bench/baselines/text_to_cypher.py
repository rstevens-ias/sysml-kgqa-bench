"""Text-to-Cypher baseline: the LLM writes a Cypher query, which is executed against the Kuzu graph.

Failed queries are sent back to the model with the error message, up to --max-attempts times.
"""

import argparse
import json

from sysml_kgqa_bench import paths
from sysml_kgqa_bench.baselines.common import (
    add_common_args,
    assert_consistent_model,
    build_baseline_client,
    results_path,
    run_checkpointed,
)
from sysml_kgqa_bench.embeddings import get_default_embed_fn
from sysml_kgqa_bench.evaluation.metrics import EXTRACTION_FAILED, build_edge_exists_fn, score_instance
from sysml_kgqa_bench.graph.backend import KuzuBackend
from sysml_kgqa_bench.utils import load_jsonl, load_yaml, record_key


def build_schema_prompt(catalogue_path, model_config):
    """Graph schema, Kuzu dialect notes and example queries given to the model."""
    with open(catalogue_path, encoding="utf-8") as f:
        catalogue = json.load(f)
    node_types = sorted(catalogue["stereotypes"].keys())
    relations = sorted(catalogue["relations"].keys())
    deprecated = model_config["deprecated_stereotypes"]

    deprecated_clause = " AND ".join(f"(n.node_types IS NULL OR NOT '{s}' IN n.node_types)" for s in deprecated)

    return f"""You are writing Cypher queries against a Kuzu graph database (an embedded,
openCypher-dialect engine) storing a SysML systems-engineering model.

SCHEMA:
- Every node has label `Entity` with properties: `id` (string, a synthetic
  OPAQUE identifier, e.g. `_10747_09263153-416e-4119-bb59-4a184d700508` --
  never guess or invent one, only use ids you find via a MATCH), `name`
  (string, the HUMAN-READABLE label, e.g. `REQ-1-OAD-0420` or `Warping
  Harness Maximum Time`), `node_types` (a LIST of stereotype strings -- one
  node can carry several; check membership with `IN`, e.g.
  `WHERE 'TMT Requirement' IN n.node_types`).
- **The question text below refers to entities by their human-readable
  code or label (e.g. "requirement REQ-1-ORD-3660") -- that is ALWAYS the
  `name` property, NEVER the `id` property, even though it may look
  id-like.** Anchor any such entity with `{{name: '<the code/label from
  the question>'}}` (or `WHERE x.name = '<...>'`), e.g. `MATCH (req:Entity
  {{name: 'REQ-1-ORD-3660'}})`. Writing `{{id: 'REQ-1-ORD-3660'}}` will
  silently match ZERO rows, every time -- `id` values never look like
  requirement codes or take the form seen in any question text.
- **`name` is NOT guaranteed unique** -- two different entities can share
  the same name. A `{{name: '...'}}` match can silently bind more than one
  node. If a query's correctness depends on a single unique anchor (e.g. a
  specific source/target pair for a derivation chain) and you have no
  further distinguishing detail from the question, be aware that matching
  by name alone may pick up an unintended node with the same label --
  this is a real property of the data, not a query mistake to "fix" by
  guessing additional filters that aren't in the question.
- Two additional properties exist on some nodes (null/absent on others,
  never invent a value): `default_value` (on ValueProperty nodes) and
  `rationale` (free text, on TMT Requirement nodes -- may legitimately be
  null if none is recorded).
- Every edge has label `Relation` with a property `relation` (string, the
  actual semantic relation type) -- relations are NOT separate Kuzu edge
  tables, they are ALL the same `Relation` label distinguished by this
  property. Filter with `WHERE r.relation = 'Refines'`, never
  `-[:Refines]->`.
- Valid `node_types` values (stereotypes) in this graph: {", ".join(node_types)}.
- Valid `relation` property values in this graph: {", ".join(relations)}.
- Deprecated/retired content convention: some nodes carry a stereotype
  marking them superseded or non-normative ({", ".join(deprecated)}).
  Exclude them unless the question is explicitly about deprecated content,
  with a clause like: {deprecated_clause}

KUZU DIALECT NOTES (checked directly against this database, not assumed
from Neo4j's Cypher):
- Variable-length traversal filtered by relation type uses Kuzu's
  recursive-predicate MATCH syntax: `-[r:Relation* (r, n | WHERE
  r.relation = 'hasParent')]->` -- NOT the property-map-on-a-quantified-
  relationship form `-[:Relation {{relation: 'hasParent'}}*1..3]->`, which
  is a PARSER ERROR in Kuzu.
- A fixed (non-variable-length) single hop filtered by relation type:
  `-[r:Relation {{relation: 'Refines'}}]->`.
- List containment: `WHERE 'X' IN n.node_types`.
- Negation via subquery: `WHERE NOT EXISTS {{ MATCH (n)-[r:Relation
  {{relation: 'Refines'}}]->(m:Entity) }}`.
- Standard `MATCH`/`WHERE`/`RETURN`/`count()`/`UNION ALL` are supported.
- **Reserved words -- do NOT use these as a node/relationship variable
  name** (e.g. `MATCH (end:Entity)` is a PARSER ERROR, not a name
  collision with anything in this data): `end`, `order`, `where`,
  `group`, `union`, `all`. Checked directly against this database, not
  assumed -- confirmed by testing each as a MATCH variable name. `end` in
  particular is a natural choice when tracing a path to its endpoint
  (`start`...`end`) -- use a different name for that variable, e.g.
  `target` or `dest` (both confirmed to work fine).

EXAMPLE QUERIES (real, working syntax from this exact database -- names/
codes below are illustrative placeholders, not literal targets. Every
example anchors its seed entity by `name`, matching how entities are
actually referenced in question text -- NEVER by `id`, per the note
above):
1. One-hop, relation-filtered:
   `MATCH (a:Entity {{name: '<code from the question>'}})-[r:Relation {{relation: 'Refines'}}]->(b:Entity)
   WHERE {deprecated_clause.replace("n.", "b.")}
   RETURN b.id AS id`
2. Variable-length, relation-filtered (recursive predicate form):
   `MATCH (d:Entity)-[r:Relation* (r, n | WHERE r.relation = 'hasParent')]->(s:Entity {{name: '<code from the question>'}})
   WHERE 'TMT Requirement' IN d.node_types
   RETURN count(d) AS count`
3. Negation (elements with no matching relation):
   `MATCH (t:Entity) WHERE 'TMT Requirement' IN t.node_types
   AND NOT EXISTS {{ MATCH (t)-[r:Relation {{relation: 'Refines'}}]->(u:Entity) }}
   RETURN t.id AS id`
4. Direct property read:
   `MATCH (n:Entity {{name: '<code from the question>'}}) RETURN n.rationale AS rationale`
5. Multi-hop derivation chain, one row per hop via UNION ALL -- the
   GENERAL rule, true regardless of how many hops the chain has: EVERY
   MATCH after the first one, WITHIN THE SAME BRANCH, must be preceded by
   a `WITH` that re-states EVERY variable still needed later in that
   branch (all prior node/relationship variables, not just the newest
   one) -- a bare MATCH right after another, with no `WITH`, is a PARSER
   ERROR in Kuzu, no matter which hop it happens on. A 2-hop chain needs
   one `WITH` per branch; a 3-hop chain needs two; a 4-hop chain needs
   three -- the pattern below extends the same way for any chain length,
   it does not stop applying after the first hop:
   `MATCH (a:Entity {{name: '<source code>'}})-[r1:Relation {{relation: 'Refines'}}]->(b:Entity)
   WITH a, b, r1
   MATCH (b)-[r2:Relation {{relation: 'Refines'}}]->(c:Entity)
   WITH a, b, r1, c, r2
   MATCH (c)-[r3:Relation {{relation: 'Refines'}}]->(d:Entity {{name: '<target code>'}})
   RETURN a.id AS from, b.id AS to, r1.relation AS relation
   UNION ALL
   MATCH (a:Entity {{name: '<source code>'}})-[r1:Relation {{relation: 'Refines'}}]->(b:Entity)
   WITH a, b, r1
   MATCH (b)-[r2:Relation {{relation: 'Refines'}}]->(c:Entity)
   WITH a, b, r1, c, r2
   MATCH (c)-[r3:Relation {{relation: 'Refines'}}]->(d:Entity {{name: '<target code>'}})
   RETURN b.id AS from, c.id AS to, r2.relation AS relation
   UNION ALL
   MATCH (a:Entity {{name: '<source code>'}})-[r1:Relation {{relation: 'Refines'}}]->(b:Entity)
   WITH a, b, r1
   MATCH (b)-[r2:Relation {{relation: 'Refines'}}]->(c:Entity)
   WITH a, b, r1, c, r2
   MATCH (c)-[r3:Relation {{relation: 'Refines'}}]->(d:Entity {{name: '<target code>'}})
   RETURN c.id AS from, d.id AS to, r3.relation AS relation`
   (a 2-hop chain simply omits the third MATCH/WITH/branch above -- same
   rule, fewer repetitions.)

WATCH FOR: a question describing elements "within", "under", "belonging
to", or otherwise contained by some broader scope/section/structure is
describing MEMBERSHIP THAT MAY BE NESTED AT ANY DEPTH -- a sub-section's
own sub-section still counts as being "within" the top one, not just its
direct children. A single fixed-length hop (as in example 1) only reaches
direct children; example 2 above already shows the unbounded-depth form
(`-[r:Relation* (r, n | WHERE ...)]->`, no upper bound on hop count) for
exactly this situation. If a single-hop match would plausibly miss
elements nested more than one level deep, use that unbounded form
instead, not a fixed hop count.
"""


# Row format each query must return, per template.
RETURN_CONTRACT = {
    "impact_analysis.requirement_closure": (
        "Return ONE ROW PER matching element, with a column aliased `id`: e.g. `RETURN x.id AS id`. "
        "An empty result (zero rows) is a valid answer if you determine nothing matches."
    ),
    "impact_analysis.structural_closure": (
        "Return ONE ROW PER matching element, with a column aliased `id`: e.g. `RETURN x.id AS id`. "
        "An empty result (zero rows) is a valid answer if you determine nothing matches."
    ),
    "negation.requirement_unrefined": (
        "Return ONE ROW PER matching element, with a column aliased `id`: e.g. `RETURN x.id AS id`. "
        "An empty result (zero rows) is a valid answer if you determine nothing matches."
    ),
    "aggregation.requirement_count": (
        "Return EXACTLY ONE ROW with a column aliased `count`: e.g. `RETURN count(x) AS count`. "
        "A count-aggregate query always returns exactly one row, even if the count is 0."
    ),
    "tracing.requirement_derivation": (
        "Return ONE ROW PER HOP of your proposed derivation chain, IN ORDER from source to target, "
        "with three columns aliased `from`, `to`, `relation`: e.g. `RETURN a.id AS from, b.id AS to, "
        "r.relation AS relation`. If more than one hop, write one MATCH per hop and combine them with "
        "UNION ALL, each contributing one row, in chain order -- EVERY MATCH after the first one WITHIN "
        "THE SAME BRANCH needs its own preceding `WITH` re-stating every variable still needed, for AS "
        "MANY hops as the chain actually has (2 hops needs one WITH per branch, 3 hops needs two, and so "
        "on -- this does not stop applying after the first hop). See EXAMPLE QUERIES #5 above for the "
        "general pattern extended to a 3-hop chain (a bare MATCH with no preceding WITH is a Kuzu parser "
        "error, on any hop). An empty result (zero rows) is a valid answer if you determine no such "
        "chain exists."
    ),
    "rationale_abstention": (
        "Return EXACTLY ONE ROW with a column aliased `rationale` (the text, or null if none is "
        "recorded for this element)."
    ),
    "lookup.value_property_attribute": (
        "Return EXACTLY ONE ROW with a column aliased `value` (the requested attribute value, or "
        "null if none is recorded)."
    ),
}


def build_prompt(template_id, question_text, schema_prompt):
    return (
        f"{schema_prompt}\n\n"
        f"QUESTION: {question_text}\n\n"
        f'Respond with a JSON object: {{"cypher": "<your query, a single string>"}}. '
        f"{RETURN_CONTRACT[template_id]} Do not invent ids, names, or relation types not listed above."
    )


def build_repair_prompt(original_prompt, cypher, error):
    return (
        f"{original_prompt}\n\n"
        f"Your previous attempt:\n{cypher}\n\n"
        f"It failed with this error:\n{error}\n\n"
        f'Respond again with a corrected query, in the same JSON format: {{"cypher": "<corrected query>"}}.'
    )


def extract_cypher(response):
    if not isinstance(response, dict) or "cypher" not in response:
        return EXTRACTION_FAILED
    return response["cypher"]


def execute_with_repair(client, base_prompt, backend, max_attempts=3):
    """Generate and execute a query; on failure, ask the model to repair it. Returns (rows, meta)."""
    prompt = base_prompt
    last_cypher = None
    last_error = None
    for attempt in range(1, max_attempts + 1):
        response = client.complete_json(prompt, 0.0)
        cypher = extract_cypher(response)
        if cypher is EXTRACTION_FAILED:
            last_cypher = None
            last_error = "response had no 'cypher' key"
            prompt = build_repair_prompt(
                base_prompt, "(no query -- your response was missing the 'cypher' key)", last_error
            )
            continue
        last_cypher = cypher
        try:
            rows = backend.run_rows(cypher)
            return rows, {"attempts_used": attempt, "syntax_error": False, "final_cypher": cypher}
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            if "Interrupted" in str(e):
                repair_error = (
                    f"{last_error} -- your query took too long to execute and was forcibly stopped, most "
                    f"likely because it used an unbounded variable-length traversal (`-[r:Relation* (r, n | "
                    f"...)]->` with no upper hop limit) combined with a broad or high-degree relation set, "
                    f"causing a combinatorial blowup, NOT because the query's logic was necessarily wrong. "
                    f"If your intent genuinely requires unbounded depth, try adding an explicit upper bound "
                    f"(e.g. `*1..15`) or narrowing the relation set to only what the question actually needs; "
                    f"if a fixed number of hops is enough, use that instead."
                )
            else:
                repair_error = last_error
            prompt = build_repair_prompt(base_prompt, cypher, repair_error)
    return None, {
        "attempts_used": max_attempts,
        "syntax_error": True,
        "final_cypher": last_cypher,
        "final_error": last_error,
    }


def rows_to_prediction(template_id, rows):
    """Convert query result rows into the typed prediction expected by the metrics."""
    meta = {}
    if template_id in (
        "impact_analysis.requirement_closure",
        "impact_analysis.structural_closure",
        "negation.requirement_unrefined",
    ):
        if rows and "id" not in rows[0]:
            meta["missing_column"] = "id"
            return EXTRACTION_FAILED, meta
        return [r["id"] for r in rows], meta

    if template_id == "aggregation.requirement_count":
        if not rows or "count" not in rows[0]:
            meta["missing_column_or_no_rows"] = "count"
            return EXTRACTION_FAILED, meta
        try:
            return int(rows[0]["count"]), meta
        except (TypeError, ValueError):
            meta["unparseable_count"] = rows[0]["count"]
            return EXTRACTION_FAILED, meta

    if template_id == "tracing.requirement_derivation":
        if not rows:
            return None, meta
        if not all(k in rows[0] for k in ("from", "to", "relation")):
            meta["missing_column"] = "from/to/relation"
            return EXTRACTION_FAILED, meta
        return [{"from": r["from"], "to": r["to"], "relation": r["relation"]} for r in rows], meta

    if template_id == "rationale_abstention":
        if not rows:
            return None, meta
        if "rationale" not in rows[0]:
            meta["missing_column"] = "rationale"
            return EXTRACTION_FAILED, meta
        return rows[0]["rationale"], meta

    if template_id == "lookup.value_property_attribute":
        if not rows:
            return None, meta
        if "value" not in rows[0]:
            meta["missing_column"] = "value"
            return EXTRACTION_FAILED, meta
        return rows[0]["value"], meta

    raise ValueError(f"no row-mapping contract for template_id {template_id!r}")


def dry_run_cypher(template_id, record, raw_gold_query):
    """Fixed query per template used by --dry-run (no LLM call)."""
    b = record["bindings"]
    if template_id == "aggregation.requirement_count":
        return raw_gold_query["cypher"].replace(" AS n", " AS count").replace(" AS n\n", " AS count\n")
    if template_id == "rationale_abstention":
        return raw_gold_query
    if template_id == "lookup.value_property_attribute":
        return raw_gold_query.replace("AS default_value", "AS value")
    if template_id in ("impact_analysis.requirement_closure", "impact_analysis.structural_closure"):
        return f"MATCH (a:Entity {{id: '{b['X']}'}})-[:Relation*1..1]-(b:Entity) RETURN b.id AS id"
    if template_id == "negation.requirement_unrefined":
        return f"MATCH (t:Entity {{id: '{b['component_id']}'}}) RETURN t.id AS id"
    if template_id == "tracing.requirement_derivation":
        return (
            f"MATCH (a:Entity {{id: '{b['source']}'}})-[r:Relation]->(b:Entity {{id: '{b['target']}'}}) "
            f"RETURN a.id AS from, b.id AS to, r.relation AS relation"
        )
    raise ValueError(template_id)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap, "text_to_cypher")
    ap.add_argument("--candidates", default=str(paths.CANDIDATES_SAMPLED))
    ap.add_argument("--catalogue", default=str(paths.CATALOGUE))
    ap.add_argument("--max-attempts", type=int, default=3, help="query attempts including repairs")
    args = ap.parse_args()

    records = load_jsonl(args.input)
    if args.template_id:
        records = [r for r in records if r["template_id"] == args.template_id]
    raw_by_key = {record_key(r): r for r in load_jsonl(args.candidates)}
    config = load_yaml(args.model_config)
    backend = KuzuBackend(args.db_path)
    edge_exists_fn = build_edge_exists_fn(
        backend, list(config["trace_relations"].keys()), config["deprecated_stereotypes"], config["trace_relations"]
    )
    embed_fn, embed_provenance = get_default_embed_fn()

    schema_prompt = build_schema_prompt(args.catalogue, config)
    print(f"Schema prompt built: {len(schema_prompt)} chars.")

    client = build_baseline_client(args)

    def process_fn(record, question_text):
        template_id = record["template_id"]
        prompt = build_prompt(template_id, question_text, schema_prompt)

        if args.dry_run:
            raw_record = raw_by_key.get(record_key(record))
            cypher = dry_run_cypher(template_id, record, raw_record["gold_query"] if raw_record else None)
            try:
                rows = backend.run_rows(cypher)
                exec_meta = {"attempts_used": 1, "syntax_error": False, "final_cypher": cypher}
            except Exception as e:
                rows = None
                exec_meta = {
                    "attempts_used": 1,
                    "syntax_error": True,
                    "final_cypher": cypher,
                    "final_error": f"{type(e).__name__}: {e}",
                }
        else:
            rows, exec_meta = execute_with_repair(client, prompt, backend, max_attempts=args.max_attempts)

        if rows is None:
            predicted, row_meta = EXTRACTION_FAILED, {"reason": "exhausted repair attempts, never executed"}
        else:
            predicted, row_meta = rows_to_prediction(template_id, rows)

        result = score_instance(record, predicted, edge_exists_fn=edge_exists_fn, embed_fn=embed_fn)
        provenance = {"dry_run": args.dry_run} if args.dry_run else client.provenance()
        if template_id == "rationale_abstention":
            provenance["embedding"] = embed_provenance
        return {
            "template_id": template_id,
            "predicted": predicted,
            "raw_response": json.dumps(rows) if rows is not None else None,
            "extraction_meta": {**exec_meta, **row_meta},
            "scores": result,
            "provenance": provenance,
        }

    out_path = results_path(args, "text_to_cypher")
    if not args.dry_run:
        assert_consistent_model(out_path, client.provenance()["model"])

    for question_field in ("canonical_question", "question"):
        print(f"\n=== text-to-cypher, question_field={question_field} ===")
        run_checkpointed(records, question_field, process_fn, out_path, limit=args.limit)


if __name__ == "__main__":
    main()
