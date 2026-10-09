"""Thin wrapper around a Kuzu database connection."""

import kuzu

DEFAULT_QUERY_TIMEOUT_MS = 15000


class KuzuBackend:
    def __init__(self, db_path, query_timeout_ms: int = DEFAULT_QUERY_TIMEOUT_MS):
        self.db = kuzu.Database(str(db_path))
        self.conn = kuzu.Connection(self.db)
        self.conn.set_query_timeout(query_timeout_ms)

    def run(self, cypher: str, params: dict = None):
        return self.conn.execute(cypher, params or {})

    def run_rows(self, cypher: str, params: dict = None) -> list:
        """Execute a Cypher query and return the rows as dicts."""
        res = self.run(cypher, params)
        cols = res.get_column_names()
        rows = []
        while res.has_next():
            rows.append(dict(zip(cols, res.get_next())))
        return rows
