import sqlite3


def get_conn():
    """
    Returns a SQLite database connection.
    Creates DB automatically if it doesn't exist.
    """

    conn = sqlite3.connect("resilience.db")

    conn.execute("""
    CREATE TABLE IF NOT EXISTS system_metrics(
        experiment_id TEXT,
        timestamp TEXT,
        cpu_usage REAL,
        memory_usage REAL,
        ready_pods INTEGER,
        total_pods INTEGER,
        pod_restarts INTEGER
    )
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS chaos_experiment(
        experiment_id TEXT PRIMARY KEY,
        experiment_type TEXT,
        service TEXT,
        namespace TEXT,
        target TEXT,
        blast_radius TEXT,
        start_time TEXT,
        end_time TEXT
    )
    """)

    return conn