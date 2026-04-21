"""Working memory - session-scoped, fast-access storage."""

import json
import logging
import uuid
import psycopg2
from datetime import datetime
from typing import Optional, Dict, Any, List
from context_engine.config import ContextEngineConfig

logger = logging.getLogger(__name__)


class WorkingMemory:
    """
    Fast, session-scoped memory for temporary state.

    No embeddings, direct SQL access, TTL-based expiration.
    """

    SOFT_LIMIT = 100
    HARD_LIMIT = 200

    def __init__(self, config: Optional[ContextEngineConfig] = None):
        """Initialize working memory."""
        self.config = config or ContextEngineConfig()
        self._conn = None

    def _get_conn(self):
        """Get database connection with lazy initialization."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.config.conn_string)
        return self._conn

    def _check_size_limit(self):
        """Check and enforce size limits."""
        conn = self._get_conn()
        cur = conn.cursor()

        try:
            # Get current count
            cur.execute("SELECT COUNT(*) FROM working.session_context")
            count = cur.fetchone()[0]

            if count >= self.HARD_LIMIT:
                # Evict lowest priority + oldest
                cur.execute("""
                    DELETE FROM working.session_context
                    WHERE ctid IN (
                        SELECT ctid FROM working.session_context
                        ORDER BY priority ASC, last_accessed ASC
                        LIMIT %s
                    )
                """, (count - self.SOFT_LIMIT + 1,))
                conn.commit()
            elif count >= self.SOFT_LIMIT:
                # Just log warning
                logger.warning(f"Working memory at {count}/{self.HARD_LIMIT} items")
        finally:
            cur.close()

    def set_session_context(self, key: str, value: str,
                           priority: int = 5, ttl_minutes: int = 60) -> None:
        """Set session context with TTL."""
        self._check_size_limit()

        conn = self._get_conn()
        cur = conn.cursor()

        try:
            cur.execute("""
                INSERT INTO working.session_context
                (key, value, priority, ttl_minutes, last_accessed, created_at)
                VALUES (%s, %s, %s, %s, NOW(), NOW())
                ON CONFLICT (key) DO UPDATE SET
                    value = EXCLUDED.value,
                    priority = EXCLUDED.priority,
                    ttl_minutes = EXCLUDED.ttl_minutes,
                    last_accessed = NOW()
            """, (key, value, priority, ttl_minutes))
            conn.commit()
        finally:
            cur.close()

    def get_session_context(self) -> Dict[str, str]:
        """Get all session context key-values."""
        conn = self._get_conn()
        cur = conn.cursor()

        try:
            cur.execute("""
                SELECT key, value FROM working.session_context
                WHERE created_at + INTERVAL '1 minute' * ttl_minutes > NOW()
            """)
            rows = cur.fetchall()
            return {key: value for key, value in rows}
        finally:
            cur.close()

    def save_task(self, description: str, plan: Optional[List[str]] = None,
                  status: str = "planning", assigned_to: Optional[str] = None,
                  priority: int = 5, task_id: Optional[str] = None) -> str:
        """Save task, auto-generate ID if not provided."""
        if task_id is None:
            task_id = f"task-{uuid.uuid4().hex[:8]}"

        conn = self._get_conn()
        cur = conn.cursor()

        try:
            cur.execute("""
                INSERT INTO working.tasks
                (task_id, description, plan, status, assigned_to, priority, created_at, updated_at)
                VALUES (%s, %s, %s::jsonb, %s, %s, %s, NOW(), NOW())
                ON CONFLICT (task_id) DO UPDATE SET
                    description = EXCLUDED.description,
                    plan = EXCLUDED.plan,
                    status = EXCLUDED.status,
                    assigned_to = EXCLUDED.assigned_to,
                    priority = EXCLUDED.priority,
                    updated_at = NOW()
            """, (task_id, description, json.dumps(plan) if plan else None,
                  status, assigned_to, priority))
            conn.commit()
            return task_id
        finally:
            cur.close()

    def get_tasks(self, status: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get tasks, optionally filtered by status."""
        conn = self._get_conn()
        cur = conn.cursor()

        try:
            if status:
                cur.execute("""
                    SELECT task_id, description, plan, status, assigned_to, priority, result
                    FROM working.tasks WHERE status = %s LIMIT %s
                """, (status, limit))
            else:
                cur.execute("""
                    SELECT task_id, description, plan, status, assigned_to, priority, result
                    FROM working.tasks ORDER BY updated_at DESC LIMIT %s
                """, (limit,))

            rows = cur.fetchall()
            tasks = []
            for row in rows:
                task = {
                    "task_id": row[0],
                    "description": row[1],
                    "plan": row[2],
                    "status": row[3],
                    "assigned_to": row[4],
                    "priority": row[5],
                    "result": row[6]
                }
                tasks.append(task)
            return tasks
        finally:
            cur.close()

    def update_task(self, task_id: str, **kwargs) -> bool:
        """Update task fields."""
        allowed = {"description", "plan", "status", "assigned_to", "priority", "result"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}

        if not updates:
            return False

        conn = self._get_conn()
        cur = conn.cursor()

        try:
            # Convert plan and result to JSON if needed
            if "plan" in updates and updates["plan"] is not None:
                updates["plan"] = json.dumps(updates["plan"])
            if "result" in updates and updates["result"] is not None:
                updates["result"] = json.dumps(updates["result"])

            # Build dynamic UPDATE
            set_clause = ", ".join([f"{k} = %s" for k in updates.keys()])
            values = list(updates.values()) + [task_id]

            cur.execute(f"""
                UPDATE working.tasks
                SET {set_clause}, updated_at = NOW()
                WHERE task_id = %s
            """, values)

            updated = cur.rowcount > 0
            conn.commit()
            return updated
        finally:
            cur.close()

    def save_decision(self, content: str, context: Optional[str] = None,
                     category: str = "decision", ttl_minutes: int = 480) -> int:
        """Save decision with auto-expire. Returns decision ID."""
        conn = self._get_conn()
        cur = conn.cursor()

        try:
            cur.execute("""
                INSERT INTO working.recent_decisions
                (content, category, context, ttl_minutes, created_at)
                VALUES (%s, %s, %s, %s, NOW())
                RETURNING id
            """, (content, category, context, ttl_minutes))

            decision_id = cur.fetchone()[0]
            conn.commit()
            return decision_id
        finally:
            cur.close()

    def get_recent_decisions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent decisions."""
        conn = self._get_conn()
        cur = conn.cursor()

        try:
            cur.execute("""
                SELECT id, content, category, context, created_at
                FROM working.recent_decisions
                WHERE created_at + INTERVAL '1 minute' * ttl_minutes > NOW()
                ORDER BY created_at DESC
                LIMIT %s
            """, (limit,))

            rows = cur.fetchall()
            decisions = []
            for row in rows:
                decisions.append({
                    "id": row[0],
                    "content": row[1],
                    "category": row[2],
                    "context": row[3],
                    "created_at": row[4]
                })
            return decisions
        finally:
            cur.close()

    def cleanup_expired(self) -> int:
        """Delete expired entries from all working tables. Returns total count."""
        conn = self._get_conn()
        cur = conn.cursor()
        total = 0

        try:
            # Cleanup session_context
            cur.execute("""
                DELETE FROM working.session_context
                WHERE created_at + INTERVAL '1 minute' * ttl_minutes <= NOW()
            """)
            total += cur.rowcount

            # Cleanup recent_decisions
            cur.execute("""
                DELETE FROM working.recent_decisions
                WHERE created_at + INTERVAL '1 minute' * ttl_minutes <= NOW()
            """)
            total += cur.rowcount

            conn.commit()
            return total
        finally:
            cur.close()

    def close(self):
        """Close database connection."""
        if self._conn and not self._conn.closed:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
