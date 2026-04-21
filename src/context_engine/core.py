"""
PGVector Context Engine - Core semantic memory manager.

This is the main public API for the context engine.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import psycopg2
import psycopg2.extras
import requests
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union

logger = logging.getLogger(__name__)

from context_engine.config import ContextEngineConfig
from context_engine.providers import OllamaProvider, EmbeddingProvider, EmbeddingError


class ContextEngine:
    """
    Semantic context/memory engine using PostgreSQL + pgvector.

    Main features:
        - Store memories with automatic embedding
        - Semantic search via vector similarity
        - Project/namespace isolation
        - TTL-based expiration
        - Importance scoring
        - Category filtering

    Usage:
        ctx = ContextEngine()
        ctx.save("Deployed to k8s", category="infra")
        context = ctx.get_context("What was I working on?")
    """

    VALID_REL_TYPES = {
        "related_to", "depends_on", "supersedes", "about",
        "blocks", "references", "contains", "derived_from",
    }

    def __init__(
        self,
        config: Optional[ContextEngineConfig] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        auto_init: bool = True,
        cache_embeddings: bool = True,
    ):
        """
        Initialize the context engine.

        Args:
            config: Configuration object. If None, loads from env/config file.
            embedding_provider: Custom embedding provider. If None, uses Ollama.
            auto_init: If True, ensure schema exists on first use.
            cache_embeddings: If True, cache embedding results to avoid redundant
                calls to the embedding provider (default: True).
        """
        self.config = config or ContextEngineConfig()
        self.namespace = self.config.namespace

        # Use custom provider or default to Ollama
        if embedding_provider is not None:
            self._embedding = embedding_provider
        else:
            self._embedding = OllamaProvider(
                url=self.config.ollama_url,
                model=self.config.embedding_model,
            )

        self._conn = None
        self._auto_init = auto_init
        self._initialized = False

        # Embedding cache
        self._embedding_cache: Optional[OrderedDict] = OrderedDict() if cache_embeddings else None
        self._cache_lock = threading.Lock()
        self._cache_hits = 0
        self._cache_misses = 0

    def _get_conn(self):
        """Get database connection with lazy initialization."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.config.conn_string)
        return self._conn

    def _ensure_initialized(self):
        """Ensure database schema is ready (called on first operation)."""
        if self._initialized:
            return

        if self._auto_init:
            from context_engine.schema import SchemaManager
            schema = SchemaManager(self.config)
            schema.ensure_database_exists()
            schema.ensure_schema(run_migrations=True)

        self._initialized = True

    def _embed(self, text: str) -> List[float]:
        """Generate embedding for text, with optional caching."""
        # If caching is disabled, always call provider
        if self._embedding_cache is None:
            try:
                return self._embedding.embed(text)
            except EmbeddingError as e:
                logger.error(f"Embedding failed: {e}")
                return [0.0] * self._embedding.dimension

        with self._cache_lock:
            if text in self._embedding_cache:
                self._cache_hits += 1
                # Move to end to mark as recently used
                self._embedding_cache.move_to_end(text)
                return self._embedding_cache[text]

        # Call provider outside lock to avoid holding lock during I/O
        try:
            result = self._embedding.embed(text)
        except EmbeddingError as e:
            logger.error(f"Embedding failed: {e}")
            return [0.0] * self._embedding.dimension

        with self._cache_lock:
            # Check again in case another thread inserted while we were outside lock
            if text in self._embedding_cache:
                self._cache_hits += 1
                self._embedding_cache.move_to_end(text)
                return self._embedding_cache[text]

            self._cache_misses += 1
            self._embedding_cache[text] = result
            # Evict oldest entry if over max size
            while len(self._embedding_cache) > 128:
                self._embedding_cache.popitem(last=False)

        return result

    def clear_embedding_cache(self):
        """Clear the embedding cache and reset hit/miss counters."""
        with self._cache_lock:
            if self._embedding_cache is not None:
                self._embedding_cache.clear()
            self._cache_hits = 0
            self._cache_misses = 0

    @property
    def embedding_cache_stats(self) -> Dict[str, Any]:
        """
        Return statistics about the embedding cache.

        Returns:
            Dict with keys: hits, misses, size, enabled.
            If caching is disabled, returns zeros and enabled=False.
        """
        if self._embedding_cache is None:
            return {"hits": 0, "misses": 0, "size": 0, "enabled": False}
        with self._cache_lock:
            return {
                "hits": self._cache_hits,
                "misses": self._cache_misses,
                "size": len(self._embedding_cache),
                "enabled": True,
            }

    def save(
        self,
        content: str,
        category: str = "general",
        importance: float = 1.0,
        ttl_days: Optional[int] = None,
        session_key: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> str:
        """
        Save a memory with automatic embedding.

        Args:
            content: The memory content (min 10 chars)
            category: Category for filtering (default: "general")
            importance: 0.1-10.0 importance score (default: 1.0)
            ttl_days: Days until expiration (None = permanent)
            session_key: Optional session identifier
            tags: Optional list of tags
            metadata: Optional additional metadata
            source: Optional source identifier
            doc_id: Optional stable ID (auto-generated from content if not provided)

        Returns:
            The doc_id of the saved memory
        """
        self._ensure_initialized()

        if len(content.strip()) < 10:
            return ""

        # Generate doc_id if not provided
        if not doc_id:
            doc_id = hashlib.sha256(content.encode()).hexdigest()[:32]

        embedding = self._embed(content)

        expires_at = None
        if ttl_days:
            expires_at = datetime.now() + timedelta(days=ttl_days)

        conn = self._get_conn()
        cur = conn.cursor()

        metadata = metadata or {}
        metadata["saved_by"] = "context_engine"
        metadata["saved_at"] = datetime.now().isoformat()

        try:
            cur.execute("""
                INSERT INTO memories
                (doc_id, content, embedding, namespace, category, importance,
                 expires_at, session_key, tags, metadata, source, created_at)
                VALUES (
                    %s, %s, %s::vector, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
                )
                ON CONFLICT (doc_id) DO UPDATE SET
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    category = EXCLUDED.category,
                    importance = EXCLUDED.importance,
                    metadata = EXCLUDED.metadata,
                    updated_at = NOW()
                RETURNING id
            """, (
                doc_id, content, embedding, self.namespace, category,
                importance, expires_at, session_key, tags,
                psycopg2.extras.Json(metadata), source
            ))

            mem_id = cur.fetchone()[0]
            conn.commit()
            return doc_id

        except psycopg2.Error as e:
            conn.rollback()
            raise ContextEngineError(f"Failed to save memory: {e}") from e
        finally:
            cur.close()

    def get_context(
        self,
        query: str,
        max_memories: int = 10,
        max_tokens: int = 4000,
        category: Optional[str] = None,
        namespace: Optional[str] = None,
        precomputed_embedding: Optional[List[float]] = None,
    ) -> str:
        """
        Get relevant context for a query using semantic search.

        Args:
            query: Search query or current task description
            max_memories: Maximum number of memories to retrieve
            max_tokens: Approximate token budget for context
            category: Optional category filter
            namespace: Optional namespace override (defaults to self.namespace)
            precomputed_embedding: Optional pre-computed embedding vector. If
                provided, skips calling _embed() for the query.

        Returns:
            Formatted context string with relevant memories
        """
        self._ensure_initialized()

        ns = namespace or self.namespace
        if precomputed_embedding is not None:
            embedding = precomputed_embedding
        else:
            embedding = self._embed(query)

        conn = self._get_conn()
        cur = conn.cursor()

        try:
            # Build query with filters
            sql = """
                SELECT id, doc_id, content, category, source, created_at,
                       importance, 1 - (embedding <=> %s::vector) as similarity
                FROM memories
                WHERE namespace = %s
                  AND (expires_at IS NULL OR expires_at > NOW())
            """
            params = [embedding, ns]

            if category:
                sql += " AND category = %s"
                params.append(category)

            sql += " ORDER BY embedding <=> %s::vector LIMIT %s"
            params.extend([embedding, max_memories])

            cur.execute(sql, params)
            rows = cur.fetchall()

            if not rows:
                return ""

            # Build context within token budget
            memories = []
            total_chars = 0
            max_chars = int(max_tokens * 4)  # ~4 chars per token

            for row in rows:
                mem_id, doc_id, content, cat, source, created_at, importance, similarity = row

                if similarity < 0.5:
                    continue

                date_str = created_at.strftime("%Y-%m-%d") if created_at else "unknown"

                parts = [f"[{cat}]"]
                if source:
                    parts.append(f"@{source}")
                parts.append(f"({date_str})")
                parts.append(content)

                formatted = " ".join(parts)

                if len(formatted) + total_chars > max_chars:
                    break

                memories.append(formatted)
                total_chars += len(formatted) + 2

                # Update access tracking (fire and forget)
                self._update_access(mem_id)

            return "\n\n".join(memories)

        except psycopg2.Error as e:
            raise ContextEngineError(f"Failed to get context: {e}") from e
        finally:
            cur.close()

    def _update_access(self, memory_id: int):
        """Update access count and last accessed time."""
        try:
            conn = self._get_conn()
            cur = conn.cursor()
            cur.execute("""
                UPDATE memories
                SET access_count = access_count + 1,
                    last_accessed = NOW()
                WHERE id = %s
            """, (memory_id,))
            conn.commit()
            cur.close()
        except psycopg2.Error:
            pass  # Non-critical, ignore

    @staticmethod
    def _leanify_search(result: dict) -> dict:
        """Map search result keys to abbreviated lean keys."""
        mapping = {
            "content": "c",
            "category": "cat",
            "similarity": "s",
            "source": "src",
            "doc_id": "id",
        }
        return {lean_key: result[key] for key, lean_key in mapping.items() if key in result}

    @staticmethod
    def _leanify_list(result: dict) -> dict:
        """Map list result keys to abbreviated lean keys."""
        out = {}
        if "content" in result:
            out["c"] = result["content"]
        if "category" in result:
            out["cat"] = result["category"]
        if "created_at" in result:
            dt = result["created_at"]
            out["d"] = dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)[:10]
        if "doc_id" in result:
            out["id"] = result["doc_id"]
        return out

    @staticmethod
    def _clean_result(row: dict) -> dict:
        """Strip internal bookkeeping fields from a result dict."""
        # Strip saved_by and saved_at from metadata
        metadata = row.get("metadata")
        if isinstance(metadata, dict):
            metadata.pop("saved_by", None)
            metadata.pop("saved_at", None)
        # Strip tags: None
        if "tags" in row and row["tags"] is None:
            del row["tags"]
        return row

    def search(
        self,
        query: str,
        limit: int = 10,
        min_similarity: float = 0.5,
        category: Optional[str] = None,
        fields: str = "full",
    ) -> Union[List[Dict[str, Any]], List[str]]:
        """
        Search memories by semantic similarity.

        Args:
            query: Search query
            limit: Maximum results
            min_similarity: Minimum similarity threshold (0-1)
            category: Optional category filter
            fields: Result format - "full" (all keys), "lean" (abbreviated keys),
                    "ids" (list of doc_id strings only)

        Returns:
            List of matching memories. Type depends on `fields`:
            - "full": List[Dict[str, Any]] with all keys
            - "lean": List[Dict[str, Any]] with abbreviated keys
            - "ids": List[str] of doc_id strings
        """
        self._ensure_initialized()

        embedding = self._embed(query)

        conn = self._get_conn()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        try:
            sql = """
                SELECT doc_id, content, category, source, tags, metadata,
                       importance, created_at, access_count,
                       1 - (embedding <=> %s::vector) as similarity
                FROM memories
                WHERE namespace = %s
                  AND (expires_at IS NULL OR expires_at > NOW())
            """
            params = [embedding, self.namespace]

            if category:
                sql += " AND category = %s"
                params.append(category)

            sql += " ORDER BY embedding <=> %s::vector LIMIT %s"
            params.extend([embedding, limit])

            cur.execute(sql, params)
            rows = cur.fetchall()

            results = []
            for row in rows:
                row = dict(row)
                if row["similarity"] >= min_similarity:
                    row["similarity"] = round(row["similarity"], 4)
                    results.append(row)

            results = results[:limit]

            # Clean results (strip tags:None and metadata.saved_by/saved_at)
            results = [self._clean_result(r) for r in results]

            if fields == "ids":
                return [r["doc_id"] for r in results]
            elif fields == "lean":
                return [self._leanify_search(r) for r in results]
            return results

        except psycopg2.Error as e:
            raise ContextEngineError(f"Failed to search: {e}") from e
        finally:
            cur.close()

    def list(
        self,
        category: Optional[str] = None,
        limit: int = 50,
        since: Optional[datetime] = None,
        fields: str = "full",
    ) -> Union[List[Dict[str, Any]], List[str]]:
        """
        List memories with optional filtering.

        Args:
            category: Optional category filter
            limit: Maximum results
            since: Only show memories created after this time
            fields: Result format - "full" (all keys), "lean" (abbreviated keys),
                    "ids" (list of doc_id strings only)

        Returns:
            List of memories. Type depends on `fields`:
            - "full": List[Dict[str, Any]] with all keys
            - "lean": List[Dict[str, Any]] with abbreviated keys
            - "ids": List[str] of doc_id strings
        """
        self._ensure_initialized()

        conn = self._get_conn()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        sql = """
            SELECT doc_id, content, category, source, tags, metadata,
                   importance, created_at, access_count, expires_at
            FROM memories
            WHERE namespace = %s
              AND (expires_at IS NULL OR expires_at > NOW())
        """
        params: list = [self.namespace]

        if category:
            sql += " AND category = %s"
            params.append(category)

        if since:
            sql += " AND created_at > %s"
            params.append(since)

        sql += " ORDER BY created_at DESC LIMIT %s"
        params.append(limit)

        try:
            cur.execute(sql, params)
            rows = cur.fetchall()

            results = [self._clean_result(dict(row)) for row in rows]

            if fields == "ids":
                return [r["doc_id"] for r in results]
            elif fields == "lean":
                return [self._leanify_list(r) for r in results]
            return results

        except psycopg2.Error as e:
            raise ContextEngineError(f"Failed to list memories: {e}") from e
        finally:
            cur.close()

    def delete(self, doc_id: str) -> bool:
        """Delete a memory by doc_id."""
        self._ensure_initialized()

        conn = self._get_conn()
        cur = conn.cursor()

        cur.execute("""
            DELETE FROM memories
            WHERE doc_id = %s AND namespace = %s
        """, (doc_id, self.namespace))

        deleted = cur.rowcount > 0
        conn.commit()
        cur.close()

        return deleted

    def cleanup_expired(self) -> int:
        """Delete expired memories. Returns count deleted."""
        self._ensure_initialized()

        conn = self._get_conn()
        cur = conn.cursor()

        cur.execute("""
            DELETE FROM memories
            WHERE expires_at IS NOT NULL AND expires_at <= NOW()
        """)

        count = cur.rowcount
        conn.commit()
        cur.close()

        return count

    def save_conversation(
        self,
        session_key: str,
        user_message: str,
        assistant_response: str,
        category: str = "conversation",
    ) -> str:
        """
        Save a conversation turn and link to session.

        Args:
            session_key: Session identifier
            user_message: User's message
            assistant_response: Assistant's response
            category: Memory category

        Returns:
            doc_id of saved memory
        """
        content = f"User: {user_message}\nAssistant: {assistant_response}"
        return self.save(
            content=content,
            category=category,
            importance=1.0,
            ttl_days=30,
            session_key=session_key,
        )

    def get_session(self, session_key: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all memories for a specific session."""
        self._ensure_initialized()

        conn = self._get_conn()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        try:
            cur.execute("""
                SELECT doc_id, content, category, created_at, importance
                FROM memories
                WHERE session_key = %s AND namespace = %s
                ORDER BY created_at ASC
                LIMIT %s
            """, (session_key, self.namespace, limit))

            rows = cur.fetchall()

            return [dict(row) for row in rows]

        except psycopg2.Error as e:
            raise ContextEngineError(f"Failed to get session: {e}") from e
        finally:
            cur.close()

    def search_one(
        self,
        query: str,
        min_similarity: float = 0.3,
        category: Optional[str] = None,
    ) -> Optional[str]:
        """
        Return single best match content string, or None if no match above threshold.

        Args:
            query: Search query
            min_similarity: Minimum similarity threshold (default 0.3, lower than search())
            category: Optional category filter

        Returns:
            Content string of the best match, or None
        """
        results = self.search(
            query, limit=1, min_similarity=min_similarity,
            category=category, fields="full",
        )
        if results:
            return results[0].get("content")
        return None

    def relate(
        self,
        source_doc_id: str,
        target_doc_id: str,
        rel_type: str = "related_to",
    ) -> bool:
        """
        Create a relationship between two memories.

        Args:
            source_doc_id: doc_id of the source memory (must be in this namespace)
            target_doc_id: doc_id of the target memory (any namespace)
            rel_type: Relationship type, must be in VALID_REL_TYPES

        Returns:
            True if relationship was created, False if it already existed

        Raises:
            ValueError: If rel_type is invalid or source == target (self-reference)
            ContextEngineError: If source/target memory not found or DB error
        """
        self._ensure_initialized()

        if source_doc_id == target_doc_id:
            raise ValueError("Cannot create self-referencing relationship")

        if rel_type not in self.VALID_REL_TYPES:
            raise ValueError(
                f"Invalid rel_type '{rel_type}'. Must be one of: {', '.join(sorted(self.VALID_REL_TYPES))}"
            )

        conn = self._get_conn()
        cur = conn.cursor()

        try:
            # Look up source memory by doc_id + namespace
            cur.execute(
                "SELECT id FROM memories WHERE doc_id = %s AND namespace = %s",
                (source_doc_id, self.namespace),
            )
            source_row = cur.fetchone()
            if source_row is None:
                raise ContextEngineError(f"Source memory '{source_doc_id}' not found in namespace '{self.namespace}'")
            source_id = source_row[0]

            # Look up target memory by doc_id (any namespace)
            cur.execute(
                "SELECT id FROM memories WHERE doc_id = %s",
                (target_doc_id,),
            )
            target_row = cur.fetchone()
            if target_row is None:
                raise ContextEngineError(f"Target memory '{target_doc_id}' not found")
            target_id = target_row[0]

            # Insert relationship, ON CONFLICT DO NOTHING
            cur.execute(
                """
                INSERT INTO relationships (source_id, target_id, rel_type, created_at)
                VALUES (%s, %s, %s, NOW())
                ON CONFLICT (source_id, target_id, rel_type) DO NOTHING
                RETURNING id
                """,
                (source_id, target_id, rel_type),
            )
            result = cur.fetchone()
            conn.commit()

            return result is not None

        except psycopg2.Error as e:
            conn.rollback()
            raise ContextEngineError(f"Failed to create relationship: {e}") from e
        finally:
            cur.close()

    def unrelate(
        self,
        source_doc_id: str,
        target_doc_id: str,
        rel_type: Optional[str] = None,
    ) -> bool:
        """
        Remove a relationship between two memories.

        Args:
            source_doc_id: doc_id of the source memory (must be in this namespace)
            target_doc_id: doc_id of the target memory (any namespace)
            rel_type: If specified, only remove this type. If None, remove all types.

        Returns:
            True if any relationships were removed, False otherwise

        Raises:
            ContextEngineError: If source/target memory not found or DB error
        """
        self._ensure_initialized()

        conn = self._get_conn()
        cur = conn.cursor()

        try:
            # Look up source memory by doc_id + namespace
            cur.execute(
                "SELECT id FROM memories WHERE doc_id = %s AND namespace = %s",
                (source_doc_id, self.namespace),
            )
            source_row = cur.fetchone()
            if source_row is None:
                raise ContextEngineError(f"Source memory '{source_doc_id}' not found in namespace '{self.namespace}'")
            source_id = source_row[0]

            # Look up target memory by doc_id (any namespace)
            cur.execute(
                "SELECT id FROM memories WHERE doc_id = %s",
                (target_doc_id,),
            )
            target_row = cur.fetchone()
            if target_row is None:
                raise ContextEngineError(f"Target memory '{target_doc_id}' not found")
            target_id = target_row[0]

            # Delete relationship(s)
            if rel_type is not None:
                cur.execute(
                    """
                    DELETE FROM relationships
                    WHERE source_id = %s AND target_id = %s AND rel_type = %s
                    """,
                    (source_id, target_id, rel_type),
                )
            else:
                cur.execute(
                    """
                    DELETE FROM relationships
                    WHERE source_id = %s AND target_id = %s
                    """,
                    (source_id, target_id),
                )

            deleted = cur.rowcount > 0
            conn.commit()

            return deleted

        except psycopg2.Error as e:
            conn.rollback()
            raise ContextEngineError(f"Failed to remove relationship: {e}") from e
        finally:
            cur.close()

    def relations(
        self,
        doc_id: str,
        direction: str = "both",
        rel_type: Optional[str] = None,
    ) -> List[Dict]:
        """
        Get relationships for a memory.

        Args:
            doc_id: doc_id of the memory (must be in this namespace)
            direction: 'outgoing', 'incoming', or 'both'
            rel_type: Optional filter by relationship type

        Returns:
            List of dicts with keys: doc_id, content, category, rel_type, direction

        Raises:
            ContextEngineError: If memory not found or DB error
        """
        self._ensure_initialized()

        conn = self._get_conn()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        try:
            # Look up memory by doc_id + namespace
            cur.execute(
                "SELECT id FROM memories WHERE doc_id = %s AND namespace = %s",
                (doc_id, self.namespace),
            )
            mem_row = cur.fetchone()
            if mem_row is None:
                raise ContextEngineError(f"Memory '{doc_id}' not found in namespace '{self.namespace}'")
            mem_id = mem_row['id']

            results = []

            if direction in ("outgoing", "both"):
                sql = """
                    SELECT m.doc_id, m.content, m.category, r.rel_type
                    FROM relationships r
                    JOIN memories m ON r.target_id = m.id
                    WHERE r.source_id = %s
                """
                params: list = [mem_id]
                if rel_type:
                    sql += " AND r.rel_type = %s"
                    params.append(rel_type)
                cur.execute(sql, params)
                for row in cur.fetchall():
                    r = dict(row)
                    r['content'] = r['content'][:200]
                    r['direction'] = 'outgoing'
                    results.append(r)

            if direction in ("incoming", "both"):
                sql = """
                    SELECT m.doc_id, m.content, m.category, r.rel_type
                    FROM relationships r
                    JOIN memories m ON r.source_id = m.id
                    WHERE r.target_id = %s
                """
                params = [mem_id]
                if rel_type:
                    sql += " AND r.rel_type = %s"
                    params.append(rel_type)
                cur.execute(sql, params)
                for row in cur.fetchall():
                    r = dict(row)
                    r['content'] = r['content'][:200]
                    r['direction'] = 'incoming'
                    results.append(r)

            return results

        except psycopg2.Error as e:
            raise ContextEngineError(f"Failed to get relations: {e}") from e
        finally:
            cur.close()

    def recall(
        self,
        query: str,
        limit: int = 5,
        min_similarity: float = 0.3,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Quick recall using lean fields and lower default similarity.
        Optimized for "give me something relevant fast".

        Args:
            query: Search query
            limit: Maximum results
            min_similarity: Minimum similarity threshold (default 0.3)
            category: Optional category filter

        Returns:
            List of matching memories with lean/abbreviated keys
        """
        return self.search(
            query, limit=limit, min_similarity=min_similarity,
            category=category, fields="lean",
        )

    def peek(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Return full single memory content by doc_id.

        Args:
            doc_id: The document ID to look up

        Returns:
            Full dict of the memory, or None if not found
        """
        self._ensure_initialized()

        conn = self._get_conn()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        try:
            cur.execute("""
                SELECT doc_id, content, category, source, tags, metadata,
                       importance, created_at
                FROM memories
                WHERE doc_id = %s AND namespace = %s
            """, (doc_id, self.namespace))

            row = cur.fetchone()

            if row is None:
                return None

            return self._clean_result(dict(row))

        except psycopg2.Error as e:
            raise ContextEngineError(f"Failed to peek: {e}") from e
        finally:
            cur.close()

    def count(self, category: Optional[str] = None) -> int:
        """
        Return count of memories, with optional category filter.

        Args:
            category: Optional category filter

        Returns:
            Integer count of memories
        """
        self._ensure_initialized()

        conn = self._get_conn()
        cur = conn.cursor()

        try:
            sql = """
                SELECT COUNT(*) FROM memories
                WHERE namespace = %s
                  AND (expires_at IS NULL OR expires_at > NOW())
            """
            params = [self.namespace]

            if category:
                sql += " AND category = %s"
                params.append(category)

            cur.execute(sql, params)
            result = cur.fetchone()

            return result[0]

        except psycopg2.Error as e:
            raise ContextEngineError(f"Failed to count memories: {e}") from e
        finally:
            cur.close()

    def stats(self) -> Dict[str, Any]:
        """
        Return statistics about memories in this namespace.

        Returns:
            Dict with keys: count, categories (dict of name:count),
            size_estimate_kb, last_saved
        """
        self._ensure_initialized()

        conn = self._get_conn()
        cur = conn.cursor()

        try:
            # Total count
            cur.execute("""
                SELECT COUNT(*) FROM memories
                WHERE namespace = %s
                  AND (expires_at IS NULL OR expires_at > NOW())
            """, (self.namespace,))
            total_count = cur.fetchone()[0]

            # Category breakdown
            cur.execute("""
                SELECT category, COUNT(*) FROM memories
                WHERE namespace = %s
                  AND (expires_at IS NULL OR expires_at > NOW())
                GROUP BY category
            """, (self.namespace,))
            categories = {row[0]: row[1] for row in cur.fetchall()}

            # Size estimate
            cur.execute("""
                SELECT COALESCE(LENGTH(content)::bigint, 0) FROM memories
                WHERE namespace = %s
                  AND (expires_at IS NULL OR expires_at > NOW())
            """, (self.namespace,))
            total_bytes = sum(row[0] for row in cur.fetchall())
            size_estimate_kb = round(total_bytes / 1024, 1)

            # Last saved
            cur.execute("""
                SELECT MAX(created_at) FROM memories
                WHERE namespace = %s
                  AND (expires_at IS NULL OR expires_at > NOW())
            """, (self.namespace,))
            last_saved_row = cur.fetchone()
            last_saved = last_saved_row[0].isoformat() if last_saved_row[0] else None

            return {
                "count": total_count,
                "categories": categories,
                "size_estimate_kb": size_estimate_kb,
                "last_saved": last_saved,
            }

        except psycopg2.Error as e:
            raise ContextEngineError(f"Failed to get stats: {e}") from e
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


class ContextEngineError(Exception):
    """Raised when a context engine operation fails."""
    pass
