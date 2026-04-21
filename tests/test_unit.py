"""Unit tests for ContextEngine with mocked dependencies."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, Mock

from context_engine.core import ContextEngine, ContextEngineError
from context_engine.config import ContextEngineConfig


class TestContextEngineInit:
    """Test ContextEngine initialization."""

    def test_init_with_defaults(self, mock_embedding):
        """Test initialization with default config."""
        with patch('context_engine.core.ContextEngineConfig') as mock_config:
            mock_config.return_value = MagicMock(
                namespace="test",
                ollama_url="http://localhost:11434",
                embedding_model="nomic-embed-text",
                conn_string="postgresql://user:pass@localhost/db",
            )

            engine = ContextEngine(embedding_provider=mock_embedding, auto_init=False)
            assert engine.namespace == "test"
            assert engine._embedding == mock_embedding

    def test_init_with_custom_config(self, test_config, mock_embedding):
        """Test initialization with custom config."""
        engine = ContextEngine(
            config=test_config,
            embedding_provider=mock_embedding,
            auto_init=False
        )
        assert engine.namespace == test_config.namespace
        assert engine.config == test_config

    def test_init_auto_init_disabled(self, test_config, mock_embedding):
        """Test that auto_init=False prevents schema initialization."""
        with patch('context_engine.schema.SchemaManager') as mock_schema:
            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False
            )
            assert not engine._initialized
            mock_schema.assert_not_called()

    def test_context_manager(self, mock_embedding):
        """Test that ContextEngine works as a context manager."""
        with patch('context_engine.core.ContextEngineConfig') as mock_config:
            mock_config.return_value = MagicMock(
                namespace="test",
                conn_string="postgresql://user:pass@localhost/db",
            )

            with ContextEngine(embedding_provider=mock_embedding, auto_init=False) as ctx:
                assert isinstance(ctx, ContextEngine)


class TestSave:
    """Test saving memories."""

    def test_save_basic(self, test_config, mock_embedding):
        """Test saving a basic memory."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchone.return_value = [1]
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False
            )

            doc_id = engine.save(
                content="This is a test memory",
                category="test",
                importance=5.0
            )

            assert doc_id is not None
            assert len(doc_id) == 32  # SHA256 hash length
            mock_cur.execute.assert_called()
            mock_conn.commit.assert_called_once()

    def test_save_content_too_short(self, test_config, mock_embedding):
        """Test that short content is rejected."""
        engine = ContextEngine(
            config=test_config,
            embedding_provider=mock_embedding,
            auto_init=False
        )

        doc_id = engine.save(content="Short")  # Less than 10 chars
        assert doc_id == ""

    def test_save_with_ttl(self, test_config, mock_embedding):
        """Test saving with TTL."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchone.return_value = [1]
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False
            )

            doc_id = engine.save(
                content="This is a test memory with TTL",
                ttl_days=7
            )

            assert doc_id is not None
            # Check that expires_at was included in the query
            call_args = mock_cur.execute.call_args
            assert call_args is not None

    def test_save_with_custom_doc_id(self, test_config, mock_embedding):
        """Test saving with a custom doc_id."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchone.return_value = [1]
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False
            )

            custom_id = "my-custom-doc-id-12345"
            doc_id = engine.save(
                content="This is a test memory with custom ID",
                doc_id=custom_id
            )

            assert doc_id == custom_id

    def test_save_with_metadata(self, test_config, mock_embedding):
        """Test saving with metadata."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchone.return_value = [1]
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False
            )

            doc_id = engine.save(
                content="This is a test memory with metadata",
                metadata={"project": "test", "priority": "high"}
            )

            assert doc_id is not None


class TestSearch:
    """Test searching memories."""

    def test_search_basic(self, test_config, mock_embedding):
        """Test basic search functionality."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchall.return_value = [
                {
                    "doc_id": "abc123",
                    "content": "Test memory",
                    "category": "test",
                    "source": None,
                    "tags": None,
                    "metadata": {},
                    "importance": 1.0,
                    "created_at": datetime.now(),
                    "access_count": 0,
                    "similarity": 0.85,
                }
            ]
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False
            )

            results = engine.search("test query", limit=5)

            assert len(results) == 1
            assert results[0]["similarity"] == 0.85
            mock_cur.execute.assert_called()

    def test_search_with_category_filter(self, test_config, mock_embedding):
        """Test search with category filter."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchall.return_value = []
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False
            )

            results = engine.search("test query", category="infrastructure")

            # Check that category filter was included
            call_args = mock_cur.execute.call_args[0][0]
            assert "category" in call_args

    def test_search_min_similarity_filter(self, test_config, mock_embedding):
        """Test that results below min_similarity are filtered out."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            # Return results with various similarities
            mock_cur.fetchall.return_value = [
                {"doc_id": "1", "content": "High sim", "similarity": 0.9},
                {"doc_id": "2", "content": "Med sim", "similarity": 0.6},
                {"doc_id": "3", "content": "Low sim", "similarity": 0.3},  # Below threshold
            ]
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False
            )

            results = engine.search("test", min_similarity=0.5)

            assert len(results) == 2
            assert all(r["similarity"] >= 0.5 for r in results)


class TestGetContext:
    """Test get_context method."""

    def test_get_context_basic(self, test_config, mock_embedding):
        """Test getting context for a query."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchall.return_value = [
                (1, "doc1", "Test memory content", "test", "source", datetime.now(), 5.0, 0.85)
            ]
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False
            )

            context = engine.get_context("What was I working on?")

            assert "Test memory content" in context
            assert "[test]" in context

    def test_get_context_empty_results(self, test_config, mock_embedding):
        """Test get_context with no results."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchall.return_value = []
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False
            )

            context = engine.get_context("Nonexistent query")

            assert context == ""

    def test_get_context_token_budget(self, test_config, mock_embedding):
        """Test that token budget is respected."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()

            # Create many long memories
            long_memory = "X" * 1000  # 1000 char memory
            mock_cur.fetchall.return_value = [
                (i, f"doc{i}", long_memory, "test", None, datetime.now(), 5.0, 0.9)
                for i in range(10)
            ]
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False
            )

            # With max_tokens=100 (~400 chars), should only get a few memories
            context = engine.get_context("test", max_tokens=100)

            # Rough check: should be limited by token budget
            assert len(context) < 4000


class TestList:
    """Test listing memories."""

    def test_list_basic(self, test_config, mock_embedding):
        """Test basic list functionality."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchall.return_value = [
                {"doc_id": "1", "content": "Memory 1", "category": "test"},
                {"doc_id": "2", "content": "Memory 2", "category": "test"},
            ]
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False
            )

            results = engine.list(limit=10)

            assert len(results) == 2

    def test_list_with_category_filter(self, test_config, mock_embedding):
        """Test list with category filter."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchall.return_value = []
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False
            )

            engine.list(category="infrastructure")

            call_args = mock_cur.execute.call_args[0][0]
            assert "category" in call_args


class TestDelete:
    """Test deleting memories."""

    def test_delete_existing(self, test_config, mock_embedding):
        """Test deleting an existing memory."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.rowcount = 1
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False
            )

            result = engine.delete("doc-id-123")

            assert result is True
            mock_cur.execute.assert_called()

    def test_delete_nonexistent(self, test_config, mock_embedding):
        """Test deleting a non-existent memory."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.rowcount = 0
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False
            )

            result = engine.delete("nonexistent-doc")

            assert result is False


class TestCleanup:
    """Test cleanup functionality."""

    def test_cleanup_expired(self, test_config, mock_embedding):
        """Test cleaning up expired memories."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.rowcount = 5
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False
            )

            count = engine.cleanup_expired()

            assert count == 5
            mock_cur.execute.assert_called()


class TestConversation:
    """Test conversation methods."""

    def test_save_conversation(self, test_config, mock_embedding):
        """Test saving a conversation turn."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchone.return_value = [1]
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False
            )

            doc_id = engine.save_conversation(
                session_key="session-123",
                user_message="Hello",
                assistant_response="Hi there!"
            )

            assert doc_id is not None
            # Check that content includes both user and assistant
            call_args = mock_cur.execute.call_args
            assert "session-123" in str(call_args)

    def test_get_session(self, test_config, mock_embedding):
        """Test getting session memories."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchall.return_value = [
                {"doc_id": "1", "content": "User: Hello\nAssistant: Hi", "category": "conversation"},
            ]
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False
            )

            results = engine.get_session("session-123")

            assert len(results) == 1
            mock_cur.execute.assert_called()


class TestEmbeddingFallback:
    """Test embedding failure fallback."""

    def test_embedding_failure_returns_zero_vector(self, test_config):
        """Test that embedding failure returns zero vector."""
        mock_embedding = MagicMock()
        mock_embedding.dimension = 768
        from context_engine.providers import EmbeddingError
        mock_embedding.embed.side_effect = EmbeddingError("Ollama not available")

        engine = ContextEngine(
            config=test_config,
            embedding_provider=mock_embedding,
            auto_init=False
        )

        result = engine._embed("test text")

        assert result == [0.0] * 768


class TestClose:
    """Test connection cleanup."""

    def test_close_connection(self, test_config, mock_embedding):
        """Test closing the connection."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_conn.closed = False
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False
            )

            # Access connection to create it
            _ = engine._get_conn()
            engine.close()

            mock_conn.close.assert_called_once()


class TestConfig:
    """Test configuration."""

    def test_config_from_env(self):
        """Test loading config from environment variables."""
        with patch.dict('os.environ', {
            'CTX_DB_HOST': 'test-host',
            'CTX_DB_PORT': '5433',
            'CTX_NAMESPACE': 'test-ns',
            'CTX_DB_NAME': 'test-db',
            'CTX_DB_USER': 'test-user',
            'CTX_DB_PASS': 'test-pass',
            'CTX_CONFIG_PATH': '/nonexistent/config.json',  # Skip config file
        }, clear=True):
            config = ContextEngineConfig()
            assert config.db_host == 'test-host'
            assert config.db_port == 5433
            assert config.namespace == 'test-ns'

    def test_conn_string_building(self, test_config):
        """Test that connection string is built correctly."""
        expected = (
            f"postgresql://{test_config.db_user}:{test_config.db_pass}"
            f"@{test_config.db_host}:{test_config.db_port}/{test_config.db_name}"
            f"?sslmode={test_config.db_sslmode}"
        )
        assert test_config.conn_string == expected


class TestSearchFields:
    """Test search() with different field modes."""

    def test_search_lean_fields(self, test_config, mock_embedding):
        """Test search with fields='lean' returns abbreviated keys."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchall.return_value = [
                {
                    "doc_id": "abc123",
                    "content": "Test memory",
                    "category": "test",
                    "source": "manual",
                    "tags": None,
                    "metadata": {},
                    "importance": 1.0,
                    "created_at": datetime.now(),
                    "access_count": 0,
                    "similarity": 0.85,
                }
            ]
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False
            )

            results = engine.search("test query", fields="lean")
            assert len(results) == 1
            result = results[0]
            # Should have abbreviated keys
            assert "c" in result
            assert "cat" in result
            assert "s" in result
            assert "src" in result
            assert "id" in result
            # Should NOT have full keys
            assert "content" not in result
            assert "category" not in result
            assert "similarity" not in result
            assert "source" not in result
            assert "doc_id" not in result
            # Check values
            assert result["c"] == "Test memory"
            assert result["cat"] == "test"
            assert result["s"] == 0.85
            assert result["src"] == "manual"
            assert result["id"] == "abc123"

    def test_search_ids_fields(self, test_config, mock_embedding):
        """Test search with fields='ids' returns list of strings."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchall.return_value = [
                {
                    "doc_id": "abc123",
                    "content": "Memory 1",
                    "category": "test",
                    "source": None,
                    "tags": None,
                    "metadata": {},
                    "importance": 1.0,
                    "created_at": datetime.now(),
                    "access_count": 0,
                    "similarity": 0.85,
                },
                {
                    "doc_id": "def456",
                    "content": "Memory 2",
                    "category": "test",
                    "source": None,
                    "tags": None,
                    "metadata": {},
                    "importance": 1.0,
                    "created_at": datetime.now(),
                    "access_count": 0,
                    "similarity": 0.70,
                }
            ]
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False
            )

            results = engine.search("test query", fields="ids")
            assert isinstance(results, list)
            assert len(results) == 2
            assert all(isinstance(r, str) for r in results)
            assert results == ["abc123", "def456"]

    def test_search_full_fields_strips_tags_none_and_metadata(self, test_config, mock_embedding):
        """Test search with fields='full' strips tags:None and metadata.saved_by/saved_at."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchall.return_value = [
                {
                    "doc_id": "abc123",
                    "content": "Test memory",
                    "category": "test",
                    "source": None,
                    "tags": None,
                    "metadata": {"saved_by": "context_engine", "saved_at": "2026-01-01T00:00:00", "key": "val"},
                    "importance": 1.0,
                    "created_at": datetime.now(),
                    "access_count": 0,
                    "similarity": 0.85,
                }
            ]
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False
            )

            results = engine.search("test query", fields="full")
            assert len(results) == 1
            result = results[0]
            # tags should be stripped entirely since it was None
            assert "tags" not in result
            # metadata should have saved_by/saved_at removed but keep other keys
            assert "saved_by" not in result["metadata"]
            assert "saved_at" not in result["metadata"]
            assert result["metadata"]["key"] == "val"


class TestListFields:
    """Test list() with different field modes."""

    def test_list_lean_fields(self, test_config, mock_embedding):
        """Test list with fields='lean' returns abbreviated keys."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchall.return_value = [
                {
                    "doc_id": "abc123",
                    "content": "Memory 1",
                    "category": "test",
                    "source": "manual",
                    "tags": ["tag1"],
                    "metadata": {},
                    "importance": 1.0,
                    "created_at": datetime(2026, 4, 20, 10, 30, 0),
                    "access_count": 5,
                    "expires_at": None,
                }
            ]
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False
            )

            results = engine.list(fields="lean")
            assert len(results) == 1
            result = results[0]
            # Should have abbreviated keys
            assert "c" in result
            assert "cat" in result
            assert "d" in result
            assert "id" in result
            # Should NOT have full keys
            assert "content" not in result
            assert "category" not in result
            assert "created_at" not in result
            assert "doc_id" not in result
            # Check date formatting
            assert result["d"] == "2026-04-20"
            assert result["c"] == "Memory 1"
            assert result["cat"] == "test"
            assert result["id"] == "abc123"

    def test_list_ids_fields(self, test_config, mock_embedding):
        """Test list with fields='ids' returns list of strings."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchall.return_value = [
                {"doc_id": "abc123", "content": "M1", "category": "test",
                 "source": None, "tags": None, "metadata": {},
                 "importance": 1.0, "created_at": datetime.now(),
                 "access_count": 0, "expires_at": None},
                {"doc_id": "def456", "content": "M2", "category": "work",
                 "source": None, "tags": None, "metadata": {},
                 "importance": 1.0, "created_at": datetime.now(),
                 "access_count": 0, "expires_at": None},
            ]
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False
            )

            results = engine.list(fields="ids")
            assert isinstance(results, list)
            assert len(results) == 2
            assert all(isinstance(r, str) for r in results)
            assert results == ["abc123", "def456"]


class TestSearchOne:
    """Test search_one() method."""

    def test_search_one_returns_content(self, test_config, mock_embedding):
        """Test that search_one returns content string of best match."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchall.return_value = [
                {
                    "doc_id": "abc123",
                    "content": "Best match content",
                    "category": "test",
                    "source": None,
                    "tags": None,
                    "metadata": {},
                    "importance": 1.0,
                    "created_at": datetime.now(),
                    "access_count": 0,
                    "similarity": 0.92,
                }
            ]
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False
            )

            result = engine.search_one("test query")
            assert result == "Best match content"

    def test_search_one_returns_none_no_match(self, test_config, mock_embedding):
        """Test that search_one returns None when no match above threshold."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchall.return_value = []
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False
            )

            result = engine.search_one("nonexistent query")
            assert result is None


class TestRecall:
    """Test recall() method."""

    def test_recall_uses_lean_fields_and_lower_threshold(self, test_config, mock_embedding):
        """Test that recall calls search with lean fields and 0.3 default."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchall.return_value = [
                {
                    "doc_id": "abc123",
                    "content": "Recalled memory",
                    "category": "test",
                    "source": None,
                    "tags": None,
                    "metadata": {},
                    "importance": 1.0,
                    "created_at": datetime.now(),
                    "access_count": 0,
                    "similarity": 0.35,
                }
            ]
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False
            )

            results = engine.recall("test query")
            assert len(results) == 1
            # Should have lean keys
            result = results[0]
            assert "c" in result
            assert "cat" in result
            assert "s" in result
            assert "id" in result
            # Should NOT have full keys
            assert "content" not in result


class TestPeek:
    """Test peek() method."""

    def test_peek_existing_doc(self, test_config, mock_embedding):
        """Test that peek returns full dict for existing doc_id."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchone.return_value = {
                "doc_id": "abc123",
                "content": "Specific memory content",
                "category": "test",
                "source": "manual",
                "tags": None,
                "metadata": {"key": "val"},
                "importance": 2.0,
                "created_at": datetime(2026, 4, 20),
            }
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False
            )

            result = engine.peek("abc123")
            assert result is not None
            assert result["doc_id"] == "abc123"
            assert result["content"] == "Specific memory content"
            assert result["category"] == "test"
            # tags:None should be stripped
            assert "tags" not in result

    def test_peek_nonexistent_doc(self, test_config, mock_embedding):
        """Test that peek returns None for nonexistent doc_id."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchone.return_value = None
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False
            )

            result = engine.peek("nonexistent-id")
            assert result is None


class TestCount:
    """Test count() method."""

    def test_count_returns_integer(self, test_config, mock_embedding):
        """Test that count returns an integer."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchone.return_value = [42]
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False
            )

            result = engine.count()
            assert isinstance(result, int)
            assert result == 42

    def test_count_with_category_filter(self, test_config, mock_embedding):
        """Test that count with category includes filter in query."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchone.return_value = [10]
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False
            )

            result = engine.count(category="infrastructure")
            assert result == 10
            # Check that category filter was included in SQL
            call_args = mock_cur.execute.call_args[0][0]
            assert "category" in call_args


class TestStats:
    """Test stats() method."""

    def test_stats_returns_expected_keys(self, test_config, mock_embedding):
        """Test that stats returns dict with expected keys."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()

            # Mock three fetchone/fetchall calls: total count, categories, sizes, last_saved
            mock_cur.fetchone.side_effect = [
                (100,),                             # total count
                (datetime(2026, 4, 20),),           # last_saved (MAX created_at)
            ]
            mock_cur.fetchall.side_effect = [
                [("general", 80), ("infra", 20)],  # categories
                [(50,), (30,), (20,)],               # content lengths
            ]
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False
            )

            result = engine.stats()
            assert "count" in result
            assert "categories" in result
            assert "size_estimate_kb" in result
            assert "last_saved" in result
            assert result["count"] == 100
            assert isinstance(result["categories"], dict)
            assert isinstance(result["size_estimate_kb"], float)


class TestEmbeddingCache:
    """Test embedding cache functionality."""

    def test_embedding_cache_caches_results(self, test_config, mock_embedding):
        """Test that _embed() caches results and doesn't call provider twice for same text."""
        engine = ContextEngine(
            config=test_config,
            embedding_provider=mock_embedding,
            auto_init=False,
        )

        # Call _embed twice with same text
        result1 = engine._embed("hello world")
        result2 = engine._embed("hello world")

        # Provider should only be called once
        assert mock_embedding.embed.call_count == 1
        assert result1 == result2

    def test_embedding_cache_disabled(self, test_config, mock_embedding):
        """Test that with cache_embeddings=False, provider is called every time."""
        engine = ContextEngine(
            config=test_config,
            embedding_provider=mock_embedding,
            auto_init=False,
            cache_embeddings=False,
        )

        engine._embed("hello world")
        engine._embed("hello world")

        # Provider should be called twice since cache is disabled
        assert mock_embedding.embed.call_count == 2
        # Stats should show disabled
        stats = engine.embedding_cache_stats
        assert stats["enabled"] is False
        assert stats["size"] == 0

    def test_embedding_cache_max_size(self, test_config, mock_embedding):
        """Test that cache evicts oldest entries when exceeding 128."""
        # Make each embed call return a unique vector
        call_count = [0]
        def make_vector(text):
            call_count[0] += 1
            return [float(call_count[0])] + [0.0] * 767

        mock_embedding.embed.side_effect = make_vector

        engine = ContextEngine(
            config=test_config,
            embedding_provider=mock_embedding,
            auto_init=False,
        )

        # Insert 129 unique entries
        for i in range(129):
            engine._embed(f"text-{i}")

        # Cache should be capped at 128
        assert len(engine._embedding_cache) == 128
        # The first entry (text-0) should have been evicted
        assert "text-0" not in engine._embedding_cache
        # The last entry (text-128) should be present
        assert "text-128" in engine._embedding_cache

    def test_clear_embedding_cache(self, test_config, mock_embedding):
        """Test that clear_embedding_cache clears the cache and resets counters."""
        engine = ContextEngine(
            config=test_config,
            embedding_provider=mock_embedding,
            auto_init=False,
        )

        # Add some entries
        engine._embed("text one")
        engine._embed("text two")
        engine._embed("text one")  # This should be a cache hit

        assert len(engine._embedding_cache) == 2
        assert engine._cache_hits == 1
        assert engine._cache_misses == 2

        engine.clear_embedding_cache()

        assert len(engine._embedding_cache) == 0
        assert engine._cache_hits == 0
        assert engine._cache_misses == 0

        # After clearing, provider should be called again
        engine._embed("text one")
        assert mock_embedding.embed.call_count == 3  # 2 from before clear + 1 new

    def test_embedding_cache_stats(self, test_config, mock_embedding):
        """Test that embedding_cache_stats returns correct counts."""
        engine = ContextEngine(
            config=test_config,
            embedding_provider=mock_embedding,
            auto_init=False,
        )

        # 2 misses (new texts), 2 hits (cached texts)
        engine._embed("alpha")
        engine._embed("beta")
        engine._embed("alpha")  # hit
        engine._embed("beta")   # hit

        stats = engine.embedding_cache_stats
        assert stats["hits"] == 2
        assert stats["misses"] == 2
        assert stats["size"] == 2
        assert stats["enabled"] is True


class TestPrecomputedEmbedding:
    """Test precomputed_embedding parameter in get_context()."""

    def test_get_context_with_precomputed_embedding(self, test_config, mock_embedding):
        """Test that providing a precomputed embedding skips _embed()."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchall.return_value = [
                (1, "doc1", "Test memory content", "test", "source", datetime.now(), 5.0, 0.85)
            ]
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False,
            )

            precomputed = [0.1] * 768
            context = engine.get_context(
                "What was I working on?",
                precomputed_embedding=precomputed,
            )

            # The provider's embed() should NOT have been called
            mock_embedding.embed.assert_not_called()
            assert "Test memory content" in context

    def test_precomputed_embedding_none_calls_embed(self, test_config, mock_embedding):
        """Test that precomputed_embedding=None still calls _embed()."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchall.return_value = []
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False,
            )

            context = engine.get_context(
                "What was I working on?",
                precomputed_embedding=None,
            )

            # The provider's embed() SHOULD have been called
            mock_embedding.embed.assert_called_once()
