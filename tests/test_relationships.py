"""Unit tests for ContextEngine relate/unrelate/relations methods and CLI commands."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch, Mock, call
from io import StringIO

from context_engine.core import ContextEngine, ContextEngineError
from context_engine.config import ContextEngineConfig


# ─── ContextEngine.relate() ───────────────────────────────────────────────

class TestRelate:
    """Test ContextEngine.relate() method."""

    def test_relate_basic(self, test_config, mock_embedding):
        """Test creating a basic relationship between two memories."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            # Regular cursor: fetchone returns tuples for SELECT, but
            # for the INSERT RETURNING, fetchone returns tuple too
            mock_cur.fetchone.side_effect = [
                (1,),     # source memory found (tuple from regular cursor)
                (2,),     # target memory found
                (42,),    # INSERT RETURNING id (new row created)
            ]
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False,
            )

            result = engine.relate("doc-a", "doc-b")

            assert result is True
            mock_conn.commit.assert_called()

    def test_relate_already_exists(self, test_config, mock_embedding):
        """Test that relating already-related docs returns False."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchone.side_effect = [
                (1,),    # source memory found
                (2,),    # target memory found
                None,    # INSERT RETURNING returned nothing (already exists)
            ]
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False,
            )

            result = engine.relate("doc-a", "doc-b")

            assert result is False

    def test_relate_with_custom_type(self, test_config, mock_embedding):
        """Test relating with a custom rel_type."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchone.side_effect = [
                (1,),     # source memory found
                (2,),     # target memory found
                (42,),    # INSERT RETURNING id
            ]
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False,
            )

            result = engine.relate("doc-a", "doc-b", rel_type="depends_on")

            assert result is True
            # Check that "depends_on" appears in the INSERT query
            insert_call = mock_cur.execute.call_args_list[-1]
            assert "depends_on" in str(insert_call)

    def test_relate_invalid_type(self, test_config, mock_embedding):
        """Test that invalid rel_type raises ValueError."""
        engine = ContextEngine(
            config=test_config,
            embedding_provider=mock_embedding,
            auto_init=False,
        )

        with pytest.raises(ValueError, match="rel_type"):
            engine.relate("doc-a", "doc-b", rel_type="invalid_type")

    def test_relate_self_reference(self, test_config, mock_embedding):
        """Test that self-reference (source == target) raises ValueError."""
        engine = ContextEngine(
            config=test_config,
            embedding_provider=mock_embedding,
            auto_init=False,
        )

        with pytest.raises(ValueError, match="self"):
            engine.relate("doc-a", "doc-a")

    def test_relate_source_not_found(self, test_config, mock_embedding):
        """Test that relating with nonexistent source memory raises ContextEngineError."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchone.return_value = None  # source not found
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False,
            )

            with pytest.raises(ContextEngineError):
                engine.relate("nonexistent", "doc-b")

    def test_relate_target_not_found(self, test_config, mock_embedding):
        """Test that relating with nonexistent target memory raises ContextEngineError."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchone.side_effect = [
                (1,),     # source found
                None,     # target not found
            ]
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False,
            )

            with pytest.raises(ContextEngineError):
                engine.relate("doc-a", "nonexistent")

    def test_relate_db_error(self, test_config, mock_embedding):
        """Test that DB error raises ContextEngineError."""
        import psycopg2
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchone.side_effect = [
                (1,),
                (2,),
            ]
            # First two execute calls succeed (lookups), third fails (INSERT)
            call_count = [0]
            def execute_side_effect(query, params=None):
                call_count[0] += 1
                if call_count[0] <= 2:
                    return None  # lookups succeed
                raise psycopg2.Error("DB error")
            mock_cur.execute.side_effect = execute_side_effect
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False,
            )

            with pytest.raises(ContextEngineError):
                engine.relate("doc-a", "doc-b")
            mock_conn.rollback.assert_called()

    def test_valid_rel_types_constant(self):
        """Test that VALID_REL_TYPES contains expected types."""
        expected = {"related_to", "depends_on", "supersedes", "about",
                    "blocks", "references", "contains", "derived_from"}
        assert ContextEngine.VALID_REL_TYPES == expected


# ─── ContextEngine.unrelate() ──────────────────────────────────────────────

class TestUnrelate:
    """Test ContextEngine.unrelate() method."""

    def test_unrelate_basic(self, test_config, mock_embedding):
        """Test removing a relationship."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchone.side_effect = [(1,), (2,)]
            mock_cur.rowcount = 1
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False,
            )

            result = engine.unrelate("doc-a", "doc-b", rel_type="related_to")

            assert result is True
            mock_conn.commit.assert_called()

    def test_unrelate_not_found(self, test_config, mock_embedding):
        """Test that unrelating non-existent relationship returns False."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchone.side_effect = [(1,), (2,)]
            mock_cur.rowcount = 0
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False,
            )

            result = engine.unrelate("doc-a", "doc-b", rel_type="related_to")

            assert result is False

    def test_unrelate_without_type(self, test_config, mock_embedding):
        """Test unrelating all types between two documents."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchone.side_effect = [(1,), (2,)]
            mock_cur.rowcount = 3  # removed 3 relationship types
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False,
            )

            result = engine.unrelate("doc-a", "doc-b")

            assert result is True
            # Check the DELETE query does NOT include rel_type filter
            delete_call = mock_cur.execute.call_args_list[-1]
            sql = str(delete_call[0][0])
            assert "AND rel_type" not in sql

    def test_unrelate_source_not_found(self, test_config, mock_embedding):
        """Test unrelate with nonexistent source raises ContextEngineError."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchone.return_value = None
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False,
            )

            with pytest.raises(ContextEngineError):
                engine.unrelate("nonexistent", "doc-b")

    def test_unrelate_target_not_found(self, test_config, mock_embedding):
        """Test unrelate with nonexistent target raises ContextEngineError."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchone.side_effect = [
                (1,),    # source found
                None,    # target not found
            ]
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False,
            )

            with pytest.raises(ContextEngineError):
                engine.unrelate("doc-a", "nonexistent")


# ─── ContextEngine.relations() ─────────────────────────────────────────────

class TestRelations:
    """Test ContextEngine.relations() method."""

    def test_relations_outgoing(self, test_config, mock_embedding):
        """Test getting outgoing relationships."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchone.return_value = {'id': 1}
            mock_cur.fetchall.return_value = [
                {
                    'doc_id': 'doc-b',
                    'content': 'Target memory content',
                    'category': 'general',
                    'rel_type': 'related_to',
                },
            ]
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False,
            )

            results = engine.relations("doc-a", direction="outgoing")

            assert len(results) == 1
            assert results[0]['doc_id'] == 'doc-b'
            assert results[0]['direction'] == 'outgoing'
            assert results[0]['rel_type'] == 'related_to'

    def test_relations_incoming(self, test_config, mock_embedding):
        """Test getting incoming relationships."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchone.return_value = {'id': 1}
            mock_cur.fetchall.return_value = [
                {
                    'doc_id': 'doc-c',
                    'content': 'Source memory content',
                    'category': 'general',
                    'rel_type': 'depends_on',
                },
            ]
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False,
            )

            results = engine.relations("doc-a", direction="incoming")

            assert len(results) == 1
            assert results[0]['doc_id'] == 'doc-c'
            assert results[0]['direction'] == 'incoming'

    def test_relations_both(self, test_config, mock_embedding):
        """Test getting both incoming and outgoing relationships."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchone.return_value = {'id': 1}
            mock_cur.fetchall.side_effect = [
                [  # outgoing
                    {
                        'doc_id': 'doc-b',
                        'content': 'Target memory',
                        'category': 'general',
                        'rel_type': 'related_to',
                    },
                ],
                [  # incoming
                    {
                        'doc_id': 'doc-c',
                        'content': 'Source memory',
                        'category': 'general',
                        'rel_type': 'depends_on',
                    },
                ],
            ]
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False,
            )

            results = engine.relations("doc-a", direction="both")

            assert len(results) == 2
            directions = [r['direction'] for r in results]
            assert 'outgoing' in directions
            assert 'incoming' in directions

    def test_relations_with_type_filter(self, test_config, mock_embedding):
        """Test filtering by rel_type."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchone.return_value = {'id': 1}
            mock_cur.fetchall.return_value = [
                {
                    'doc_id': 'doc-b',
                    'content': 'Target memory',
                    'category': 'general',
                    'rel_type': 'depends_on',
                },
            ]
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False,
            )

            results = engine.relations("doc-a", direction="outgoing", rel_type="depends_on")

            assert len(results) == 1
            assert results[0]['rel_type'] == 'depends_on'

    def test_relations_empty(self, test_config, mock_embedding):
        """Test getting relations when none exist."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchone.return_value = {'id': 1}
            mock_cur.fetchall.return_value = []
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False,
            )

            results = engine.relations("doc-a", direction="outgoing")

            assert results == []

    def test_relations_content_truncated(self, test_config, mock_embedding):
        """Test that content is truncated to 200 chars."""
        long_content = "x" * 300
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchone.return_value = {'id': 1}
            mock_cur.fetchall.return_value = [
                {
                    'doc_id': 'doc-b',
                    'content': long_content,
                    'category': 'general',
                    'rel_type': 'related_to',
                },
            ]
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False,
            )

            results = engine.relations("doc-a", direction="outgoing")

            assert len(results[0]['content']) == 200

    def test_relations_memory_not_found(self, test_config, mock_embedding):
        """Test that looking up relations for nonexistent memory raises ContextEngineError."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchone.return_value = None
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False,
            )

            with pytest.raises(ContextEngineError):
                engine.relations("nonexistent")


# ─── CLI subcommands ───────────────────────────────────────────────────────

class TestCLIRelate:
    """Test CLI relate subcommand."""

    def test_relate_command_parsing(self):
        """Test that relate command parses arguments correctly."""
        with patch('sys.argv', ['ctx-engine', 'relate', 'doc-a', 'doc-b']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'relate', return_value=True) as mock_relate:
                    with patch.object(ContextEngine, 'close'):
                        try:
                            from context_engine.cli import main
                            main()
                        except SystemExit:
                            pass

                        mock_relate.assert_called_once()
                        call_kwargs = mock_relate.call_args.kwargs
                        assert call_kwargs['source_doc_id'] == 'doc-a'
                        assert call_kwargs['target_doc_id'] == 'doc-b'
                        assert call_kwargs['rel_type'] == 'related_to'

    def test_relate_with_type(self):
        """Test relate command with --type flag."""
        with patch('sys.argv', ['ctx-engine', 'relate', 'doc-a', 'doc-b', '--type', 'depends_on']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'relate', return_value=True) as mock_relate:
                    with patch.object(ContextEngine, 'close'):
                        try:
                            from context_engine.cli import main
                            main()
                        except SystemExit:
                            pass

                        call_kwargs = mock_relate.call_args.kwargs
                        assert call_kwargs['rel_type'] == 'depends_on'

    def test_relate_compact_output(self):
        """Test relate compact output format."""
        with patch('sys.argv', ['ctx-engine', '--format', 'compact', 'relate', 'doc-a', 'doc-b']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'relate', return_value=True):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                from context_engine.cli import main
                                main()
                            except SystemExit:
                                pass

                            assert "ok" in mock_stdout.getvalue()

    def test_relate_compact_exists(self):
        """Test relate compact output when relationship already exists."""
        with patch('sys.argv', ['ctx-engine', '--format', 'compact', 'relate', 'doc-a', 'doc-b']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'relate', return_value=False):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                from context_engine.cli import main
                                main()
                            except SystemExit:
                                pass

                            assert "exists" in mock_stdout.getvalue()

    def test_relate_text_output(self):
        """Test relate text output format."""
        with patch('sys.argv', ['ctx-engine', 'relate', 'doc-a', 'doc-b']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'relate', return_value=True):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                from context_engine.cli import main
                                main()
                            except SystemExit:
                                pass

                            output = mock_stdout.getvalue()
                            assert "doc-a" in output
                            assert "doc-b" in output
                            assert "related_to" in output

    def test_relate_text_output_exists(self):
        """Test relate text output when relationship already exists."""
        with patch('sys.argv', ['ctx-engine', 'relate', 'doc-a', 'doc-b']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'relate', return_value=False):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                from context_engine.cli import main
                                main()
                            except SystemExit:
                                pass

                            assert "already exists" in mock_stdout.getvalue().lower()


class TestCLIUnrelate:
    """Test CLI unrelate subcommand."""

    def test_unrelate_command_parsing(self):
        """Test that unrelate command parses arguments correctly."""
        with patch('sys.argv', ['ctx-engine', 'unrelate', 'doc-a', 'doc-b']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'unrelate', return_value=True) as mock_unrelate:
                    with patch.object(ContextEngine, 'close'):
                        try:
                            from context_engine.cli import main
                            main()
                        except SystemExit:
                            pass

                        mock_unrelate.assert_called_once()
                        call_kwargs = mock_unrelate.call_args.kwargs
                        assert call_kwargs['source_doc_id'] == 'doc-a'
                        assert call_kwargs['target_doc_id'] == 'doc-b'
                        assert call_kwargs['rel_type'] is None

    def test_unrelate_with_type(self):
        """Test unrelate command with --type flag."""
        with patch('sys.argv', ['ctx-engine', 'unrelate', 'doc-a', 'doc-b', '--type', 'related_to']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'unrelate', return_value=True) as mock_unrelate:
                    with patch.object(ContextEngine, 'close'):
                        try:
                            from context_engine.cli import main
                            main()
                        except SystemExit:
                            pass

                        call_kwargs = mock_unrelate.call_args.kwargs
                        assert call_kwargs['rel_type'] == 'related_to'

    def test_unrelate_compact_output(self):
        """Test unrelate compact output format."""
        with patch('sys.argv', ['ctx-engine', '--format', 'compact', 'unrelate', 'doc-a', 'doc-b']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'unrelate', return_value=True):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                from context_engine.cli import main
                                main()
                            except SystemExit:
                                pass

                            assert "ok" in mock_stdout.getvalue()

    def test_unrelate_compact_not_found(self):
        """Test unrelate compact output when relationship not found."""
        with patch('sys.argv', ['ctx-engine', '--format', 'compact', 'unrelate', 'doc-a', 'doc-b']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'unrelate', return_value=False):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                from context_engine.cli import main
                                main()
                            except SystemExit:
                                pass

                            assert "not_found" in mock_stdout.getvalue()

    def test_unrelate_text_output(self):
        """Test unrelate text output format."""
        with patch('sys.argv', ['ctx-engine', 'unrelate', 'doc-a', 'doc-b']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'unrelate', return_value=True):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                from context_engine.cli import main
                                main()
                            except SystemExit:
                                pass

                            assert "removed" in mock_stdout.getvalue().lower()


class TestCLIRelations:
    """Test CLI relations subcommand."""

    def test_relations_command_parsing(self):
        """Test that relations command parses arguments correctly."""
        with patch('sys.argv', ['ctx-engine', 'relations', 'doc-a']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'relations', return_value=[]) as mock_rels:
                    with patch.object(ContextEngine, 'close'):
                        try:
                            from context_engine.cli import main
                            main()
                        except SystemExit:
                            pass

                        mock_rels.assert_called_once()
                        call_kwargs = mock_rels.call_args.kwargs
                        assert call_kwargs['doc_id'] == 'doc-a'
                        assert call_kwargs['direction'] == 'both'
                        assert call_kwargs['rel_type'] is None

    def test_relations_with_direction(self):
        """Test relations command with --direction flag."""
        with patch('sys.argv', ['ctx-engine', 'relations', 'doc-a', '--direction', 'outgoing']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'relations', return_value=[]) as mock_rels:
                    with patch.object(ContextEngine, 'close'):
                        try:
                            from context_engine.cli import main
                            main()
                        except SystemExit:
                            pass

                        call_kwargs = mock_rels.call_args.kwargs
                        assert call_kwargs['direction'] == 'outgoing'

    def test_relations_compact_output(self):
        """Test relations compact output format."""
        rels = [
            {'doc_id': 'doc-b', 'content': 'Test content', 'category': 'general',
             'rel_type': 'related_to', 'direction': 'outgoing'},
        ]
        with patch('sys.argv', ['ctx-engine', '--format', 'compact', 'relations', 'doc-a']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'relations', return_value=rels):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                from context_engine.cli import main
                                main()
                            except SystemExit:
                                pass

                            output = mock_stdout.getvalue()
                            assert "related_to" in output
                            assert "-->" in output
                            assert "doc-b" in output

    def test_relations_text_output(self):
        """Test relations text output format."""
        rels = [
            {'doc_id': 'doc-b', 'content': 'Test content', 'category': 'general',
             'rel_type': 'related_to', 'direction': 'outgoing'},
        ]
        with patch('sys.argv', ['ctx-engine', 'relations', 'doc-a']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'relations', return_value=rels):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                from context_engine.cli import main
                                main()
                            except SystemExit:
                                pass

                            output = mock_stdout.getvalue()
                            assert "-->" in output
                            assert "related_to" in output

    def test_relations_incoming_arrow(self):
        """Test that incoming relations use <-- arrow."""
        rels = [
            {'doc_id': 'doc-c', 'content': 'Source content', 'category': 'general',
             'rel_type': 'depends_on', 'direction': 'incoming'},
        ]
        with patch('sys.argv', ['ctx-engine', '--format', 'compact', 'relations', 'doc-a']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'relations', return_value=rels):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                from context_engine.cli import main
                                main()
                            except SystemExit:
                                pass

                            output = mock_stdout.getvalue()
                            assert "<--" in output

    def test_relations_with_type_filter(self):
        """Test relations command with --type flag."""
        with patch('sys.argv', ['ctx-engine', 'relations', 'doc-a', '--type', 'depends_on']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'relations', return_value=[]) as mock_rels:
                    with patch.object(ContextEngine, 'close'):
                        try:
                            from context_engine.cli import main
                            main()
                        except SystemExit:
                            pass

                        call_kwargs = mock_rels.call_args.kwargs
                        assert call_kwargs['rel_type'] == 'depends_on'


class TestAgentInfoCompact:
    """Test that format_agent_info_compact includes new commands."""

    def test_agent_info_compact_includes_relate(self):
        """Test that compact agent info includes relate command."""
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            from context_engine.cli import format_agent_info_compact
            format_agent_info_compact()
            output = mock_stdout.getvalue()
            assert "relate" in output

    def test_agent_info_compact_includes_unrelate(self):
        """Test that compact agent info includes unrelate command."""
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            from context_engine.cli import format_agent_info_compact
            format_agent_info_compact()
            output = mock_stdout.getvalue()
            assert "unrelate" in output

    def test_agent_info_compact_includes_relations(self):
        """Test that compact agent info includes relations command."""
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            from context_engine.cli import format_agent_info_compact
            format_agent_info_compact()
            output = mock_stdout.getvalue()
            assert "relations" in output