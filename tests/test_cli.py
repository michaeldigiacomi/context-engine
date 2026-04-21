"""Tests for the ContextEngine CLI."""

import json
import pytest
import subprocess
import sys
from unittest.mock import patch, MagicMock
from io import StringIO

from context_engine.cli import main
from context_engine import ContextEngine


class TestCLIArgs:
    """Test CLI argument parsing."""

    def test_cli_no_args_prints_help(self):
        """Test that running without args prints help."""
        with patch('sys.argv', ['ctx-engine']):
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                try:
                    main()
                except SystemExit:
                    pass

                output = mock_stdout.getvalue()
                assert "usage:" in output or "Commands" in output

    def test_save_command_parsing(self):
        """Test that save command parses arguments correctly."""
        with patch('sys.argv', [
            'ctx-engine', 'save', 'Test content',
            '--category', 'test',
            '--importance', '5.0',
            '--ttl', '7',
            '--tags', 'tag1', 'tag2'
        ]):
            with patch.object(ContextEngine, '__init__', return_value=None) as mock_init:
                with patch.object(ContextEngine, 'save', return_value='doc-id-123') as mock_save:
                    with patch.object(ContextEngine, 'close'):
                        try:
                            main()
                        except SystemExit:
                            pass

                        mock_save.assert_called_once()
                        call_kwargs = mock_save.call_args.kwargs
                        assert call_kwargs['content'] == 'Test content'
                        assert call_kwargs['category'] == 'test'
                        assert call_kwargs['importance'] == 5.0
                        assert call_kwargs['ttl_days'] == 7

    def test_search_command_parsing(self):
        """Test that search command parses arguments correctly."""
        with patch('sys.argv', [
            'ctx-engine', 'search', 'test query',
            '--limit', '5',
            '--min-similarity', '0.7',
            '--category', 'infrastructure'
        ]):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'search', return_value=[]) as mock_search:
                    with patch.object(ContextEngine, 'close'):
                        try:
                            main()
                        except SystemExit:
                            pass

                        mock_search.assert_called_once()
                        call_kwargs = mock_search.call_args.kwargs
                        assert call_kwargs['query'] == 'test query'
                        assert call_kwargs['limit'] == 5
                        assert call_kwargs['min_similarity'] == 0.7
                        assert call_kwargs['category'] == 'infrastructure'

    def test_list_command_parsing(self):
        """Test that list command parses arguments correctly."""
        with patch('sys.argv', [
            'ctx-engine', 'list',
            '--category', 'preference',
            '--limit', '20'
        ]):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'list', return_value=[]) as mock_list:
                    with patch.object(ContextEngine, 'close'):
                        try:
                            main()
                        except SystemExit:
                            pass

                        mock_list.assert_called_once()
                        call_kwargs = mock_list.call_args.kwargs
                        assert call_kwargs['category'] == 'preference'
                        assert call_kwargs['limit'] == 20

    def test_delete_command_parsing(self):
        """Test that delete command parses arguments correctly."""
        with patch('sys.argv', ['ctx-engine', 'delete', 'doc-id-123']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'delete', return_value=True) as mock_delete:
                    with patch.object(ContextEngine, 'close'):
                        try:
                            main()
                        except SystemExit:
                            pass

                        mock_delete.assert_called_once_with('doc-id-123')

    def test_cleanup_command(self):
        """Test that cleanup command works."""
        with patch('sys.argv', ['ctx-engine', 'cleanup']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'cleanup_expired', return_value=5) as mock_cleanup:
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                main()
                            except SystemExit:
                                pass

                            mock_cleanup.assert_called_once()
                            assert "5" in mock_stdout.getvalue()

    def test_init_command(self):
        """Test that init command works."""
        with patch('sys.argv', ['ctx-engine', 'init']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, '_ensure_initialized') as mock_init:
                    with patch.object(ContextEngine, 'close'):
                        try:
                            main()
                        except SystemExit:
                            pass

                        mock_init.assert_called_once()


class TestCLIOutput:
    """Test CLI output formatting."""

    def test_save_output_shows_doc_id(self):
        """Test that save command shows doc_id."""
        with patch('sys.argv', ['ctx-engine', 'save', 'Test memory']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'save', return_value='abc123'):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                main()
                            except SystemExit:
                                pass

                            assert "Saved:" in mock_stdout.getvalue()
                            assert "abc123" in mock_stdout.getvalue()

    def test_search_output_format(self):
        """Test search output format."""
        results = [
            {
                'similarity': 0.85,
                'category': 'infrastructure',
                'content': 'Deployed to Kubernetes cluster'
            }
        ]

        with patch('sys.argv', ['ctx-engine', 'search', 'k8s']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'search', return_value=results):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                main()
                            except SystemExit:
                                pass

                            output = mock_stdout.getvalue()
                            assert "[0.85]" in output
                            assert "[infrastructure]" in output

    def test_search_no_results(self):
        """Test search with no results message."""
        with patch('sys.argv', ['ctx-engine', 'search', 'query']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'search', return_value=[]):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                main()
                            except SystemExit:
                                pass

                            assert "No results found" in mock_stdout.getvalue()

    def test_list_output_format(self):
        """Test list output format."""
        from datetime import datetime

        memories = [
            {
                'created_at': datetime(2024, 1, 15, 10, 30, 0),
                'category': 'preference',
                'content': 'User prefers dark mode'
            }
        ]

        with patch('sys.argv', ['ctx-engine', 'list']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'list', return_value=memories):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                main()
                            except SystemExit:
                                pass

                            output = mock_stdout.getvalue()
                            assert "2024-01-15" in output
                            assert "[preference]" in output

    def test_delete_success_output(self):
        """Test delete success message."""
        with patch('sys.argv', ['ctx-engine', 'delete', 'doc-id']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'delete', return_value=True):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                main()
                            except SystemExit:
                                pass

                            assert "Deleted" in mock_stdout.getvalue()

    def test_delete_not_found_output(self):
        """Test delete not found message."""
        with patch('sys.argv', ['ctx-engine', 'delete', 'nonexistent']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'delete', return_value=False):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                main()
                            except SystemExit:
                                pass

                            assert "Not found" in mock_stdout.getvalue()

    def test_get_context_output(self):
        """Test get-context output."""
        context = "[infrastructure] Deployed to k8s"

        with patch('sys.argv', ['ctx-engine', 'get-context', 'deployment']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'get_context', return_value=context):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                main()
                            except SystemExit:
                                pass

                            assert context in mock_stdout.getvalue()

    def test_get_context_empty_output(self):
        """Test get-context with no results."""
        with patch('sys.argv', ['ctx-engine', 'get-context', 'query']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'get_context', return_value=''):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                main()
                            except SystemExit:
                                pass

                            assert "(no context found)" in mock_stdout.getvalue()


class TestCLIErrorHandling:
    """Test CLI error handling."""

    def test_init_failure_prints_error(self):
        """Test that initialization failure shows error."""
        with patch('sys.argv', ['ctx-engine', 'save', 'test']):
            with patch.object(ContextEngine, '__init__', side_effect=Exception("DB error")):
                with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
                    with patch('sys.stdout', new_callable=StringIO):
                        with pytest.raises(SystemExit) as exc_info:
                            main()

                        assert exc_info.value.code == 1

    def test_command_error_prints_message(self):
        """Test that command errors are printed to stderr."""
        with patch('sys.argv', ['ctx-engine', 'save', 'test']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'save', side_effect=Exception("Save failed")):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
                            with pytest.raises(SystemExit) as exc_info:
                                main()

                            assert "Error:" in mock_stderr.getvalue()
                            assert exc_info.value.code == 1


class TestFormatArg:
    """Test --format flag and CTX_OUTPUT_FORMAT env var."""

    def test_format_flag_compact(self):
        """Test --format compact flag."""
        with patch('sys.argv', ['ctx-engine', '--format', 'compact', 'save', 'Test']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'save', return_value='abc123'):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                main()
                            except SystemExit:
                                pass
                            assert mock_stdout.getvalue().strip() == 'abc123'

    def test_format_flag_json(self):
        """Test --format json flag."""
        import json
        with patch('sys.argv', ['ctx-engine', '--format', 'json', 'save', 'Test']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'save', return_value='abc123'):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                main()
                            except SystemExit:
                                pass
                            data = json.loads(mock_stdout.getvalue())
                            assert data == {'id': 'abc123'}

    def test_format_default_is_text(self):
        """Test that default format is text."""
        with patch('sys.argv', ['ctx-engine', 'save', 'Test']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'save', return_value='abc123'):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                main()
                            except SystemExit:
                                pass
                            assert 'Saved:' in mock_stdout.getvalue()

    def test_ctx_output_format_env_var(self):
        """Test CTX_OUTPUT_FORMAT env var fallback."""
        with patch.dict('os.environ', {'CTX_OUTPUT_FORMAT': 'compact'}):
            with patch('sys.argv', ['ctx-engine', 'save', 'Test']):
                with patch.object(ContextEngine, '__init__', return_value=None):
                    with patch.object(ContextEngine, 'save', return_value='abc123'):
                        with patch.object(ContextEngine, 'close'):
                            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                                try:
                                    main()
                                except SystemExit:
                                    pass
                                assert mock_stdout.getvalue().strip() == 'abc123'

    def test_format_flag_overrides_env_var(self):
        """Test that --format flag overrides CTX_OUTPUT_FORMAT env var."""
        with patch.dict('os.environ', {'CTX_OUTPUT_FORMAT': 'json'}):
            with patch('sys.argv', ['ctx-engine', '--format', 'compact', 'save', 'Test']):
                with patch.object(ContextEngine, '__init__', return_value=None):
                    with patch.object(ContextEngine, 'save', return_value='abc123'):
                        with patch.object(ContextEngine, 'close'):
                            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                                try:
                                    main()
                                except SystemExit:
                                    pass
                                # compact outputs bare doc_id, not JSON
                                assert mock_stdout.getvalue().strip() == 'abc123'

    def test_invalid_env_var_ignored(self):
        """Test that invalid CTX_OUTPUT_FORMAT value is ignored (defaults to text)."""
        with patch.dict('os.environ', {'CTX_OUTPUT_FORMAT': 'xml'}):
            with patch('sys.argv', ['ctx-engine', 'save', 'Test']):
                with patch.object(ContextEngine, '__init__', return_value=None):
                    with patch.object(ContextEngine, 'save', return_value='abc123'):
                        with patch.object(ContextEngine, 'close'):
                            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                                try:
                                    main()
                                except SystemExit:
                                    pass
                                # Falls back to text mode
                                assert 'Saved:' in mock_stdout.getvalue()


class TestCompactFormat:
    """Test compact output format for all commands."""

    def test_search_compact(self):
        """Test search compact format: similarity|category|content"""
        results = [
            {'similarity': 0.69, 'category': 'infra', 'content': 'Deployed to k8s'},
            {'similarity': 0.55, 'category': 'work', 'content': 'Working on API'},
        ]
        with patch('sys.argv', ['ctx-engine', '--format', 'compact', 'search', 'k8s']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'search', return_value=results):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                main()
                            except SystemExit:
                                pass
                            lines = mock_stdout.getvalue().strip().split('\n')
                            assert lines[0] == '0.69|infra|Deployed to k8s'
                            assert lines[1] == '0.55|work|Working on API'

    def test_search_compact_pipe_escaping(self):
        r"""Test that literal | in content is escaped as \| in compact mode."""
        results = [
            {'similarity': 0.70, 'category': 'test', 'content': 'a|b|c'},
        ]
        with patch('sys.argv', ['ctx-engine', '--format', 'compact', 'search', 'test']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'search', return_value=results):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                main()
                            except SystemExit:
                                pass
                            assert 'a\\|b\\|c' in mock_stdout.getvalue()

    def test_search_compact_no_results(self):
        """Test search compact with no results prints nothing."""
        with patch('sys.argv', ['ctx-engine', '--format', 'compact', 'search', 'nothing']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'search', return_value=[]):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                main()
                            except SystemExit:
                                pass
                            assert mock_stdout.getvalue().strip() == ''

    def test_list_compact(self):
        """Test list compact format: YYYY-MM-DD|category|content"""
        from datetime import datetime
        memories = [
            {'created_at': datetime(2026, 4, 20, 14, 30), 'category': 'infra', 'content': 'Deployed app'},
        ]
        with patch('sys.argv', ['ctx-engine', '--format', 'compact', 'list']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'list', return_value=memories):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                main()
                            except SystemExit:
                                pass
                            assert mock_stdout.getvalue().strip() == '2026-04-20|infra|Deployed app'

    def test_list_compact_no_truncation(self):
        """Test list compact does NOT truncate content."""
        from datetime import datetime
        long_content = 'A' * 500
        memories = [
            {'created_at': datetime(2026, 4, 20), 'category': 'test', 'content': long_content},
        ]
        with patch('sys.argv', ['ctx-engine', '--format', 'compact', 'list']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'list', return_value=memories):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                main()
                            except SystemExit:
                                pass
                            assert long_content in mock_stdout.getvalue()

    def test_list_compact_no_results(self):
        """Test list compact with no results prints nothing."""
        with patch('sys.argv', ['ctx-engine', '--format', 'compact', 'list']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'list', return_value=[]):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                main()
                            except SystemExit:
                                pass
                            assert mock_stdout.getvalue().strip() == ''

    def test_get_context_compact_strips_prefix(self):
        """Test get-context compact strips [category] @source (date) prefix."""
        context = "[infrastructure] @manual (2026-04-20) Deployed to k8s\n[work] @chat (2026-04-19) Working on API"
        with patch('sys.argv', ['ctx-engine', '--format', 'compact', 'get-context', 'deploy']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'get_context', return_value=context):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                main()
                            except SystemExit:
                                pass
                            output = mock_stdout.getvalue()
                            assert '[infrastructure]' not in output
                            assert 'Deployed to k8s' in output
                            assert '---' in output

    def test_get_context_compact_empty(self):
        """Test get-context compact with empty context prints nothing."""
        with patch('sys.argv', ['ctx-engine', '--format', 'compact', 'get-context', 'nothing']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'get_context', return_value=''):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                main()
                            except SystemExit:
                                pass
                            assert mock_stdout.getvalue().strip() == ''

    def test_save_compact(self):
        """Test save compact just prints doc_id."""
        with patch('sys.argv', ['ctx-engine', '--format', 'compact', 'save', 'Test']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'save', return_value='abc123'):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                main()
                            except SystemExit:
                                pass
                            assert mock_stdout.getvalue().strip() == 'abc123'

    def test_delete_compact_ok(self):
        """Test delete compact prints ok on success."""
        with patch('sys.argv', ['ctx-engine', '--format', 'compact', 'delete', 'doc-id']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'delete', return_value=True):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                main()
                            except SystemExit:
                                pass
                            assert mock_stdout.getvalue().strip() == 'ok'

    def test_delete_compact_not_found(self):
        """Test delete compact prints not_found on failure."""
        with patch('sys.argv', ['ctx-engine', '--format', 'compact', 'delete', 'nonexistent']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'delete', return_value=False):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                main()
                            except SystemExit:
                                pass
                            assert mock_stdout.getvalue().strip() == 'not_found'

    def test_cleanup_compact(self):
        """Test cleanup compact prints just the number."""
        with patch('sys.argv', ['ctx-engine', '--format', 'compact', 'cleanup']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'cleanup_expired', return_value=5):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                main()
                            except SystemExit:
                                pass
                            assert mock_stdout.getvalue().strip() == '5'

    def test_init_compact(self):
        """Test init compact prints ok."""
        with patch('sys.argv', ['ctx-engine', '--format', 'compact', 'init']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, '_ensure_initialized'):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                main()
                            except SystemExit:
                                pass
                            assert mock_stdout.getvalue().strip() == 'ok'

    def test_agent_info_compact(self):
        """Test agent-info compact is 3-line summary."""
        with patch('sys.argv', ['ctx-engine', '--format', 'compact', 'agent-info']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'close'):
                    with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                        try:
                            main()
                        except SystemExit:
                            pass
                        output = mock_stdout.getvalue().strip()
                        lines = output.split('\n')
                        assert len(lines) == 3
                        assert 'ctx-engine: semantic memory via pgvector' in lines[0]
                        assert 'Commands:' in lines[1]
                        assert 'Config: ~/.config/context_engine/config.json' in lines[2]


class TestJsonFormat:
    """Test JSON output format for all commands."""

    def test_search_json(self):
        """Test search JSON format."""
        import json
        results = [
            {'similarity': 0.69, 'category': 'infra', 'content': 'Deployed', 'doc_id': 'abc123'},
        ]
        with patch('sys.argv', ['ctx-engine', '--format', 'json', 'search', 'test']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'search', return_value=results):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                main()
                            except SystemExit:
                                pass
                            data = json.loads(mock_stdout.getvalue())
                            assert isinstance(data, list)
                            assert len(data) == 1
                            assert data[0]['s'] == 0.69
                            assert data[0]['cat'] == 'infra'
                            assert data[0]['content'] == 'Deployed'
                            assert data[0]['id'] == 'abc123'

    def test_search_json_no_results(self):
        """Test search JSON with no results returns empty array."""
        import json
        with patch('sys.argv', ['ctx-engine', '--format', 'json', 'search', 'nothing']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'search', return_value=[]):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                main()
                            except SystemExit:
                                pass
                            data = json.loads(mock_stdout.getvalue())
                            assert data == []

    def test_list_json(self):
        """Test list JSON format."""
        import json
        from datetime import datetime
        memories = [
            {'created_at': datetime(2026, 4, 20), 'category': 'infra', 'content': 'Deployed', 'doc_id': 'abc123'},
        ]
        with patch('sys.argv', ['ctx-engine', '--format', 'json', 'list']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'list', return_value=memories):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                main()
                            except SystemExit:
                                pass
                            data = json.loads(mock_stdout.getvalue())
                            assert isinstance(data, list)
                            assert data[0]['cat'] == 'infra'
                            assert data[0]['date'] == '2026-04-20'
                            assert data[0]['id'] == 'abc123'

    def test_get_context_json(self):
        """Test get-context JSON format."""
        import json
        context = "[infra] @man (2026-04-20) Deployed to k8s"
        with patch('sys.argv', ['ctx-engine', '--format', 'json', 'get-context', 'deploy']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'get_context', return_value=context):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                main()
                            except SystemExit:
                                pass
                            data = json.loads(mock_stdout.getvalue())
                            assert 'context' in data
                            assert 'tokens_est' in data
                            assert 'memories' in data
                            assert data['tokens_est'] == len(context) // 4
                            assert data['memories'] == 1  # one [category] prefix

    def test_save_json(self):
        """Test save JSON format."""
        import json
        with patch('sys.argv', ['ctx-engine', '--format', 'json', 'save', 'Test']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'save', return_value='abc123'):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                main()
                            except SystemExit:
                                pass
                            data = json.loads(mock_stdout.getvalue())
                            assert data == {'id': 'abc123'}

    def test_delete_json_true(self):
        """Test delete JSON with deleted=true."""
        import json
        with patch('sys.argv', ['ctx-engine', '--format', 'json', 'delete', 'doc-id']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'delete', return_value=True):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                main()
                            except SystemExit:
                                pass
                            data = json.loads(mock_stdout.getvalue())
                            assert data == {'deleted': True}

    def test_delete_json_false(self):
        """Test delete JSON with deleted=false."""
        import json
        with patch('sys.argv', ['ctx-engine', '--format', 'json', 'delete', 'nonexistent']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'delete', return_value=False):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                main()
                            except SystemExit:
                                pass
                            data = json.loads(mock_stdout.getvalue())
                            assert data == {'deleted': False}

    def test_cleanup_json(self):
        """Test cleanup JSON format."""
        import json
        with patch('sys.argv', ['ctx-engine', '--format', 'json', 'cleanup']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'cleanup_expired', return_value=5):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                main()
                            except SystemExit:
                                pass
                            data = json.loads(mock_stdout.getvalue())
                            assert data == {'deleted': 5}

    def test_init_json(self):
        """Test init JSON format."""
        import json
        with patch('sys.argv', ['ctx-engine', '--format', 'json', 'init']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, '_ensure_initialized'):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                main()
                            except SystemExit:
                                pass
                            data = json.loads(mock_stdout.getvalue())
                            assert data == {'status': 'ok'}

    def test_agent_info_json(self):
        """Test agent-info JSON format."""
        import json
        with patch('sys.argv', ['ctx-engine', '--format', 'json', 'agent-info']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'close'):
                    with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                        try:
                            main()
                        except SystemExit:
                            pass
                        data = json.loads(mock_stdout.getvalue())
                        assert 'name' in data
                        assert 'commands' in data
                        assert data['name'] == 'ctx-engine'

    def test_json_output_is_valid_json_on_stdout(self):
        """Test that JSON output is always valid JSON even with errors to stderr."""
        import json
        with patch('sys.argv', ['ctx-engine', '--format', 'json', 'save', 'Test']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'save', return_value='abc123'):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                main()
                            except SystemExit:
                                pass
                            # Should be valid JSON
                            data = json.loads(mock_stdout.getvalue())
                            assert isinstance(data, dict)


class TestTextFormatTweaks:
    """Test text mode changes (truncation/timestamp updates)."""

    def test_search_text_truncation_200(self):
        """Test search text mode truncates at 200 chars, not 80."""
        long_content = 'A' * 200
        results = [
            {'similarity': 0.85, 'category': 'test', 'content': long_content + 'EXTRA'},
        ]
        with patch('sys.argv', ['ctx-engine', 'search', 'test']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'search', return_value=results):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                main()
                            except SystemExit:
                                pass
                            output = mock_stdout.getvalue()
                            # Should contain all 200 A's
                            assert 'A' * 200 in output
                            # Should have ... after truncation
                            assert '...' in output

    def test_list_text_timestamp_format(self):
        """Test list text mode uses YYYY-MM-DD HH:MM format."""
        from datetime import datetime
        memories = [
            {'created_at': datetime(2024, 1, 15, 10, 30, 45, 123456), 'category': 'test', 'content': 'Hello'},
        ]
        with patch('sys.argv', ['ctx-engine', 'list']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'list', return_value=memories):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                main()
                            except SystemExit:
                                pass
                            output = mock_stdout.getvalue()
                            assert '2024-01-15 10:30' in output
                            # Should NOT have seconds/microseconds
                            assert '10:30:45' not in output

    def test_list_text_truncation_200(self):
        """Test list text mode truncates at 200 chars, not 60."""
        from datetime import datetime
        long_content = 'B' * 200
        memories = [
            {'created_at': datetime(2024, 1, 15, 10, 30), 'category': 'test', 'content': long_content + 'EXTRA'},
        ]
        with patch('sys.argv', ['ctx-engine', 'list']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'list', return_value=memories):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                main()
                            except SystemExit:
                                pass
                            output = mock_stdout.getvalue()
                            assert 'B' * 200 in output


class TestCLIAgentCommands:
    """Test new CLI agent commands (Phase 4)."""

    def test_stats_command_text(self):
        with patch('sys.argv', ['ctx-engine', 'stats']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'stats', return_value={'total': 21, 'categories': {'infra': 12, 'project': 5, 'user': 4}, 'avg_importance': 7.3, 'total_size_kb': 8.4, 'last_saved': '2026-04-20'}):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            main()
                            output = mock_stdout.getvalue()
                            assert "21" in output

    def test_stats_command_compact(self):
        with patch('sys.argv', ['ctx-engine', '--format', 'compact', 'stats']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'stats', return_value={'total': 21, 'categories': {'infra': 12}, 'avg_importance': 7.3, 'total_size_kb': 8.4, 'last_saved': '2026-04-20'}):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            main()
                            output = mock_stdout.getvalue()
                            assert "21 memories" in output

    def test_stats_command_json(self):
        with patch('sys.argv', ['ctx-engine', '--format', 'json', 'stats']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'stats', return_value={'total': 5, 'categories': {'test': 5}, 'avg_importance': 5.0, 'total_size_kb': 2.0, 'last_saved': None}):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            main()
                            output = mock_stdout.getvalue()
                            parsed = json.loads(output)
                            assert parsed['total'] == 5

    def test_peek_command_text(self):
        with patch('sys.argv', ['ctx-engine', 'peek', 'doc123']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'peek', return_value={'doc_id': 'doc123', 'content': 'Test content', 'category': 'test', 'source': 'unit', 'importance': 5.0, 'created_at': '2026-04-20', 'tags': None}):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            main()
                            output = mock_stdout.getvalue()
                            assert "Test content" in output

    def test_peek_not_found(self):
        with patch('sys.argv', ['ctx-engine', 'peek', 'nonexistent']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'peek', return_value=None):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            main()
                            output = mock_stdout.getvalue()
                            assert "Not found" in output

    def test_peek_compact(self):
        with patch('sys.argv', ['ctx-engine', '--format', 'compact', 'peek', 'doc123']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'peek', return_value={'doc_id': 'doc123', 'content': 'Compact content', 'category': 'test'}):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            main()
                            output = mock_stdout.getvalue()
                            assert output.strip() == "Compact content"

    def test_count_command(self):
        with patch('sys.argv', ['ctx-engine', 'count']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'count', return_value=42):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            main()
                            assert mock_stdout.getvalue().strip() == "42"

    def test_count_compact(self):
        with patch('sys.argv', ['ctx-engine', '--format', 'compact', 'count']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'count', return_value=15):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            main()
                            assert mock_stdout.getvalue().strip() == "15"

    def test_count_json(self):
        with patch('sys.argv', ['ctx-engine', '--format', 'json', 'count']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'count', return_value=10):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            main()
                            parsed = json.loads(mock_stdout.getvalue())
                            assert parsed['count'] == 10

    def test_search_one_command(self):
        with patch('sys.argv', ['ctx-engine', 'search-one', 'k3s server']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'search_one', return_value='K3s deployment info...'):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            main()
                            assert "K3s deployment info" in mock_stdout.getvalue()

    def test_search_one_no_match(self):
        with patch('sys.argv', ['ctx-engine', 'search-one', 'nonexistent']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'search_one', return_value=None):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            main()
                            assert "No match" in mock_stdout.getvalue()

    def test_search_one_compact(self):
        with patch('sys.argv', ['ctx-engine', '--format', 'compact', 'search-one', 'test']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'search_one', return_value='Result content'):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            main()
                            assert mock_stdout.getvalue().strip() == "Result content"

    def test_search_default_limit_is_5(self):
        """Verify search --limit defaults to 5 (changed from 10)."""
        with patch('sys.argv', ['ctx-engine', 'search', 'test']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'search', return_value=[]) as mock_search:
                    with patch.object(ContextEngine, 'close'):
                        main()
                        call_kwargs = mock_search.call_args.kwargs
                        assert call_kwargs['limit'] == 5

    def test_get_context_default_tokens_is_2000(self):
        """Verify get-context --max-tokens defaults to 2000 (changed from 4000)."""
        with patch('sys.argv', ['ctx-engine', 'get-context', 'test']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'get_context', return_value='') as mock_gc:
                    with patch.object(ContextEngine, 'close'):
                        main()
                        call_kwargs = mock_gc.call_args.kwargs
                        assert call_kwargs['max_tokens'] == 2000

    def test_agent_info_default_is_slim(self):
        """Verify agent-info default output is slim (5-line summary, not 2757-char wall)."""
        with patch('sys.argv', ['ctx-engine', 'agent-info']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'close'):
                    with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                        main()
                        output = mock_stdout.getvalue()
                        # Slim output should be short (< 300 chars), not the massive ASCII art
                        assert len(output) < 300
                        assert "ctx-engine" in output

    def test_agent_info_verbose_shows_full(self):
        """Verify agent-info --verbose shows full output."""
        with patch('sys.argv', ['ctx-engine', 'agent-info', '--verbose']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'close'):
                    with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                        main()
                        output = mock_stdout.getvalue()
                        # Full output should be much longer
                        assert len(output) > 500
