#!/usr/bin/env python3
"""
CLI for Context Engine.

Usage:
    ctx-engine save "Memory content" --category work
    ctx-engine search "What was I working on?"
    ctx-engine list --category work
    ctx-engine cleanup
"""

import argparse
import json
import os
import sys

from context_engine import ContextEngine
from context_engine.core import ContextEngine as _CE

# Pull VALID_REL_TYPES from the class for CLI choices
VALID_REL_TYPES = sorted(_CE.VALID_REL_TYPES)


def get_output_format(args):
    """Determine output format from args or CTX_OUTPUT_FORMAT env var.

    Priority: explicit --format flag > CTX_OUTPUT_FORMAT env var > 'text' default.
    """
    # If the user explicitly passed --format, use it
    fmt = getattr(args, 'format', None)
    if fmt is not None:
        return fmt
    # Fallback to env var
    env_fmt = os.environ.get('CTX_OUTPUT_FORMAT', '').lower()
    if env_fmt in ('text', 'json', 'compact'):
        return env_fmt
    # Default
    return 'text'


def _escape_pipe(s):
    """Escape literal pipe characters in a string for compact mode."""
    return s.replace('|', '\\|')


def _format_date(dt):
    """Format a datetime to YYYY-MM-DD string."""
    if dt is None:
        return 'unknown'
    if isinstance(dt, str):
        return dt[:10] if len(dt) >= 10 else dt
    return dt.strftime('%Y-%m-%d')


def _format_datetime_short(dt):
    """Format a datetime to YYYY-MM-DD HH:MM string (for text mode list)."""
    if dt is None:
        return 'unknown'
    if isinstance(dt, str):
        # If already a string, try to parse or just return it
        return dt[:16] if len(dt) >= 16 else dt
    return dt.strftime('%Y-%m-%d %H:%M')


# ── Text formatters (default, backward-compatible with tweaks) ──────────────

def format_search_text(results):
    lines = []
    for r in results:
        content = r['content'][:200]
        if len(r['content']) > 200:
            content += '...'
        lines.append(f"[{r['similarity']:.2f}] [{r['category']}] {content}")
    if not results:
        print("No results found.")
    else:
        print('\n'.join(lines))


def format_list_text(memories):
    for m in memories:
        created = _format_datetime_short(m.get('created_at'))
        content = m['content'][:200]
        if len(m['content']) > 200:
            content += '...'
        print(f"[{created}] [{m['category']}] {content}")


def format_get_context_text(context):
    print(context or "(no context found)")


def format_save_text(doc_id):
    print(f"Saved: {doc_id}")


def format_delete_text(deleted):
    print("Deleted." if deleted else "Not found.")


def format_cleanup_text(count):
    print(f"Deleted {count} expired memories.")


def format_init_text():
    print("Database initialized.")


def format_agent_info_text(show_python=False):
    show_agent_info(show_python=show_python)


def format_stats_compact(result):
    cats = ', '.join(f"{v} {k}" for k, v in result.get('categories', {}).items())
    print(f"{result.get('total', 0)} memories | {cats} | ~{result.get('total_size_kb', 0):.1f}KB", file=sys.stdout)


def format_stats_json(result):
    _json_print(result)


def format_stats_text(result):
    print(f"Total memories: {result.get('total', 0)}")
    cats = result.get('categories', {})
    if cats:
        print("Categories:")
        for k, v in cats.items():
            print(f"  {k}: {v}")
    print(f"Avg importance: {result.get('avg_importance', 0):.1f}")
    print(f"Total size: ~{result.get('total_size_kb', 0):.1f}KB")
    if result.get('last_saved'):
        print(f"Last saved: {result['last_saved']}")


def format_peek_compact(result):
    if result is None:
        return
    print(result.get('content', ''), file=sys.stdout)


def format_peek_json(result):
    _json_print(result if result else None)


def format_peek_text(result, doc_id):
    if result is None:
        print("Not found.")
        return
    print(f"--- {result.get('doc_id', doc_id)} ---")
    print(f"Category: {result.get('category', 'unknown')}")
    print(f"Source: {result.get('source', 'unknown')}")
    print(f"Importance: {result.get('importance', 0)}")
    print(f"Created: {result.get('created_at', 'unknown')}")
    if result.get('tags'):
        print(f"Tags: {', '.join(result['tags'])}")
    print(f"\n{result.get('content', '')}")


def format_count_compact(n):
    print(n, file=sys.stdout)


def format_count_json(n):
    _json_print({'count': n})


def format_count_text(n):
    print(n)


def format_search_one_compact(content):
    if content is None:
        return
    print(content, file=sys.stdout)


def format_search_one_json(content):
    _json_print({'content': content})


def format_search_one_text(content):
    if content is None:
        print("No match found.")
    else:
        print(content)


# ── Relate formatters ──────────────────────────────────────────────────────

def format_relate_compact(created):
    print("ok" if created else "exists", file=sys.stdout)


def format_relate_json(created, source, target, rel_type):
    _json_print({'created': created, 'source': source, 'target': target, 'rel_type': rel_type})


def format_relate_text(created, source, target, rel_type):
    if created:
        print(f"Related {source} --{rel_type}--> {target}")
    else:
        print("Relationship already exists.")


# ── Unrelate formatters ────────────────────────────────────────────────────

def format_unrelate_compact(removed):
    print("ok" if removed else "not_found", file=sys.stdout)


def format_unrelate_json(removed, source, target, rel_type):
    _json_print({'removed': removed, 'source': source, 'target': target, 'rel_type': rel_type})


def format_unrelate_text(removed):
    print("Relationship removed." if removed else "Relationship not found.")


# ── Relations formatters ────────────────────────────────────────────────────

def format_relations_compact(relations):
    for r in relations:
        arrow = "-->" if r['direction'] == 'outgoing' else "<--"
        content = _escape_pipe(r['content'][:80])
        print(f"{r['rel_type']}|{arrow}|{r['doc_id']}|{content}", file=sys.stdout)


def format_relations_json(relations):
    _json_print(relations)


def format_relations_text(relations):
    for r in relations:
        arrow = "-->" if r['direction'] == 'outgoing' else "<--"
        doc_id_short = r['doc_id'][:16] + "..." if len(r['doc_id']) > 16 else r['doc_id']
        content_short = r['content'][:80]
        print(f"  {arrow} [{r['rel_type']}] {doc_id_short} {content_short}")


# ── Compact formatters ──────────────────────────────────────────────────────

def format_search_compact(results):
    if not results:
        return
    for r in results:
        content = _escape_pipe(r['content'])
        print(f"{r['similarity']:.2f}|{r['category']}|{content}", file=sys.stdout)


def format_list_compact(memories):
    if not memories:
        return
    for m in memories:
        date = _format_date(m.get('created_at'))
        content = _escape_pipe(m['content'])
        print(f"{date}|{m['category']}|{content}", file=sys.stdout)


def format_get_context_compact(context):
    if not context:
        return
    # Strip [category] @source (date) prefixes, return content paragraphs
    # separated by \n---\n
    import re
    # Match prefix patterns like [category] @source (2026-04-20) or [category] (date)
    # Applied repeatedly to handle any remaining prefixes
    lines = context.split('\n')
    content_parts = []
    for line in lines:
        stripped = line
        # Keep stripping [category] prefixes from the beginning of the line
        while True:
            new_stripped = re.sub(
                r'^\s*\[[^\]]*\]\s*(?:@\S+\s*)?(?:\(\d{4}-\d{2}-\d{2}[^)]*\)\s*)?',
                '', stripped
            )
            if new_stripped == stripped:
                break
            stripped = new_stripped
        stripped = stripped.strip()
        if stripped:
            content_parts.append(stripped)
    if content_parts:
        print('\n---\n'.join(content_parts), file=sys.stdout)


def format_save_compact(doc_id):
    print(doc_id, file=sys.stdout)


def format_delete_compact(deleted):
    print("ok" if deleted else "not_found", file=sys.stdout)


def format_cleanup_compact(count):
    print(str(count), file=sys.stdout)


def format_init_compact():
    print("ok", file=sys.stdout)


def format_agent_info_compact():
    lines = [
        "ctx-engine: semantic memory via pgvector",
        "Commands: save, search, get-context, list, delete, cleanup, init, stats, peek, count, search-one, relate, unrelate, relations",
        "Config: ~/.config/context_engine/config.json",
    ]
    print('\n'.join(lines), file=sys.stdout)


# ── JSON formatters ─────────────────────────────────────────────────────────

def _json_print(obj):
    """Print valid JSON to stdout."""
    print(json.dumps(obj, ensure_ascii=False), file=sys.stdout)


def format_search_json(results):
    if not results:
        _json_print([])
        return
    out = []
    for r in results:
        out.append({
            's': round(r['similarity'], 2),
            'cat': r['category'],
            'content': r['content'],
            'id': r.get('doc_id', r.get('id', '')),
        })
    _json_print(out)


def format_list_json(memories):
    if not memories:
        _json_print([])
        return
    out = []
    for m in memories:
        out.append({
            'cat': m['category'],
            'content': m['content'],
            'id': m.get('doc_id', m.get('id', '')),
            'date': _format_date(m.get('created_at')),
        })
    _json_print(out)


def format_get_context_json(context, result_count=0):
    # tokens_est = len(context) // 4; memories = count of results
    tokens_est = len(context) // 4 if context else 0
    _json_print({
        'context': context or '',
        'tokens_est': tokens_est,
        'memories': result_count,
    })


def format_save_json(doc_id):
    _json_print({'id': doc_id})


def format_delete_json(deleted):
    _json_print({'deleted': bool(deleted)})


def format_cleanup_json(count):
    _json_print({'deleted': count})


def format_init_json():
    _json_print({'status': 'ok'})


def format_agent_info_json():
    _json_print({
        'name': 'ctx-engine',
        'description': 'semantic memory via pgvector',
        'commands': ['save', 'search', 'get-context', 'list', 'delete', 'cleanup', 'init', 'stats', 'peek', 'count', 'search-one', 'relate', 'unrelate', 'relations'],
        'config': '~/.config/context_engine/config.json',
    })


def show_agent_info(show_python=False):
    """Display information for AI agents about the context engine."""

    if show_python:
        python_example = '''
# QUICK START FOR AGENTS
# ======================

from context_engine import ContextEngine

# Initialize (auto-loads config from ~/.config/context_engine/config.json)
memory = ContextEngine()

# Save a memory
memory.save(
    content="User prefers Python",
    category="preference",
    importance=9.0
)

# Get relevant context for a query
context = memory.get_context(
    "What language should I use?",
    max_tokens=2000
)

# Save a conversation turn
memory.save_conversation(
    session_key="session-123",
    user_message="Hello",
    assistant_response="Hi there!"
)

# Search memories
results = memory.search("user preferences", limit=5)

# Cleanup
memory.close()
'''
        print(python_example)
        return

    info = '''
╔════════════════════════════════════════════════════════════════╗
║          PGVECTOR CONTEXT ENGINE - AGENT INFO                  ║
╚════════════════════════════════════════════════════════════════╝

WHAT IS THIS?
=============
A semantic memory system for AI agents. Store and retrieve relevant
context using vector embeddings (not just keyword search).

WHY USE IT?
===========
• Persistent memory across sessions
• Semantic search (find by meaning, not exact words)
• Automatic token budgeting for LLM context
• Namespace isolation for multi-agent setups
• TTL support for temporary memories

QUICK START
===========

1. Verify it's set up:
   ctx-engine list

2. Use in your agent code:

   from context_engine import ContextEngine
   memory = ContextEngine()

   # Save something
   memory.save("User likes Python", category="preference")

   # Retrieve relevant context
   context = memory.get_context("What language?", max_tokens=2000)

   # Use with your LLM
   response = call_llm(context + user_message)

COMMON COMMANDS
===============

ctx-engine save "Content" --category preference --importance 9
  → Save a memory with high importance

ctx-engine search "user preferences" --limit 5
  → Find relevant memories

ctx-engine get-context "current task" --max-tokens 3000
  → Get formatted context for LLM

ctx-engine list --category preference
  → List memories by category

AGENT BASE CLASS
================

For a complete agent framework:

from context_engine.agent import ContextAgent

class MyAgent(ContextAgent):
    def process(self, message):
        context = self.get_relevant_context(message)
        # Your LLM logic here
        return response

CONFIGURATION
=============

Config file: ~/.config/context_engine/config.json

Environment variables:
  CTX_DB_HOST      - PostgreSQL host
  CTX_DB_PORT      - PostgreSQL port
  CTX_DB_NAME      - Database name
  CTX_DB_USER      - Database user
  CTX_DB_PASS      - Database password
  CTX_NAMESPACE    - Agent namespace (isolation)
  CTX_OLLAMA_URL   - Ollama URL for embeddings

DOCUMENTATION
=============

• AGENT_SETUP.md      - Quick setup guide for agents
• AGENT_INTEGRATION.md - Detailed integration patterns
• README.md           - Full documentation

EXAMPLES
========

See examples/agent_example.py for a complete working agent.

MORE INFO
=========

Show Python code example:
  ctx-engine agent-info --python

Repository: https://github.com/michaeldigiacomi/ai-context-engine
'''
    print(info)


def main():
    parser = argparse.ArgumentParser(description="Context Engine CLI")
    parser.add_argument(
        '--format',
        choices=['text', 'json', 'compact'],
        default=None,
        help='Output format (default: text). Overrides CTX_OUTPUT_FORMAT env var.',
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # save command
    save_parser = subparsers.add_parser("save", help="Save a memory")
    save_parser.add_argument("content", help="Memory content")
    save_parser.add_argument("--category", default="general")
    save_parser.add_argument("--importance", type=float, default=1.0)
    save_parser.add_argument("--ttl", type=int, help="Days until expiration")
    save_parser.add_argument("--session", help="Session key")
    save_parser.add_argument("--tags", nargs="+", help="Tags")
    save_parser.add_argument("--source", help="Source identifier")
    save_parser.add_argument("--doc-id", help="Stable document ID")

    # search command
    search_parser = subparsers.add_parser("search", help="Search memories")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--limit", type=int, default=5)
    search_parser.add_argument("--min-similarity", type=float, default=0.5)
    search_parser.add_argument("--category", help="Filter by category")

    # get-context command
    ctx_parser = subparsers.add_parser("get-context", help="Get context for a query")
    ctx_parser.add_argument("query", help="Query or task description")
    ctx_parser.add_argument("--max-memories", type=int, default=10)
    ctx_parser.add_argument("--max-tokens", type=int, default=2000)
    ctx_parser.add_argument("--category", help="Filter by category")

    # list command
    list_parser = subparsers.add_parser("list", help="List memories")
    list_parser.add_argument("--category", help="Filter by category")
    list_parser.add_argument("--limit", type=int, default=50)

    # delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a memory")
    delete_parser.add_argument("doc_id", help="Document ID to delete")

    # cleanup command
    subparsers.add_parser("cleanup", help="Delete expired memories")

    # init command
    subparsers.add_parser("init", help="Initialize database schema")

    # agent-info command
    agent_info_parser = subparsers.add_parser("agent-info", help="Show information for AI agents")
    agent_info_parser.add_argument("--python", action="store_true", help="Show Python code example")
    agent_info_parser.add_argument("--verbose", action="store_true", help="Show full output")

    # stats command
    stats_parser = subparsers.add_parser("stats", help="Show memory statistics")

    # peek command
    peek_parser = subparsers.add_parser("peek", help="Show full content of a memory")
    peek_parser.add_argument("doc_id", help="Document ID to peek")

    # count command
    count_parser = subparsers.add_parser("count", help="Print memory count")
    count_parser.add_argument("--category", help="Filter by category")

    # search-one command
    search_one_parser = subparsers.add_parser("search-one", help="Return single best match content")
    search_one_parser.add_argument("query", help="Search query")
    search_one_parser.add_argument("--min-similarity", type=float, default=0.3)
    search_one_parser.add_argument("--category", help="Filter by category")

    # relate command
    relate_parser = subparsers.add_parser("relate", help="Create a relationship between two memories")
    relate_parser.add_argument("source", help="Source document ID")
    relate_parser.add_argument("target", help="Target document ID")
    relate_parser.add_argument("--type", "-t", dest="rel_type", default="related_to",
                               choices=VALID_REL_TYPES, help="Relationship type")

    # unrelate command
    unrelate_parser = subparsers.add_parser("unrelate", help="Remove a relationship between two memories")
    unrelate_parser.add_argument("source", help="Source document ID")
    unrelate_parser.add_argument("target", help="Target document ID")
    unrelate_parser.add_argument("--type", "-t", dest="rel_type", default=None,
                                  choices=VALID_REL_TYPES, help="Relationship type (omit to remove all types)")

    # relations command
    relations_parser = subparsers.add_parser("relations", help="Show relationships for a memory")
    relations_parser.add_argument("doc_id", help="Document ID")
    relations_parser.add_argument("--direction", "-d", choices=["outgoing", "incoming", "both"],
                                  default="both", help="Relationship direction")
    relations_parser.add_argument("--type", "-t", dest="rel_type", default=None,
                                  help="Filter by relationship type")

    # working command
    working_parser = subparsers.add_parser("working", help="Working memory commands")
    working_subparsers = working_parser.add_subparsers(dest="working_command")

    # working set
    set_parser = working_subparsers.add_parser("set", help="Set session context")
    set_parser.add_argument("key", help="Context key")
    set_parser.add_argument("value", help="Context value")
    set_parser.add_argument("--priority", type=int, default=5)
    set_parser.add_argument("--ttl", type=int, default=60, help="TTL in minutes")

    # working get
    get_parser = working_subparsers.add_parser("get", help="Get session context")

    # working tasks
    tasks_parser = working_subparsers.add_parser("tasks", help="List tasks")
    tasks_parser.add_argument("--status", help="Filter by status")

    # working add-task
    add_task_parser = working_subparsers.add_parser("add-task", help="Add a task")
    add_task_parser.add_argument("description", help="Task description")
    add_task_parser.add_argument("--priority", type=int, default=5)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Determine output format
    fmt = get_output_format(args)

    # Initialize engine
    try:
        ctx = ContextEngine()
    except Exception as e:
        print(f"Failed to initialize ContextEngine: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        if args.command == "save":
            doc_id = ctx.save(
                content=args.content,
                category=args.category,
                importance=args.importance,
                ttl_days=args.ttl,
                session_key=args.session,
                tags=args.tags,
                source=args.source,
                doc_id=args.doc_id,
            )
            if fmt == 'compact':
                format_save_compact(doc_id)
            elif fmt == 'json':
                format_save_json(doc_id)
            else:
                format_save_text(doc_id)

        elif args.command == "search":
            results = ctx.search(
                query=args.query,
                limit=args.limit,
                min_similarity=args.min_similarity,
                category=args.category,
            )
            if fmt == 'compact':
                format_search_compact(results)
            elif fmt == 'json':
                format_search_json(results)
            else:
                format_search_text(results)

        elif args.command == "get-context":
            context = ctx.get_context(
                query=args.query,
                max_memories=args.max_memories,
                max_tokens=args.max_tokens,
                category=args.category,
            )
            # For JSON, we need the result count. Since get_context returns a
            # string, we estimate result count from the context format
            # (paragraphs separated by blank lines or prefixes).
            # We also try to get the actual result list for counting.
            result_count = 0
            if context:
                # Count [category] prefixes as a heuristic for result count
                import re
                result_count = len(re.findall(r'^\[[^\]]+\]', context, re.MULTILINE))
                if result_count == 0:
                    # If no prefixes found, treat the whole thing as 1 result
                    result_count = 1
            if fmt == 'compact':
                format_get_context_compact(context)
            elif fmt == 'json':
                format_get_context_json(context, result_count)
            else:
                format_get_context_text(context)

        elif args.command == "list":
            memories = ctx.list(category=args.category, limit=args.limit)
            if fmt == 'compact':
                format_list_compact(memories)
            elif fmt == 'json':
                format_list_json(memories)
            else:
                format_list_text(memories)

        elif args.command == "delete":
            deleted = ctx.delete(args.doc_id)
            if fmt == 'compact':
                format_delete_compact(deleted)
            elif fmt == 'json':
                format_delete_json(deleted)
            else:
                format_delete_text(deleted)

        elif args.command == "cleanup":
            count = ctx.cleanup_expired()
            if fmt == 'compact':
                format_cleanup_compact(count)
            elif fmt == 'json':
                format_cleanup_json(count)
            else:
                format_cleanup_text(count)

        elif args.command == "init":
            ctx._ensure_initialized()
            if fmt == 'compact':
                format_init_compact()
            elif fmt == 'json':
                format_init_json()
            else:
                format_init_text()

        elif args.command == "agent-info":
            if fmt == 'compact':
                format_agent_info_compact()
            elif fmt == 'json':
                format_agent_info_json()
            else:
                if args.python:
                    # Show Python example (existing behavior)
                    format_agent_info_text(show_python=True)
                elif getattr(args, 'verbose', False):
                    # Full ASCII art output (existing behavior)
                    format_agent_info_text(show_python=False)
                else:
                    # Slim 5-line summary (new default)
                    lines = [
                        "ctx-engine: semantic memory via pgvector",
                        "CLI: ctx-engine save|search|get-context|list|delete|stats|peek|count|search-one",
                        "Python: from context_engine import ContextEngine; ctx = ContextEngine()",
                        "Config: ~/.config/context_engine/config.json",
                        "Use --verbose for full info, --python for code example",
                    ]
                    print('\n'.join(lines))

        elif args.command == "stats":
            result = ctx.stats()
            if fmt == 'compact':
                format_stats_compact(result)
            elif fmt == 'json':
                format_stats_json(result)
            else:
                format_stats_text(result)

        elif args.command == "peek":
            result = ctx.peek(args.doc_id)
            if fmt == 'compact':
                format_peek_compact(result)
            elif fmt == 'json':
                format_peek_json(result)
            else:
                format_peek_text(result, args.doc_id)

        elif args.command == "count":
            n = ctx.count(category=args.category if hasattr(args, 'category') else None)
            if fmt == 'compact':
                format_count_compact(n)
            elif fmt == 'json':
                format_count_json(n)
            else:
                format_count_text(n)

        elif args.command == "search-one":
            content = ctx.search_one(query=args.query, min_similarity=args.min_similarity, category=args.category if hasattr(args, 'category') else None)
            if fmt == 'compact':
                format_search_one_compact(content)
            elif fmt == 'json':
                format_search_one_json(content)
            else:
                format_search_one_text(content)

        elif args.command == "relate":
            created = ctx.relate(
                source_doc_id=args.source,
                target_doc_id=args.target,
                rel_type=args.rel_type,
            )
            if fmt == 'compact':
                format_relate_compact(created)
            elif fmt == 'json':
                format_relate_json(created, args.source, args.target, args.rel_type)
            else:
                format_relate_text(created, args.source, args.target, args.rel_type)

        elif args.command == "unrelate":
            removed = ctx.unrelate(
                source_doc_id=args.source,
                target_doc_id=args.target,
                rel_type=args.rel_type,
            )
            if fmt == 'compact':
                format_unrelate_compact(removed)
            elif fmt == 'json':
                format_unrelate_json(removed, args.source, args.target, args.rel_type)
            else:
                format_unrelate_text(removed)

        elif args.command == "relations":
            rels = ctx.relations(
                doc_id=args.doc_id,
                direction=args.direction,
                rel_type=args.rel_type,
            )
            if fmt == 'compact':
                format_relations_compact(rels)
            elif fmt == 'json':
                format_relations_json(rels)
            else:
                format_relations_text(rels)

        elif args.command == "working":
            from context_engine.memory_manager import MemoryManager
            manager = MemoryManager()

            if args.working_command == "set":
                manager.working.set_session_context(
                    args.key, args.value,
                    priority=args.priority, ttl_minutes=args.ttl
                )
                if fmt == 'compact':
                    print(f"{args.key}={args.value}", file=sys.stdout)
                elif fmt == 'json':
                    _json_print({'key': args.key, 'value': args.value})
                else:
                    print(f"Set {args.key} = {args.value}")

            elif args.working_command == "get":
                wctx = manager.working.get_session_context()
                if fmt == 'compact':
                    if wctx:
                        for k, v in wctx.items():
                            print(f"{k}={v}", file=sys.stdout)
                elif fmt == 'json':
                    _json_print(wctx if wctx else {})
                else:
                    if wctx:
                        for k, v in wctx.items():
                            print(f"{k}: {v}")
                    else:
                        print("No session context")

            elif args.working_command == "tasks":
                tasks = manager.working.get_tasks(status=args.status)
                if fmt == 'compact':
                    for t in tasks:
                        print(f"{t['status']}|{t['task_id']}|{t['description']}", file=sys.stdout)
                elif fmt == 'json':
                    _json_print([{'status': t['status'], 'id': t['task_id'], 'desc': t['description']} for t in tasks])
                else:
                    for t in tasks:
                        print(f"[{t['status']}] {t['task_id']}: {t['description']}")

            elif args.working_command == "add-task":
                task_id = manager.working.save_task(
                    description=args.description,
                    priority=args.priority
                )
                if fmt == 'compact':
                    print(task_id, file=sys.stdout)
                elif fmt == 'json':
                    _json_print({'id': task_id})
                else:
                    print(f"Created task: {task_id}")

            manager.close()

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        ctx.close()


if __name__ == "__main__":
    main()