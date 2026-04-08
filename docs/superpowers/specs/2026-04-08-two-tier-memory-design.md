# Two-Tier Memory System Design

## Overview

Enhance the PGVector Context Engine with a two-tier memory architecture (Working + Reference) to optimize for both local and remote AI agents. This provides fast session-scoped state for local agents and semantic long-term storage for remote agents, with intelligent coordination between them.

## Problem Statement

Current context engine only provides reference memory (long-term, semantic search). Agents working across multiple projects with markdown files face:

1. **Token waste** - Reading entire files to find relevant sections
2. **Manual relevance decisions** - Guessing which files are relevant by name/path
3. **No cross-project learning** - Solutions from Project A invisible to Project B
4. **Session state gaps** - No place for temporary task state, current errors, active decisions
5. **Context window pressure** - Even with 32k+ tokens, delivering irrelevant context degrades LLM performance

## Solution

### Two-Tier Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MEMORY MANAGER                           │
├────────────────────────┬────────────────────────────────────┤
│     WORKING MEMORY     │          REFERENCE MEMORY          │
│    (working schema)    │         (reference schema)         │
├────────────────────────┼────────────────────────────────────┤
│ • Session context      │ • Project knowledge base          │
│ • Active tasks         │ • Historical patterns             │
│ • Recent decisions     │ • Code examples                   │
│ • Current errors       │ • Cross-project learnings         │
│ • User preferences     │ • Archived discussions            │
├────────────────────────┼────────────────────────────────────┤
│ Fast: No embeddings    │ Semantic: Vector similarity       │
│ TTL: Minutes-hours     │ TTL: Configurable (default: ∞)    │
│ Size limit: 200 items  │ Size: Unbounded (DB-limited)      │
│ Auto-expire on close   │ Explicit cleanup only             │
└────────────────────────┴────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  MEMORY MANAGER │
                    │  Orchestration  │
                    └─────────────────┘
```

### Design Principles

1. **Single database, multiple schemas** - One PostgreSQL instance, separate `working` and `reference` schemas
2. **No embedding overhead for working memory** - Direct key-value and query-based access
3. **Strict token budgets** - Never exceed `max_tokens`, truncate intelligently
4. **Relevance over volume** - Better to deliver 2k highly relevant tokens than 32k of noise
5. **Cross-project optional** - Namespaces isolate by default, but can search across for patterns

## Components

### 1. WorkingMemory (New)

Session-scoped, fast-access storage for temporary state.

**Responsibilities:**
- Task tracking (status, priority, assignment)
- Session context (preferences, active state)
- Recent decisions (last N, auto-expire)
- Current errors (transient issues)

**Key behaviors:**
- Soft limit: 100 entries (warning)
- Hard limit: 200 entries (LRU eviction)
- TTL enforcement: Background cleanup of expired entries
- No embeddings: Direct SQL queries

**Schema:**
```sql
CREATE SCHEMA working;

CREATE TABLE working.session_context (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    priority INTEGER DEFAULT 5,  -- 1-10, higher = keep longer
    ttl_minutes INTEGER DEFAULT 60,
    last_accessed TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE working.tasks (
    task_id TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    plan JSONB,  -- Array of steps
    status TEXT DEFAULT 'planning',  -- planning, ready, executing, done, error
    assigned_to TEXT,
    priority INTEGER DEFAULT 5,
    result JSONB,  -- Output data
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE working.recent_decisions (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    category TEXT DEFAULT 'decision',
    context TEXT,  -- What prompted this decision
    ttl_minutes INTEGER DEFAULT 480,  -- 8 hours default
    created_at TIMESTAMP DEFAULT NOW()
);
```

### 2. MemoryManager (New)

Orchestrates access to working and reference memory with intelligent context assembly.

**Responsibilities:**
- Route saves to appropriate tier (`working` vs `reference`)
- Assemble context respecting token budgets
- Auto-promote highly relevant reference matches to working memory
- Handle cross-namespace search (optional)

**Token budget allocation:**
```python
DEFAULT_TOKEN_BUDGETS = {
    "local-8k": 4000,
    "local-32k": 8000,
    "claude-haiku": 6000,
    "claude-sonnet": 8000,
    "claude-opus": 12000,
    "gpt-4o": 8000,
}
```

**Context assembly priority:**
1. Working memory (max 20% of budget)
2. Recent decisions (max 10%)
3. Reference matches (remaining, ranked)

**Ranking signals (normalized, weighted):**
- Vector similarity: 50%
- Importance score: 20%
- Recency decay: 20%
- Access frequency: 10%

### 3. Reference Memory (Existing)

Existing `ContextEngine` class, moved to `reference` schema.

**Enhancements:**
- Optional chunking for granularity
- Cross-namespace search
- Pre-computed bullet summaries

**Chunking (optional):**
```sql
CREATE TABLE reference.chunks (
    id SERIAL PRIMARY KEY,
    memory_id INTEGER REFERENCES reference.memories(id),
    content TEXT NOT NULL,
    chunk_index INTEGER,
    embedding VECTOR(768)
);
```

### 4. Cross-Project Search

Enable learning patterns across projects via namespace inclusion.

```python
# Search current namespace + specific others
context = manager.get_context(
    query="authentication pattern",
    include_namespaces=["project-a", "project-b"]
)

# Search all namespaces
context = manager.get_context(
    query="common pattern",
    include_namespaces=["*"]
)
```

## Coordination: Local + Remote Agents

### Local Agent Workflow (Fast, Conversational)

```python
# Local agent initializes
manager = MemoryManager(model_type="local-8k")

# Conversation loop
while True:
    message = get_user_input()
    
    # Fast working memory lookup (no embeddings)
    context = manager.working.get_session_context()
    
    # Only search reference if working memory insufficient
    if not has_enough_context(context, message):
        ref = manager.reference.get_context(message, max_tokens=2000)
        context += ref
    
    response = local_llm(context, message)
    
    # Create task for remote if needed
    if is_task_request(message):
        manager.working.save_task(
            description=extract_task(message),
            status="ready",
            assigned_to="remote-agent"
        )
```

### Remote Agent Workflow (Powerful, Execution)

```python
# Remote agent picks up task
task = manager.working.get_ready_task()

# Load rich context from reference
context = manager.get_context(
    query=task["description"],
    max_tokens=12000,  # Can use more
    include_namespaces=["*"]  # Cross-project patterns
)

# Execute task
result = remote_llm(context, task)

# Save learnings to reference
manager.reference.save(
    content=f"Pattern learned: {result['pattern']}",
    category="pattern",
    importance=8
)

# Mark task done
manager.working.update_task(task["id"], status="done", result=result)
```

## API Design

### MemoryManager (Public API)

```python
class MemoryManager:
    def __init__(self, config=None, model_type="claude-sonnet"):
        self.working = WorkingMemory(config)
        self.reference = ContextEngine(config)
        self.model_type = model_type
    
    def get_context(self, query: str, max_tokens: int = None, 
                   include_namespaces: List[str] = None) -> str:
        """Get assembled context respecting token budget."""
        pass
    
    def remember(self, content: str, tier: str = "reference", **kwargs) -> str:
        """Save to appropriate tier."""
        pass
    
    def save_task(self, description: str, **kwargs) -> str:
        """Save task to working memory."""
        pass
    
    def get_ready_tasks(self) -> List[Dict]:
        """Get tasks ready for execution."""
        pass
    
    def close(self):
        """Cleanup, optionally clear working memory."""
        pass
```

### WorkingMemory (Internal)

```python
class WorkingMemory:
    def get_session_context(self) -> Dict[str, str]:
        """Get all session context key-values."""
        pass
    
    def set_session_context(self, key: str, value: str, 
                           priority: int = 5, ttl_minutes: int = 60):
        """Set session context with TTL."""
        pass
    
    def save_task(self, task_id: str, description: str, **kwargs) -> str:
        """Save task, auto-generate ID if not provided."""
        pass
    
    def get_tasks(self, status: str = None, limit: int = 50) -> List[Dict]:
        """Get tasks, optionally filtered by status."""
        pass
    
    def update_task(self, task_id: str, **kwargs):
        """Update task fields."""
        pass
    
    def save_decision(self, content: str, context: str = None, 
                     ttl_minutes: int = 480) -> int:
        """Save decision with auto-expire."""
        pass
    
    def get_recent_decisions(self, limit: int = 10) -> List[Dict]:
        """Get recent decisions."""
        pass
    
    def cleanup_expired(self) -> int:
        """Delete expired entries, return count."""
        pass
```

## Data Flow

### Save Flow

```
Agent calls remember(content, tier="working")
                    │
                    ▼
            ┌───────────────┐
            │ MemoryManager │
            └───────────────┘
                    │
            ┌───────┴───────┐
            ▼               ▼
    ┌───────────┐   ┌───────────┐
    │  Working  │   │ Reference │
    │  (direct  │   │ (embed +  │
    │   SQL)    │   │  insert)  │
    └───────────┘   └───────────┘
```

### Retrieve Flow

```
Agent calls get_context(query, max_tokens=8000)
                    │
                    ▼
            ┌───────────────┐
            │ MemoryManager │
            └───────────────┘
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
   ┌────────┐  ┌────────┐  ┌─────────┐
   │Working │  │Recent  │  │Reference│
   │(20%)   │  │(10%)   │  │(70%)    │
   └────┬───┘  └────┬───┘  └────┬────┘
        │           │           │
        │           │     (rank, truncate)
        └───────────┼───────────┘
                    ▼
            ┌───────────────┐
            │ Assemble with │
            │ section headers│
            └───────────────┘
                    │
                    ▼
              Context String
```

## Migration Path

### Phase 1: Add Working Schema
- Add `working` schema alongside existing `public` schema
- Existing `ContextEngine` continues working unchanged
- New `WorkingMemory` class available

### Phase 2: Add MemoryManager
- New `MemoryManager` orchestrates both tiers
- Backward compatible: `ContextEngine` still works standalone

### Phase 3: Migrate Existing Code (optional)
- Existing code can gradually migrate to `MemoryManager`
- Or continue using `ContextEngine` directly

## Error Handling

**Working memory full (200 items):**
- Evict lowest priority + oldest items
- Log warning
- Continue operation

**Reference search fails:**
- Return working memory only
- Log error
- Don't fail the request

**Token budget exceeded during assembly:**
- Truncate from lowest priority section
- Never return more than max_tokens

## Testing Strategy

**Unit tests:**
- Mock both working and reference layers
- Test token budget enforcement
- Test ranking algorithms
- Test LRU eviction

**Integration tests:**
- Real PostgreSQL with both schemas
- Test cross-namespace search
- Test TTL expiration
- Test task lifecycle

## Future Enhancements (Out of Scope)

- **Sync to external systems** - Webhooks for task creation
- **Compression layer** - On-demand summarization for very tight budgets
- **Auto-archival** - Move old working memory to reference
- **Conflict resolution** - Merge strategy when same key set multiple times

## Success Metrics

1. **Token efficiency** - Average tokens delivered vs requested < 90%
2. **Relevance** - Manual rating of retrieved context quality > 4/5
3. **Speed** - Working memory queries < 10ms
4. **Cross-project hits** - % of queries that find matches in other namespaces > 20%
