"""Integration tests for two-tier memory system."""

import pytest
import os
import time

pytestmark = pytest.mark.integration


def test_working_memory_full_lifecycle(postgres_available):
    """Test complete working memory lifecycle with real PostgreSQL."""
    if not postgres_available:
        pytest.skip("PostgreSQL not available")

    from context_engine.memory_manager import MemoryManager
    from context_engine.config import ContextEngineConfig

    config = ContextEngineConfig(
        db_host=os.getenv("CTX_DB_HOST", "localhost"),
        db_port=int(os.getenv("CTX_DB_PORT", "5432")),
        db_name=os.getenv("CTX_DB_NAME", "context_engine"),
        db_user=os.getenv("CTX_DB_USER", ""),
        db_pass=os.getenv("CTX_DB_PASS", ""),
        namespace="test-two-tier"
    )

    manager = MemoryManager(config=config)

    # Save session context
    manager.working.set_session_context("test_key", "test_value")

    # Retrieve
    ctx = manager.working.get_session_context()
    assert ctx["test_key"] == "test_value"

    # Save task
    task_id = manager.working.save_task(
        description="Test task",
        plan=["Step 1", "Step 2"],
        status="ready"
    )
    assert task_id is not None

    # Retrieve task
    tasks = manager.working.get_tasks(status="ready")
    assert len(tasks) >= 1

    # Update task
    manager.working.update_task(task_id, status="done")
    tasks = manager.working.get_tasks(status="done")
    assert any(t["task_id"] == task_id for t in tasks)

    manager.close()


def test_memory_manager_context_assembly(postgres_available):
    """Test context assembly with both tiers."""
    if not postgres_available:
        pytest.skip("PostgreSQL not available")

    from context_engine.memory_manager import MemoryManager
    from context_engine.config import ContextEngineConfig

    config = ContextEngineConfig(namespace="test-assembly")
    manager = MemoryManager(config=config)

    # Save to reference
    manager.reference.save(
        content="Project uses FastAPI framework",
        category="tech_stack",
        importance=8
    )

    # Save to working
    manager.working.set_session_context("current_task", "refactor auth")

    # Get assembled context
    context = manager.get_context("What framework to use?", max_tokens=2000)

    # Verify both tiers present
    assert "current_task" in context
    assert "FastAPI" in context

    manager.close()


def test_ttl_expiration(postgres_available):
    """Test that expired entries are cleaned up."""
    if not postgres_available:
        pytest.skip("PostgreSQL not available")

    from context_engine.working_memory import WorkingMemory
    from context_engine.config import ContextEngineConfig

    config = ContextEngineConfig(namespace="test-ttl")
    wm = WorkingMemory(config)

    # Save with 1 second TTL
    wm.set_session_context("temp", "value", ttl_minutes=0.0167)  # ~1 second

    # Verify exists
    ctx = wm.get_session_context()
    assert ctx.get("temp") == "value"

    # Wait for expiration
    time.sleep(2)

    # Cleanup
    wm.cleanup_expired()

    # Verify gone
    ctx = wm.get_session_context()
    assert "temp" not in ctx

    wm.close()
