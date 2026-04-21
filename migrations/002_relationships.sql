-- Migration: 002_relationships
-- Adds explicit relationship support between memories

-- Relationships table: directed edges between memories
CREATE TABLE IF NOT EXISTS relationships (
    id SERIAL PRIMARY KEY,
    source_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    target_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    rel_type VARCHAR(50) NOT NULL DEFAULT 'related_to',
    created_at TIMESTAMP DEFAULT NOW(),
    
    -- Prevent duplicate relationships (same source+target+type)
    UNIQUE(source_id, target_id, rel_type)
);

-- Index for looking up relationships from a memory
CREATE INDEX IF NOT EXISTS idx_relationships_source
    ON relationships (source_id);

-- Index for looking up relationships to a memory
CREATE INDEX IF NOT EXISTS idx_relationships_target
    ON relationships (target_id);

-- Index for filtering by type
CREATE INDEX IF NOT EXISTS idx_relationships_type
    ON relationships (rel_type);

-- Index for namespace-scoped relationship lookups
CREATE INDEX IF NOT EXISTS idx_relationships_source_type
    ON relationships (source_id, rel_type);

COMMENT ON TABLE relationships IS 'Explicit typed relationships between memories';
COMMENT ON COLUMN relationships.source_id IS 'Source memory ID';
COMMENT ON COLUMN relationships.target_id IS 'Target memory ID';
COMMENT ON COLUMN relationships.rel_type IS 'Relationship type: related_to, depends_on, supersedes, about, blocks, references, contains, derived_from';