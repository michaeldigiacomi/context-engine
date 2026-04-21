"""
Configuration for Context Engine.

Reads from environment variables or config file (~/.context_engine/config.json).
Credentials are NEVER hardcoded.
"""

from __future__ import annotations

import os
import json
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _get_default_config_dir() -> Path:
    """Get platform-appropriate config directory."""
    if platform.system() == "Windows":
        base = Path(os.getenv("APPDATA", str(Path.home() / "AppData" / "Roaming")))
    else:
        base = Path(os.getenv("XDG_CONFIG_HOME", str(Path.home() / ".config")))
    return base / "context_engine"


@dataclass
class ContextEngineConfig:
    """
    Configuration for the context engine.

    Loads defaults from environment variables or ~/.context_engine/config.json

    Environment variables (all optional):
        CTX_DB_HOST      - PostgreSQL host
        CTX_DB_PORT      - PostgreSQL port
        CTX_DB_NAME      - Database name
        CTX_DB_USER      - Database user
        CTX_DB_PASS      - Database password
        CTX_DB_SSLMODE   - SSL mode (disable, allow, prefer, require)
        CTX_OLLAMA_URL   - Ollama URL for embeddings
        CTX_EMBEDDING_MODEL - Embedding model name
        CTX_NAMESPACE    - Project namespace for isolation
        CTX_CONFIG_PATH  - Override config file path
    """

    # Database
    db_host: str = field(default_factory=lambda: os.getenv("CTX_DB_HOST", "localhost"))
    db_port: int = field(default_factory=lambda: int(os.getenv("CTX_DB_PORT", "5432")))
    db_name: str = field(default_factory=lambda: os.getenv("CTX_DB_NAME", "context_engine"))
    db_user: str = field(default_factory=lambda: os.getenv("CTX_DB_USER", ""))
    db_pass: str = field(default_factory=lambda: os.getenv("CTX_DB_PASS", ""))
    db_sslmode: str = field(default_factory=lambda: os.getenv("CTX_DB_SSLMODE", "disable"))

    # Embedding
    ollama_url: str = field(
        default_factory=lambda: os.getenv("CTX_OLLAMA_URL", "http://localhost:11434")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("CTX_EMBEDDING_MODEL", "nomic-embed-text")
    )

    # Isolation
    namespace: str = field(default_factory=lambda: os.getenv("CTX_NAMESPACE", "default"))

    # Internals
    _config_file: Optional[Path] = field(default=None, repr=False)

    def __post_init__(self):
        # Load from config file if it exists and credentials are missing
        config_path = os.getenv("CTX_CONFIG_PATH")
        if config_path:
            self._config_file = Path(config_path)
        else:
            self._config_file = _get_default_config_dir() / "config.json"

        self._load_from_file()

    def _load_from_file(self):
        """Load missing values from config file."""
        if not self._config_file or not self._config_file.exists():
            return

        try:
            with open(self._config_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            return

        # Only fill in missing values from file (env takes precedence)
        if not self.db_host or self.db_host == "localhost":
            self.db_host = data.get("db_host", self.db_host)
        if not self.db_user:
            self.db_user = data.get("db_user", self.db_user)
        if not self.db_pass:
            self.db_pass = data.get("db_pass", self.db_pass)
        self.db_name = data.get("db_name", self.db_name)
        self.db_port = data.get("db_port", self.db_port)
        self.db_sslmode = data.get("db_sslmode", self.db_sslmode)
        self.ollama_url = data.get("ollama_url", self.ollama_url)
        self.embedding_model = data.get("embedding_model", self.embedding_model)
        # Only override namespace from config if it's still the default
        if self.namespace == "default":
            self.namespace = data.get("namespace", self.namespace)

    @property
    def conn_string(self) -> str:
        """Build PostgreSQL connection string (with real password for psycopg2)."""
        return (
            f"postgresql://{self.db_user}:{self.db_pass}@{self.db_host}:{self.db_port}/{self.db_name}"
            f"?sslmode={self.db_sslmode}"
        )

    @property
    def conn_string_safe(self) -> str:
        """Build PostgreSQL connection string with masked password (for logging)."""
        return (
            f"postgresql://{self.db_user}:***@{self.db_host}:{self.db_port}/{self.db_name}"
            f"?sslmode={self.db_sslmode}"
        )

    def save_to_file(self, path: Optional[Path] = None):
        """
        Save current config to file (for initial setup).

        Only saves non-sensitive defaults - credentials should be in env or ask user.
        """
        target = path or self._config_file
        if not target:
            return

        target.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "db_host": self.db_host,
            "db_port": self.db_port,
            "db_name": self.db_name,
            "db_user": self.db_user,
            "ollama_url": self.ollama_url,
            "embedding_model": self.embedding_model,
            "namespace": self.namespace,
            # Note: db_pass is NOT saved to file by default
        }
        with open(target, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_env(cls) -> ContextEngineConfig:
        """Create config entirely from environment variables."""
        return cls(
            db_host=os.getenv("CTX_DB_HOST", "localhost"),
            db_port=int(os.getenv("CTX_DB_PORT", "5432")),
            db_name=os.getenv("CTX_DB_NAME", "context_engine"),
            db_user=os.getenv("CTX_DB_USER", ""),
            db_pass=os.getenv("CTX_DB_PASS", ""),
            db_sslmode=os.getenv("CTX_DB_SSLMODE", "disable"),
            ollama_url=os.getenv("CTX_OLLAMA_URL", "http://localhost:11434"),
            embedding_model=os.getenv("CTX_EMBEDDING_MODEL", "nomic-embed-text"),
            namespace=os.getenv("CTX_NAMESPACE", "default"),
            _config_file=None,
        )
