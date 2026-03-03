"""
config.py - API Key & Configuration Management
================================================
Handles API key loading in priority order:
  1. Directly passed in code (highest priority)
  2. .env file in project root
  3. ANTHROPIC_API_KEY environment variable (fallback)

HOW TO GET YOUR API KEY:
  1. Go to: https://console.anthropic.com
  2. Click "Account Settings" → "API Keys"
  3. Click "Create Key", give it a name
  4. COPY IT IMMEDIATELY — it's only shown once
  5. Paste it in your .env file (see below)

SETUP:
  Create a file named .env in the same folder as this project:
  ┌─────────────────────────────────────────┐
  │  ANTHROPIC_API_KEY=sk-ant-api03-...     │
  │  I2M_MODEL=claude-sonnet-4-20250514     │
  │  I2M_OPTIMIZE_LEVEL=3                   │
  └─────────────────────────────────────────┘
"""

import os
from pathlib import Path


# ─────────────────────────────────────────────
# .env LOADER (no external deps needed)
# ─────────────────────────────────────────────

def _load_env_file(env_path: str = ".env"):
    """
    Load key=value pairs from a .env file into os.environ.
    Skips comments (#) and blank lines.
    Does NOT override already-set environment variables.
    """
    path = Path(env_path)
    if not path.exists():
        # Try looking in the script's directory
        path = Path(__file__).parent / ".env"

    if not path.exists():
        return  # No .env file — that's fine

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")  # Remove optional quotes
            if key and key not in os.environ:
                os.environ[key] = value


# Load .env on import
_load_env_file()


# ─────────────────────────────────────────────
# CONFIG CLASS
# ─────────────────────────────────────────────

class Config:
    """
    Central configuration for Intent2Machine.
    Reads from environment (which includes .env file loaded above).
    """

    # ── Anthropic API ──────────────────────────────────────────
    # Get your key at: https://console.anthropic.com/settings/keys
    ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")

    # ── Model selection ────────────────────────────────────────
    # claude-sonnet-4-20250514  ← Best balance of speed + quality (recommended)
    # claude-opus-4-6           ← Most powerful, slower, more expensive
    # claude-haiku-4-5-20251001 ← Fastest, cheapest, less accurate
    MODEL: str = os.environ.get("I2M_MODEL", "claude-sonnet-4-20250514")

    # ── Compilation settings ───────────────────────────────────
    OPTIMIZE_LEVEL: int = int(os.environ.get("I2M_OPTIMIZE_LEVEL", "3"))  # 0-3
    DEFAULT_TARGET_ARCH: str = os.environ.get("I2M_TARGET_ARCH", "native")
    DEFAULT_OUTPUT_TYPE: str = os.environ.get("I2M_OUTPUT_TYPE", "executable")

    # ── Pipeline flags ─────────────────────────────────────────
    SKIP_AI_OPTIMIZATION: bool = os.environ.get("I2M_SKIP_AI_OPT", "false").lower() == "true"
    SKIP_SAFETY_CHECK: bool = os.environ.get("I2M_SKIP_SAFETY", "false").lower() == "true"
    VERBOSE: bool = os.environ.get("I2M_VERBOSE", "true").lower() == "true"

    # ── API endpoint (advanced) ────────────────────────────────
    # Only change this if using a proxy or custom endpoint
    API_BASE_URL: str = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

    @classmethod
    def validate(cls) -> None:
        """
        Validate that required config is present.
        Raises clear error with instructions if API key is missing.
        """
        if not cls.ANTHROPIC_API_KEY:
            raise EnvironmentError(
                "\n"
                "╔══════════════════════════════════════════════════════════╗\n"
                "║          ANTHROPIC API KEY NOT FOUND                    ║\n"
                "╠══════════════════════════════════════════════════════════╣\n"
                "║  To get your API key:                                   ║\n"
                "║    1. Go to: https://console.anthropic.com              ║\n"
                "║    2. Click Account Settings → API Keys                 ║\n"
                "║    3. Click 'Create Key' and copy it immediately        ║\n"
                "║                                                         ║\n"
                "║  Then set it ONE of these ways:                         ║\n"
                "║                                                         ║\n"
                "║  Option A — .env file (recommended):                   ║\n"
                "║    Create a .env file in your project folder:           ║\n"
                "║    ANTHROPIC_API_KEY=sk-ant-api03-YOUR-KEY-HERE         ║\n"
                "║                                                         ║\n"
                "║  Option B — Environment variable:                      ║\n"
                "║    Linux/Mac:  export ANTHROPIC_API_KEY=sk-ant-...      ║\n"
                "║    Windows:    set ANTHROPIC_API_KEY=sk-ant-...         ║\n"
                "║                                                         ║\n"
                "║  Option C — Pass directly in code:                     ║\n"
                "║    pipeline = Intent2MachinePipeline(                   ║\n"
                "║        api_key='sk-ant-api03-YOUR-KEY-HERE'             ║\n"
                "║    )                                                    ║\n"
                "╚══════════════════════════════════════════════════════════╝"
            )

        if not cls.ANTHROPIC_API_KEY.startswith("sk-ant-"):
            print(
                "  ⚠  Warning: API key doesn't look like a valid Anthropic key.\n"
                "     Expected format: sk-ant-api03-...\n"
                "     Get your key at: https://console.anthropic.com/settings/keys"
            )

    @classmethod
    def summary(cls) -> str:
        key = cls.ANTHROPIC_API_KEY
        masked = f"{key[:12]}...{key[-4:]}" if len(key) > 16 else "NOT SET"
        return (
            f"Config:\n"
            f"  API Key   : {masked}\n"
            f"  Model     : {cls.MODEL}\n"
            f"  Opt Level : -O{cls.OPTIMIZE_LEVEL}\n"
            f"  Arch      : {cls.DEFAULT_TARGET_ARCH}\n"
            f"  AI Opt    : {'disabled' if cls.SKIP_AI_OPTIMIZATION else 'enabled'}\n"
            f"  Safety    : {'disabled' if cls.SKIP_SAFETY_CHECK else 'enabled'}"
        )


# Singleton instance
config = Config()
