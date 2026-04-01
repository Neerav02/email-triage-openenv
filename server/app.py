"""
server/app.py — OpenEnv multi-mode deployment entry point.
Provides the required main() function and re-exports the Flask app
from api/server so that openenv validate finds everything it needs.
"""
import os
import sys

# Ensure repo root is on path so `api` and `env` packages resolve
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from api.server import app  # noqa: F401


def main():
    """Entry point called by openenv for multi-mode deployment."""
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
