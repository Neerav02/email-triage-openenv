"""
server/app.py  — OpenEnv multi-mode deployment entry point.
Re-exports the Flask `app` from api/server so that the openenv
validator can find it at the expected location.
"""
from api.server import app  # noqa: F401

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
