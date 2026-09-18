"""Run the debugger backend.

    python run.py                 # http://localhost:8000
    MDEBUG_PORT=9000 python run.py

The client expects this at `environment.apiBase`. See docs/api-contract.md.
"""

import os
import sys
from pathlib import Path

import uvicorn

# Runnable from anywhere: `python server/run.py` from the repo root works, and
# so does `python run.py` from inside server/.
HERE = Path(__file__).resolve().parent
os.chdir(HERE)
sys.path.insert(0, str(HERE))

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=os.environ.get("MDEBUG_HOST", "127.0.0.1"),
        port=int(os.environ.get("MDEBUG_PORT", 8000)),
        reload=bool(os.environ.get("MDEBUG_RELOAD")),
        log_level=os.environ.get("MDEBUG_LOG_LEVEL", "info"),
    )
