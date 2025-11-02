from __future__ import annotations

import uvicorn

# Direct import for standalone execution: `python main.py`
from app import app


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


