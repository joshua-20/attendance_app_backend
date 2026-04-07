"""
Entry point — start the Attendance System API server.

Usage:
    python run.py
"""

import os
import uvicorn
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    # Disable reload in production (Railway/Render/Fly set PORT env var)
    is_production = os.getenv("RAILWAY_ENVIRONMENT") or os.getenv("RENDER") or os.getenv("FLY_APP_NAME")
    reload = False if is_production else os.getenv("RELOAD", "true").lower() == "true"

    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=reload,
        log_level="info",
    )
