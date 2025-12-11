import os
import uvicorn

port = int(os.environ.get("PORT", 8000))  # Railway set PORT, default 8000 jika local
uvicorn.run("app:app", host="0.0.0.0", port=port)
