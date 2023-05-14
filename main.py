import os
from dotenv import load_dotenv
import uvicorn

load_dotenv()
if __name__ == "__main__":
    port=int(os.environ.get("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, log_level="info")