from api.main import app


if __name__ == "__main__":
    import uvicorn

    import os
    port = int(os.getenv("PORT", os.getenv("AGENT_PORT", "8000")))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=debug)
