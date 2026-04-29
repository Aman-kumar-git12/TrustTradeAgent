import uvicorn


if __name__ == '__main__':
    import os
    port = int(os.getenv("PORT", os.getenv("AGENT_PORT", "8000")))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
