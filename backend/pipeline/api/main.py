from fastapi import FastAPI
app = FastAPI(title="AdityaScan API", version="1.0.0")

@app.get("/api/health")
def health():
    return {"status": "ok"}
