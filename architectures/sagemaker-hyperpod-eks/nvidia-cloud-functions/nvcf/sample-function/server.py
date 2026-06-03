import os
import time
import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, status
from fastapi.responses import StreamingResponse


app = FastAPI(
    title="NVCF Echo Function",
    description="Minimal echo function for validating NVCF on SageMaker HyperPod EKS",
    version="1.0.0",
)


class HealthCheck(BaseModel):
    """Health check response model."""

    status: str = "OK"


class EchoRequest(BaseModel):
    """Echo request model."""

    message: str
    delay: float = 0.0
    repeats: int = 1
    stream: bool = False


# ---------------------------------------------------------------------------
# Health endpoint (required by NVCF)
# Must return HTTP 200 when the container is ready for inference.
# ---------------------------------------------------------------------------
@app.get(
    "/health",
    tags=["healthcheck"],
    summary="Health Check",
    response_description="Return HTTP 200 (OK)",
    status_code=status.HTTP_200_OK,
    response_model=HealthCheck,
)
def get_health() -> HealthCheck:
    return HealthCheck(status="OK")


# ---------------------------------------------------------------------------
# Inference endpoint (called by NVCF during function invocation)
# ---------------------------------------------------------------------------
@app.post("/echo", summary="Echo inference endpoint")
async def echo(request: EchoRequest):
    """
    Echo the message back. Supports optional delay, repetition, and streaming.
    """
    if request.stream:

        def stream_text():
            for _ in range(request.repeats):
                time.sleep(request.delay)
                yield f"data: {request.message}\n\n"

        return StreamingResponse(stream_text(), media_type="text/event-stream")
    else:
        time.sleep(request.delay)
        return {"echo": request.message * request.repeats}


# ---------------------------------------------------------------------------
# Root endpoint (informational)
# ---------------------------------------------------------------------------
@app.get("/", summary="Root endpoint")
def root():
    return {
        "service": "nvcf-echo-function",
        "version": "1.0.0",
        "description": "NVCF echo function on SageMaker HyperPod EKS",
        "endpoints": {
            "health": "/health",
            "inference": "/echo",
        },
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    workers = int(os.getenv("WORKER_COUNT", "10"))
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        workers=workers,
    )
