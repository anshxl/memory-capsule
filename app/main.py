import os
from fastapi import FastAPI # type: ignore
from dotenv import load_dotenv # type: ignore

load_dotenv()

from app.routers import entry # flashback, tune, stats  # type: ignore
app = FastAPI(name="Memory Capsule API")

app.include_router(entry.router, prefix="/entry", tags=["entry"])
# app.include_router(flashback.router, prefix="/flashback", tags=["flashback"])
# app.include_router(tune.router, prefix="/tune", tags=["tune"])
# app.include_router(stats.router, prefix="/stats", tags=["stats"])

