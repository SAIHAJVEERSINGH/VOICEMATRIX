from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routes import router
from app.spectra_routes import router as spectra_router

app = FastAPI(
title=settings.app_name,
description="Experimental voice-clone detection and scam-prevention API.",
version="0.1.0",
)

app.add_middleware(
CORSMiddleware,
allow_origins=[
origin.strip()
for origin in settings.allowed_origins.split(",")
if origin.strip()
],
allow_credentials=False,
allow_methods=["GET", "POST"],
allow_headers=["*"],
)

app.include_router(router, prefix="/api")

app.include_router(spectra_router, prefix="/api")

