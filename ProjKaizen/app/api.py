# Main FastAPI app instance 
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import preprocess  # adjust if not using 'app' prefix
from app.routers import transformation
from app.middlewares.rate_limiting import RateLimitMiddleware  # optional

app = FastAPI(
    title="Kaizen AI",
    description="Modular and Advanced AI backend for preprocessing, transformation, modeling, and visualization.",
    version="1.0.0"
)

# CORS (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting middleware (optional)
# app.add_middleware(RateLimiterMiddleware)

# Register routers
app.include_router(preprocess.router, prefix="/preprocess", tags=["Preprocessing"])
app.include_router(transformation.router, prefix="/transform", tags=["Transformation"])

# Health check
@app.get("/health", tags=["System"])
def health_check():
    return {"status": "ok"}
