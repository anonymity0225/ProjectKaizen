# app/api.py
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import uvicorn

# Import routers
from app.routers import preprocessing, transformation, modeling, visualization

# Import settings and utilities
from app.config import settings
from app.core.auth import verify_token
from app.core.model_registry import ModelRegistry
from app.core.exceptions import KaizenException

# Prometheus metrics
REQUEST_COUNT = Counter(
    'kaizen_http_requests_total',
    'Total number of HTTP requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_DURATION = Histogram(
    'kaizen_http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)

# Global variables for shared resources
model_registry: ModelRegistry = None
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer(auto_error=False)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Authentication dependency for protected routes."""
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Authentication required"
        )
    
    try:
        user = await verify_token(credentials.credentials)
        return user
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials"
        )

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    logger.info("Starting Kaizen Data Pipeline API...")
    
    try:
        # Initialize model registry
        global model_registry
        model_registry = ModelRegistry()
        await model_registry.initialize()
        
        # Set up logging
        logging.basicConfig(
            level=getattr(logging, settings.LOG_LEVEL.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        logger.info("Application startup completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Kaizen Data Pipeline API...")
    
    try:
        # Cleanup model registry
        if model_registry:
            await model_registry.cleanup()
        
        # Additional cleanup tasks
        logger.info("Application shutdown completed successfully")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

# Create FastAPI instance with lifespan
app = FastAPI(
    title="Kaizen Data Pipeline API",
    description="Advanced data processing pipeline with ML capabilities for continuous improvement",
    version="1.0.0",
    docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
    openapi_url="/openapi.json" if settings.ENVIRONMENT != "production" else None,
    lifespan=lifespan
)

# Security middleware
if settings.ENVIRONMENT == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.ALLOWED_HOSTS or ["*"]
    )

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"]
)

# Request tracking middleware
@app.middleware("http")
async def request_tracking_middleware(request: Request, call_next):
    """Middleware for logging and monitoring HTTP requests."""
    start_time = time.time()
    request_id = f"req_{int(time.time() * 1000)}"
    
    # Add request ID to headers
    request.state.request_id = request_id
    
    # Log request
    logger.info(
        f"Request {request_id}: {request.method} {request.url.path} "
        f"from {request.client.host if request.client else 'unknown'}"
    )
    
    try:
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Update metrics
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code
        ).inc()
        
        REQUEST_DURATION.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)
        
        # Add response headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration:.3f}s"
        
        # Log response
        logger.info(
            f"Response {request_id}: {response.status_code} "
            f"in {duration:.3f}s"
        )
        
        return response
        
    except Exception as e:
        duration = time.time() - start_time
        
        # Update error metrics
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code=500
        ).inc()
        
        logger.error(f"Request {request_id} failed in {duration:.3f}s: {e}")
        raise

# Authentication middleware for protected routes
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Authentication middleware for all routes except public ones."""
    # Public routes that don't require authentication
    public_routes = {"/health", "/metrics", "/docs", "/redoc", "/openapi.json"}
    
    if request.url.path in public_routes or request.url.path.startswith("/static"):
        return await call_next(request)
    
    # For protected routes, verify authentication
    auth_header = request.headers.get("authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return JSONResponse(
            status_code=401,
            content={"error": "Authentication required", "detail": "Missing or invalid authorization header"}
        )
    
    try:
        token = auth_header.split(" ")[1]
        user = await verify_token(token)
        request.state.user = user
        return await call_next(request)
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        return JSONResponse(
            status_code=401,
            content={"error": "Authentication failed", "detail": str(e)}
        )

# Include routers with authentication dependency
app.include_router(
    preprocessing.router,
    prefix="/api/v1/preprocess",
    tags=["Preprocessing"],
    dependencies=[Depends(get_current_user)]
)

app.include_router(
    transformation.router,
    prefix="/api/v1/transform",
    tags=["Transformation"],
    dependencies=[Depends(get_current_user)]
)

app.include_router(
    modeling.router,
    prefix="/api/v1/modeling",
    tags=["Modeling"],
    dependencies=[Depends(get_current_user)]
)

app.include_router(
    visualization.router,
    prefix="/api/v1/visualize",
    tags=["Visualization"],
    dependencies=[Depends(get_current_user)]
)

# Health check endpoint
@app.get("/health", tags=["Health"], response_model=Dict[str, Any])
def health_check():
    """Health check endpoint for load balancers and monitoring."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0",
        "environment": settings.ENVIRONMENT,
        "services": {
            "model_registry": "healthy" if model_registry and model_registry.is_healthy() else "unhealthy",
            "database": "healthy",  # Add actual DB health check
            "redis": "healthy"      # Add actual Redis health check
        }
    }

# Metrics endpoint for Prometheus
@app.get("/metrics", tags=["Monitoring"])
def get_metrics():
    """Prometheus metrics endpoint."""
    return PlainTextResponse(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

# Custom exception handlers
@app.exception_handler(KaizenException)
async def kaizen_exception_handler(request: Request, exc: KaizenException):
    """Handler for custom Kaizen exceptions."""
    logger.error(f"Kaizen exception in {request.url.path}: {exc}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.error_type,
            "detail": exc.detail,
            "request_id": getattr(request.state, "request_id", None)
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handler for HTTP exceptions."""
    logger.warning(f"HTTP exception in {request.url.path}: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Error",
            "detail": exc.detail,
            "request_id": getattr(request.state, "request_id", None)
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled exceptions."""
    logger.exception(f"Unhandled exception in {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred. Please try again later.",
            "request_id": getattr(request.state, "request_id", None)
        }
    )

# Root endpoint
@app.get("/", tags=["Root"])
def read_root():
    """Root endpoint with API information."""
    return {
        "name": "Kaizen Data Pipeline API",
        "version": "1.0.0",
        "status": "running",
        "docs_url": "/docs" if settings.ENVIRONMENT != "production" else None,
        "health_url": "/health",
        "metrics_url": "/metrics"
    }

# For running with uvicorn directly
if __name__ == "__main__":
    uvicorn.run(
        "app.api:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.ENVIRONMENT == "development",
        log_level=settings.LOG_LEVEL.lower()
    )