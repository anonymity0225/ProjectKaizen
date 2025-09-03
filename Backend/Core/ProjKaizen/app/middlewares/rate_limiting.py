# Rate limiting middleware for FastAPI 
# Rate limiting and metrics middleware (for future implementation)
import time
import logging
from typing import Dict, Optional
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class RateLimitMiddleware:
    """
    Simple in-memory rate limiting middleware.
    For production, consider using Redis or other persistent storage.
    """
    
    def __init__(
        self, 
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        cleanup_interval: int = 300  # 5 minutes
    ):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.cleanup_interval = cleanup_interval
        
        # Store request timestamps per IP
        self.request_history: Dict[str, deque] = defaultdict(deque)
        self.last_cleanup = time.time()
    
    def _cleanup_old_requests(self):
        """Remove old request records to prevent memory leaks."""
        current_time = time.time()
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
            
        cutoff_time = current_time - 3600  # Keep 1 hour of history
        
        for ip, requests in self.request_history.items():
            # Remove requests older than 1 hour
            while requests and requests[0] < cutoff_time:
                requests.popleft()
        
        # Remove empty entries
        empty_ips = [ip for ip, requests in self.request_history.items() if not requests]
        for ip in empty_ips:
            del self.request_history[ip]
            
        self.last_cleanup = current_time
        logger.debug(f"Cleaned up rate limit history. Active IPs: {len(self.request_history)}")
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers first (if behind proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
            
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
            
        # Fallback to direct client IP
        return request.client.host if request.client else "unknown"
    
    def check_rate_limit(self, request: Request) -> Optional[JSONResponse]:
        """
        Check if request should be rate limited.
        Returns None if allowed, JSONResponse if rate limited.
        """
        try:
            self._cleanup_old_requests()
            
            client_ip = self._get_client_ip(request)
            current_time = time.time()
            
            # Get request history for this IP
            requests = self.request_history[client_ip]
            
            # Count requests in the last minute and hour
            minute_ago = current_time - 60
            hour_ago = current_time - 3600
            
            recent_requests = [req_time for req_time in requests if req_time > minute_ago]
            hourly_requests = [req_time for req_time in requests if req_time > hour_ago]
            
            # Check limits
            if len(recent_requests) >= self.requests_per_minute:
                logger.warning(f"Rate limit exceeded (per minute) for IP {client_ip}: {len(recent_requests)} requests")
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "detail": f"Rate limit exceeded: {self.requests_per_minute} requests per minute",
                        "retry_after": 60
                    },
                    headers={"Retry-After": "60"}
                )
            
            if len(hourly_requests) >= self.requests_per_hour:
                logger.warning(f"Rate limit exceeded (per hour) for IP {client_ip}: {len(hourly_requests)} requests")
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "detail": f"Rate limit exceeded: {self.requests_per_hour} requests per hour",
                        "retry_after": 3600
                    },
                    headers={"Retry-After": "3600"}
                )
            
            # Record this request
            requests.append(current_time)
            
            # Keep only recent requests to save memory
            while requests and requests[0] < hour_ago:
                requests.popleft()
            
            return None  # Request allowed
            
        except Exception as e:
            logger.error(f"Error in rate limiting: {str(e)}")
            # On error, allow the request to proceed
            return None

class MetricsCollector:
    """
    Simple metrics collector for monitoring preprocessing operations.
    For production, consider using Prometheus or similar monitoring system.
    """
    
    def __init__(self):
        self.reset_metrics()
    
    def reset_metrics(self):
        """Reset all metrics."""
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_processing_time = 0.0
        self.file_sizes_processed = []
        self.error_counts = defaultdict(int)
        self.start_time = time.time()
    
    def record_request_start(self) -> float:
        """Record the start of a request. Returns timestamp for duration calculation."""
        self.total_requests += 1
        return time.time()
    
    def record_request_success(self, start_time: float, file_size: Optional[int] = None):
        """Record a successful request."""
        self.successful_requests += 1
        processing_time = time.time() - start_time
        self.total_processing_time += processing_time
        
        if file_size:
            self.file_sizes_processed.append(file_size)
        
        logger.info(f"Request completed successfully in {processing_time:.2f}s")
    
    def record_request_failure(self, start_time: float, error_type: str):
        """Record a failed request."""
        self.failed_requests += 1
        self.error_counts[error_type] += 1
        processing_time = time.time() - start_time
        self.total_processing_time += processing_time
        
        logger.warning(f"Request failed after {processing_time:.2f}s: {error_type}")
    
    def get_metrics(self) -> Dict:
        """Get current metrics summary."""
        uptime = time.time() - self.start_time
        avg_processing_time = (
            self.total_processing_time / self.total_requests 
            if self.total_requests > 0 else 0
        )
        avg_file_size = (
            sum(self.file_sizes_processed) / len(self.file_sizes_processed)
            if self.file_sizes_processed else 0
        )
        
        return {
            "uptime_seconds": uptime,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": (
                self.successful_requests / self.total_requests 
                if self.total_requests > 0 else 0
            ),
            "average_processing_time_seconds": avg_processing_time,
            "total_processing_time_seconds": self.total_processing_time,
            "files_processed": len(self.file_sizes_processed),
            "average_file_size_bytes": avg_file_size,
            "error_breakdown": dict(self.error_counts)
        }

# Global instances (for simple deployment)
rate_limiter = RateLimitMiddleware()
metrics_collector = MetricsCollector()

async def rate_limit_middleware(request: Request, call_next):
    """
    FastAPI middleware function for rate limiting.
    
    Usage in main.py:
    from fastapi.middleware.base import BaseHTTPMiddleware
    app.add_middleware(BaseHTTPMiddleware, dispatch=rate_limit_middleware)
    """
    # Skip rate limiting for health checks and metrics endpoints
    if request.url.path in ["/health", "/metrics", "/docs", "/redoc", "/openapi.json"]:
        response = await call_next(request)
        return response
    
    # Check rate limit
    rate_limit_response = rate_limiter.check_rate_limit(request)
    if rate_limit_response:
        return rate_limit_response
    
    # Record request start
    start_time = metrics_collector.record_request_start()
    
    try:
        # Process request
        response = await call_next(request)
        
        # Record success
        if response.status_code < 400:
            # Try to get file size from request if it's a file upload
            file_size = None
            if hasattr(request, 'form'):
                try:
                    form = await request.form()
                    if 'file' in form:
                        file_size = len(await form['file'].read())
                except:
                    pass
            
            metrics_collector.record_request_success(start_time, file_size)
        else:
            metrics_collector.record_request_failure(start_time, f"HTTP_{response.status_code}")
        
        return response
        
    except Exception as e:
        # Record failure
        metrics_collector.record_request_failure(start_time, type(e).__name__)
        raise

# Metrics endpoint (add to your router)
def create_metrics_router():
    """Create a router with metrics endpoints."""
    from fastapi import APIRouter
    
    router = APIRouter(prefix="/metrics", tags=["Monitoring"])
    
    @router.get("/", summary="Get system metrics")
    async def get_metrics():
        """Get current system metrics and statistics."""
        return metrics_collector.get_metrics()
    
    @router.post("/reset", summary="Reset metrics")
    async def reset_metrics():
        """Reset all metrics counters."""
        metrics_collector.reset_metrics()
        return {"message": "Metrics reset successfully"}
    
    return router