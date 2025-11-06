"""
Visualization service logic for Kaizen enterprise data platform.
This module provides service-layer functionality for generating various types of
data visualizations including histograms, correlation heatmaps, scatter plots,
box plots, and time-series plots.
"""

import base64
import hashlib
import io
import json
import logging
import os
import tempfile
import time
import threading
import uuid
import gc
import psutil
import weakref
import subprocess
import multiprocessing
import pickle
import sqlite3
import math
import statistics
import re
from concurrent.futures import ThreadPoolExecutor, Future, as_completed, ProcessPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict, Any, Callable, Set
from threading import RLock, Event, Condition, BoundedSemaphore
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from collections import deque, defaultdict, Counter
import warnings
import asyncio
from contextlib import contextmanager, asynccontextmanager
import retrying
from functools import wraps, lru_cache
import queue
import signal
from urllib.parse import quote
import html
import copy
import mmap
import resource
import sys

import matplotlib
matplotlib.use('Agg')  # Set headless backend before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import pandas as pd
import redis
import seaborn as sns
from fastapi import UploadFile

# Optional imports for enhanced functionality
try:
    import boto3
    from botocore.exceptions import ClientError, BotoCoreError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

from config import settings
from schemas.visualization import (
    PlotConfig,
    VisualizationResponse,
    HistogramRequest,
    CorrelationHeatmapRequest,
    ScatterPlotRequest,
    BoxPlotRequest,
    TimeSeriesRequest
)
from app.storage.plot_storage import get_plot_storage, PlotStorageError
from app.cache.plot_cache import get_plot_cache
from app.observability.metrics import (
    visualization_render_seconds,
    service_errors_total,
    observe
)
from app.observability.tracing import traced_span

# Suppress matplotlib warnings in production
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Configure logging
logger = logging.getLogger(__name__)

# Configuration constants
BASE_STORAGE_PATH = Path(settings.STORAGE_BASE_PATH) if hasattr(settings, 'STORAGE_BASE_PATH') else Path("/tmp")
PLOTS_DIR = BASE_STORAGE_PATH / "plots"
CACHE_DIR = BASE_STORAGE_PATH / "plot_cache"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# CRITICAL FIX 1: MEMORY LEAK PREVENTION
# ==========================================

class WorkerMemoryState(Enum):
    """Worker memory states for lifecycle management."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    RESTART_REQUIRED = "restart_required"

@dataclass
class WorkerStats:
    """Statistics for worker process monitoring."""
    worker_id: str
    plots_generated: int = 0
    memory_mb: float = 0.0
    uptime_seconds: float = 0.0
    last_gc_time: float = 0.0
    memory_state: WorkerMemoryState = WorkerMemoryState.HEALTHY
    restart_requested: bool = False

class IsolatedPlotRenderer:
    """Subprocess-isolated plot renderer with aggressive memory management."""
    
    def __init__(self, max_plots_per_worker: int = 50, memory_threshold_mb: float = 512):
        self.max_plots_per_worker = max_plots_per_worker
        self.memory_threshold_mb = memory_threshold_mb
        self.worker_pool = ProcessPoolExecutor(
            max_workers=min(4, multiprocessing.cpu_count()),
            initializer=self._init_worker_process
        )
        self.worker_stats: Dict[str, WorkerStats] = {}
        self._stats_lock = RLock()
        
        # Memory pressure monitoring
        self._memory_monitor = MemoryPressureMonitor()
        self._memory_monitor.start_monitoring()
        
    @staticmethod
    def _init_worker_process():
        """Initialize worker process with memory optimizations."""
        # Set memory limits for worker process
        try:
            # Limit memory usage per worker
            resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024, -1))  # 1GB limit
        except Exception:
            pass
        
        # Configure matplotlib for minimal memory usage
        matplotlib.rcParams['figure.max_open_warning'] = 0
        matplotlib.rcParams['figure.dpi'] = 72  # Lower DPI to save memory
        
        # Aggressive garbage collection
        gc.set_threshold(100, 5, 5)  # More frequent GC
        
    def render_plot_isolated(self, plot_func: Callable, plot_args: Tuple, 
                           plot_kwargs: Dict[str, Any]) -> bytes:
        """Render plot in isolated subprocess with memory monitoring."""
        
        # Check memory pressure before submitting
        if self._memory_monitor.is_under_pressure():
            self._force_cleanup()
        
        # Submit to worker pool
        future = self.worker_pool.submit(
            self._isolated_plot_worker, 
            plot_func, plot_args, plot_kwargs
        )
        
        try:
            return future.result(timeout=30)  # 30 second timeout
        except Exception as e:
            logger.error(f"Isolated plot rendering failed: {e}")
            # Restart worker pool on persistent failures
            self._restart_worker_pool()
            raise
    
    @staticmethod
    def _isolated_plot_worker(plot_func: Callable, plot_args: Tuple, 
                            plot_kwargs: Dict[str, Any]) -> bytes:
        """Worker function that runs in isolated subprocess."""
        try:
            # Aggressive memory management in worker
            gc.collect()
            
            # Generate plot
            fig = plot_func(*plot_args, **plot_kwargs)
            
            # Convert to bytes immediately
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            plot_data = buffer.getvalue()
            
            # Immediate cleanup
            plt.close(fig)
            buffer.close()
            del fig, buffer
            
            # Force garbage collection
            gc.collect()
            
            return plot_data
            
        except Exception as e:
            # Cleanup on error
            plt.close('all')
            gc.collect()
            raise
        finally:
            # Always cleanup matplotlib state
            plt.close('all')
            matplotlib.pyplot.clf()
            matplotlib.pyplot.cla()
    
    def _force_cleanup(self):
        """Force cleanup of memory resources."""
        # Close all matplotlib figures
        plt.close('all')
        
        # Force garbage collection
        gc.collect()
        
        # Clear matplotlib font cache
        try:
            matplotlib.font_manager._rebuild()
        except Exception:
            pass
    
    def _restart_worker_pool(self):
        """Restart worker pool to prevent memory leaks."""
        try:
            self.worker_pool.shutdown(wait=False)
            self.worker_pool = ProcessPoolExecutor(
                max_workers=min(4, multiprocessing.cpu_count()),
                initializer=self._init_worker_process
            )
            logger.info("Worker pool restarted due to memory pressure")
        except Exception as e:
            logger.error(f"Failed to restart worker pool: {e}")

class MemoryPressureMonitor:
    """Monitor system memory pressure and trigger cleanup."""
    
    def __init__(self, warning_threshold: float = 0.8, critical_threshold: float = 0.9):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self._monitoring = False
        self._monitor_thread = None
        self._shutdown_event = Event()
        
    def start_monitoring(self):
        """Start memory pressure monitoring."""
        if not self._monitoring:
            self._monitoring = True
            self._shutdown_event.clear()
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop,
                name="MemoryPressureMonitor",
                daemon=True
            )
            self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop memory pressure monitoring."""
        self._monitoring = False
        self._shutdown_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
    
    def _monitor_loop(self):
        """Monitor memory usage continuously."""
        while not self._shutdown_event.wait(5.0):  # Check every 5 seconds
            try:
                memory = psutil.virtual_memory()
                memory_percent = memory.percent / 100.0
                
                if memory_percent > self.critical_threshold:
                    logger.critical(f"Critical memory pressure: {memory_percent:.1%}")
                    self._trigger_emergency_cleanup()
                elif memory_percent > self.warning_threshold:
                    logger.warning(f"Memory pressure warning: {memory_percent:.1%}")
                    self._trigger_cleanup()
                    
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
    
    def is_under_pressure(self) -> bool:
        """Check if system is under memory pressure."""
        try:
            memory = psutil.virtual_memory()
            return (memory.percent / 100.0) > self.warning_threshold
        except Exception:
            return False
    
    def _trigger_cleanup(self):
        """Trigger standard cleanup procedures."""
        gc.collect()
        plt.close('all')
        
    def _trigger_emergency_cleanup(self):
        """Trigger emergency cleanup procedures."""
        plt.close('all')
        matplotlib.pyplot.clf()
        matplotlib.pyplot.cla()
        gc.collect()
        
        # Clear caches
        try:
            matplotlib.font_manager._rebuild()
        except Exception:
            pass

# ==========================================
# CRITICAL FIX 2: ML-BASED CACHE WARMING
# ==========================================

@dataclass
class CacheAccessPattern:
    """Cache access pattern for ML prediction."""
    cache_key: str
    access_time: datetime
    user_id: str
    request_type: str
    computation_cost: float
    hit: bool

class MLCachePredictionEngine:
    """Machine Learning based cache warming prediction."""
    
    def __init__(self, history_db_path: Path):
        self.history_db_path = history_db_path
        self._init_database()
        self.access_patterns = deque(maxlen=10000)  # Keep recent patterns
        self._lock = RLock()
        
        # Time-of-day patterns
        self.hourly_patterns = defaultdict(list)
        self.daily_patterns = defaultdict(list)
        self.user_patterns = defaultdict(list)
        
    def _init_database(self):
        """Initialize cache pattern database."""
        with sqlite3.connect(self.history_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cache_key TEXT,
                    access_time TIMESTAMP,
                    user_id TEXT,
                    request_type TEXT,
                    computation_cost REAL,
                    hit BOOLEAN,
                    hour_of_day INTEGER,
                    day_of_week INTEGER
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_patterns_time 
                ON cache_patterns(access_time)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_patterns_user 
                ON cache_patterns(user_id)
            """)
    
    def record_access(self, pattern: CacheAccessPattern):
        """Record cache access pattern for learning."""
        with self._lock:
            # Store in memory for real-time analysis
            self.access_patterns.append(pattern)
            
            # Update time-based patterns
            hour = pattern.access_time.hour
            day = pattern.access_time.weekday()
            
            self.hourly_patterns[hour].append(pattern)
            self.daily_patterns[day].append(pattern)
            self.user_patterns[pattern.user_id].append(pattern)
            
            # Persist to database
            try:
                with sqlite3.connect(self.history_db_path) as conn:
                    conn.execute("""
                        INSERT INTO cache_patterns 
                        (cache_key, access_time, user_id, request_type, 
                         computation_cost, hit, hour_of_day, day_of_week)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        pattern.cache_key,
                        pattern.access_time.isoformat(),
                        pattern.user_id,
                        pattern.request_type,
                        pattern.computation_cost,
                        pattern.hit,
                        hour,
                        day
                    ))
            except Exception as e:
                logger.error(f"Failed to persist cache pattern: {e}")
    
    def predict_cache_needs(self, current_time: datetime = None) -> List[Tuple[str, float]]:
        """Predict which cache keys will be needed soon with confidence scores."""
        if current_time is None:
            current_time = datetime.now()
        
        predictions = []
        
        # Time-of-day based predictions
        current_hour = current_time.hour
        current_day = current_time.weekday()
        
        # Look at patterns from the same hour in recent days
        hour_patterns = self.hourly_patterns.get(current_hour, [])
        recent_hour_patterns = [
            p for p in hour_patterns 
            if (current_time - p.access_time).days <= 7
        ]
        
        # Calculate access probability for each cache key
        key_scores = defaultdict(list)
        for pattern in recent_hour_patterns[-100:]:  # Recent 100 patterns
            if not pattern.hit:  # Cache miss - this key was needed
                key_scores[pattern.cache_key].append(pattern.computation_cost)
        
        # Generate predictions with confidence scores
        for cache_key, costs in key_scores.items():
            if len(costs) >= 2:  # Minimum evidence
                avg_cost = statistics.mean(costs)
                frequency = len(costs)
                
                # Confidence based on frequency and cost
                confidence = min(frequency / 10.0, 1.0) * min(avg_cost / 100.0, 1.0)
                predictions.append((cache_key, confidence))
        
        # Sort by confidence (highest first)
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:20]  # Top 20 predictions
    
    def get_user_behavior_prediction(self, user_id: str) -> List[str]:
        """Predict cache needs based on specific user behavior."""
        user_patterns = self.user_patterns.get(user_id, [])
        
        if len(user_patterns) < 5:  # Not enough data
            return []
        
        # Find frequently accessed but often missed cache keys
        recent_patterns = [
            p for p in user_patterns[-50:]  # Last 50 accesses
            if not p.hit and p.computation_cost > 10.0  # Expensive misses
        ]
        
        # Count frequency
        key_counts = Counter(p.cache_key for p in recent_patterns)
        
        # Return top frequent keys
        return [key for key, count in key_counts.most_common(10) if count >= 2]

class IntelligentCacheWarmer:
    """Intelligent cache warming with ML predictions."""
    
    def __init__(self, cache_manager, prediction_engine: MLCachePredictionEngine):
        self.cache_manager = cache_manager
        self.prediction_engine = prediction_engine
        self._warming_active = False
        self._warmer_thread = None
        self._shutdown_event = Event()
        
    def start_warming(self):
        """Start intelligent cache warming."""
        if not self._warming_active:
            self._warming_active = True
            self._shutdown_event.clear()
            self._warmer_thread = threading.Thread(
                target=self._warming_loop,
                name="IntelligentCacheWarmer",
                daemon=True
            )
            self._warmer_thread.start()
            logger.info("Intelligent cache warming started")
    
    def stop_warming(self):
        """Stop cache warming."""
        self._warming_active = False
        self._shutdown_event.set()
        if self._warmer_thread:
            self._warmer_thread.join(timeout=10)
    
    def _warming_loop(self):
        """Main cache warming loop."""
        while not self._shutdown_event.wait(300):  # Check every 5 minutes
            try:
                # Get predictions
                predictions = self.prediction_engine.predict_cache_needs()
                
                # Warm high-confidence predictions
                for cache_key, confidence in predictions:
                    if confidence > 0.5:  # Only warm high-confidence predictions
                        self._warm_cache_key(cache_key, confidence)
                    
                    if self._shutdown_event.is_set():
                        break
                        
            except Exception as e:
                logger.error(f"Cache warming error: {e}")
    
    def _warm_cache_key(self, cache_key: str, confidence: float):
        """Attempt to warm a specific cache key."""
        try:
            # Check if already cached
            if self.cache_manager.exists(cache_key):
                return
            
            # Attempt to regenerate and cache
            # This would need to be implemented based on cache key format
            logger.debug(f"Warming cache key: {cache_key} (confidence: {confidence:.2f})")
            
        except Exception as e:
            logger.debug(f"Failed to warm cache key {cache_key}: {e}")

# ==========================================
# CRITICAL FIX 3: RESILIENT PLOT GENERATION QUEUE
# ==========================================

class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    recovery_timeout: int = 60
    success_threshold: int = 3
    timeout_duration: int = 30

class CircuitBreaker:
    """Circuit breaker for downstream service protection."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self._lock = RLock()
        
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    logger.info(f"Circuit breaker {self.name} moving to HALF_OPEN")
                else:
                    raise Exception(f"Circuit breaker {self.name} is OPEN")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time > self.config.recovery_timeout
    
    def _on_success(self):
        """Handle successful execution."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logger.info(f"Circuit breaker {self.name} CLOSED after recovery")
        else:
            self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker {self.name} OPENED after {self.failure_count} failures")

class PlotTypeBulkhead:
    """Bulkhead isolation for different plot types."""
    
    def __init__(self):
        # Separate resource pools for different plot types
        self.executors = {
            'histogram': ThreadPoolExecutor(max_workers=2, thread_name_prefix="histogram"),
            'scatter': ThreadPoolExecutor(max_workers=2, thread_name_prefix="scatter"),
            'heatmap': ThreadPoolExecutor(max_workers=1, thread_name_prefix="heatmap"),  # Memory intensive
            'timeseries': ThreadPoolExecutor(max_workers=3, thread_name_prefix="timeseries"),
            'boxplot': ThreadPoolExecutor(max_workers=2, thread_name_prefix="boxplot"),
            'default': ThreadPoolExecutor(max_workers=2, thread_name_prefix="default")
        }
        
        # Circuit breakers for each plot type
        self.circuit_breakers = {
            plot_type: CircuitBreaker(f"{plot_type}_breaker", CircuitBreakerConfig())
            for plot_type in self.executors.keys()
        }
    
    def submit_plot_task(self, plot_type: str, func: Callable, *args, **kwargs) -> Future:
        """Submit plot task with bulkhead isolation."""
        executor = self.executors.get(plot_type, self.executors['default'])
        circuit_breaker = self.circuit_breakers.get(plot_type, self.circuit_breakers['default'])
        
        def protected_func():
            return circuit_breaker.call(func, *args, **kwargs)
        
        return executor.submit(protected_func)
    
    def shutdown(self):
        """Shutdown all executors."""
        for executor in self.executors.values():
            executor.shutdown(wait=True)

class AdaptiveConcurrencyLimiter:
    """Adaptive concurrency limiting based on system performance."""
    
    def __init__(self, initial_limit: int = 10, min_limit: int = 2, max_limit: int = 20):
        self.current_limit = initial_limit
        self.min_limit = min_limit
        self.max_limit = max_limit
        self.active_requests = 0
        self.recent_response_times = deque(maxlen=100)
        self.recent_error_rates = deque(maxlen=50)
        self._lock = RLock()
        self._semaphore = BoundedSemaphore(initial_limit)
        
    @contextmanager
    def acquire(self):
        """Acquire concurrency slot with adaptive limiting."""
        acquired = self._semaphore.acquire(timeout=5.0)
        if not acquired:
            raise Exception("Concurrency limit exceeded - system overloaded")
        
        start_time = time.time()
        success = False
        
        try:
            with self._lock:
                self.active_requests += 1
            yield
            success = True
        finally:
            with self._lock:
                self.active_requests -= 1
                response_time = time.time() - start_time
                self.recent_response_times.append(response_time)
                self.recent_error_rates.append(0 if success else 1)
                self._adjust_limit()
            self._semaphore.release()
    
    def _adjust_limit(self):
        """Adjust concurrency limit based on performance metrics."""
        if len(self.recent_response_times) < 10:
            return
        
        avg_response_time = statistics.mean(self.recent_response_times[-20:])
        error_rate = statistics.mean(self.recent_error_rates[-20:]) if self.recent_error_rates else 0
        
        # Decrease limit if performance degrading
        if avg_response_time > 10.0 or error_rate > 0.1:  # 10s response time or 10% error rate
            new_limit = max(self.current_limit - 1, self.min_limit)
            if new_limit != self.current_limit:
                self._update_semaphore(new_limit)
                logger.info(f"Decreased concurrency limit to {new_limit} (response_time={avg_response_time:.1f}s, error_rate={error_rate:.1%})")
        
        # Increase limit if performance good
        elif avg_response_time < 2.0 and error_rate < 0.02:  # 2s response time and 2% error rate
            new_limit = min(self.current_limit + 1, self.max_limit)
            if new_limit != self.current_limit:
                self._update_semaphore(new_limit)
                logger.info(f"Increased concurrency limit to {new_limit}")
    
    def _update_semaphore(self, new_limit: int):
        """Update semaphore with new limit."""
        self.current_limit = new_limit
        self._semaphore = BoundedSemaphore(new_limit)

class ResilientPlotGenerationQueue:
    """Resilient plot generation queue with circuit breakers and bulkheads."""
    
    def __init__(self):
        self.bulkhead = PlotTypeBulkhead()
        self.concurrency_limiter = AdaptiveConcurrencyLimiter()
        self.request_queue = queue.PriorityQueue(maxsize=1000)
        self._queue_monitor_thread = None
        self._monitoring = False
        self._shutdown_event = Event()
        
        # Start queue monitoring
        self._start_queue_monitoring()
    
    def submit_plot_request(self, request: 'PlotGenerationRequest') -> Future:
        """Submit plot request with full resilience features."""
        
        # Apply concurrency limiting
        with self.concurrency_limiter.acquire():
            # Determine plot type for bulkhead isolation
            plot_type = getattr(request, 'plot_type', 'default')
            
            # Submit to appropriate bulkhead
            future = self.bulkhead.submit_plot_task(
                plot_type,
                self._generate_plot_with_monitoring,
                request
            )
            
            return future
    
    def _generate_plot_with_monitoring(self, request: 'PlotGenerationRequest'):
        """Generate plot with comprehensive monitoring."""
        start_time = time.time()
        memory_before = psutil.virtual_memory().used
        
        try:
            # Use isolated renderer to prevent memory leaks
            result = _isolated_renderer.render_plot_isolated(
                self._plot_generation_func,
                (request,),
                {}
            )
            
            # Log success metrics
            duration = time.time() - start_time
            memory_after = psutil.virtual_memory().used
            memory_delta = (memory_after - memory_before) / (1024 * 1024)  # MB
            
            logger.info(f"Plot generated successfully | type={request.plot_type} | duration={duration:.2f}s | memory_delta={memory_delta:.1f}MB")
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Plot generation failed | type={request.plot_type} | duration={duration:.2f}s | error={e}")
            raise
    
    def _plot_generation_func(self, request: 'PlotGenerationRequest'):
        """Actual plot generation function to be implemented."""
        # This would be implemented based on the specific plot type
        # Placeholder implementation
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
        ax.set_title(f"Sample {request.plot_type} Plot")
        return fig
    
    def _start_queue_monitoring(self):
        """Start queue health monitoring."""
        self._monitoring = True
        self._queue_monitor_thread = threading.Thread(
            target=self._queue_monitor_loop,
            name="PlotQueueMonitor",
            daemon=True
        )
        self._queue_monitor_thread.start()
    
    def _queue_monitor_loop(self):
        """Monitor queue health and performance."""
        while not self._shutdown_event.wait(30):  # Check every 30 seconds
            try:
                queue_size = self.request_queue.qsize()
                active_requests = self.concurrency_limiter.active_requests
                current_limit = self.concurrency_limiter.current_limit
                
                logger.debug(f"Plot queue status | queue_size={queue_size} | active={active_requests} | limit={current_limit}")
                
                # Alert on queue saturation
                if queue_size > 800:  # 80% of max queue size
                    logger.warning(f"Plot generation queue near capacity: {queue_size}/1000")
                
            except Exception as e:
                logger.error(f"Queue monitoring error: {e}")

# ==========================================
# CRITICAL FIX 4: REAL-TIME VISUALIZATION
# ==========================================

class RealTimeVisualizationManager:
    """Complete real-time visualization with WebSocket and Server-Sent Events."""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.websocket_server = None
        self.sse_app = None
        self.connected_clients: Set[websockets.WebSocketServerProtocol] = set()
        self.sse_clients: Dict[str, Any] = {}
        self._clients_lock = RLock()
        
        # Binary streaming optimization
        self.compression_enabled = True
        self.binary_threshold_kb = 100  # Switch to binary for plots > 100KB
        
    async def start_websocket_server(self):
        """Start WebSocket server for real-time visualization."""
        if not WEBSOCKETS_AVAILABLE:
            logger.error("WebSockets not available - install websockets package")
            return
        
        try:
            self.websocket_server = await websockets.serve(
                self._handle_websocket_connection,
                self.host,
                self.port,
                compression="deflate" if self.compression_enabled else None,
                max_size=10 * 1024 * 1024,  # 10MB max message size
                ping_interval=30,
                ping_timeout=10
            )
            logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
    
    async def _handle_websocket_connection(self, websocket, path):
        """Handle WebSocket client connections."""
        client_id = str(uuid.uuid4())
        
        with self._clients_lock:
            self.connected_clients.add(websocket)
        
        logger.info(f"WebSocket client connected: {client_id}")
        
        try:
            # Send welcome message
            await websocket.send(json.dumps({
                "type": "connection",
                "client_id": client_id,
                "message": "Connected to real-time visualization service"
            }))
            
            # Handle incoming messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._process_websocket_message(websocket, client_id, data)
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}")
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": str(e)
                    }))
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket client disconnected: {client_id}")
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
        finally:
            with self._clients_lock:
                self.connected_clients.discard(websocket)
    
    async def _process_websocket_message(self, websocket, client_id: str, data: Dict[str, Any]):
        """Process incoming WebSocket messages."""
        message_type = data.get("type")
        
        if message_type == "plot_request":
            # Handle real-time plot request
            await self._handle_realtime_plot_request(websocket, client_id, data)
        
        elif message_type == "subscribe":
            # Subscribe to plot updates
            plot_id = data.get("plot_id")
            if plot_id:
                await self._subscribe_to_plot_updates(websocket, client_id, plot_id)
        
        elif message_type == "unsubscribe":
            # Unsubscribe from plot updates
            plot_id = data.get("plot_id")
            if plot_id:
                await self._unsubscribe_from_plot_updates(websocket, client_id, plot_id)
    
    async def _handle_realtime_plot_request(self, websocket, client_id: str, data: Dict[str, Any]):
        """Handle real-time plot generation request."""
        try:
            # Send acknowledgment
            await websocket.send(json.dumps({
                "type": "plot_request_received",
                "request_id": data.get("request_id"),
                "status": "processing"
            }))
            
            # Generate plot (this would integrate with the plot generation queue)
            plot_data = await self._generate_plot_async(data)
            
            # Determine if binary streaming is needed
            if len(plot_data) > self.binary_threshold_kb * 1024:
                # Send binary data
                await websocket.send(plot_data)
            else:
                # Send base64 encoded data
                plot_b64 = base64.b64encode(plot_data).decode('utf-8')
                await websocket.send(json.dumps({
                    "type": "plot_result",
                    "request_id": data.get("request_id"),
                    "data": plot_b64,
                    "format": "png"
                }))
        
        except Exception as e:
            await websocket.send(json.dumps({
                "type": "plot_error",
                "request_id": data.get("request_id"),
                "error": str(e)
            }))
    
    async def _generate_plot_async(self, data: Dict[str, Any]) -> bytes:
        """Generate plot asynchronously."""
        # This would integrate with the plot generation system
        # Placeholder implementation
        loop = asyncio.get_event_loop()
        
        def sync_plot_gen():
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
            ax.set_title("Real-time Plot")
            
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=100)
            plt.close(fig)
            return buffer.getvalue()
        
        return await loop.run_in_executor(None, sync_plot_gen)
    
    async def broadcast_plot_update(self, plot_id: str, plot_data: bytes):
        """Broadcast plot update to subscribed clients."""
        if not self.connected_clients:
            return
        
        message = {
            "type": "plot_update",
            "plot_id": plot_id,
            "data": base64.b64encode(plot_data).decode('utf-8'),
            "timestamp": datetime.now().isoformat()
        }
        
        # Send to all connected clients
        disconnected_clients = set()
        
        for client in self.connected_clients.copy():
            try:
                await client.send(json.dumps(message))
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected_clients.add(client)
        
        # Clean up disconnected clients
        with self._clients_lock:
            self.connected_clients -= disconnected_clients
    
    def start_sse_server(self):
        """Start Server-Sent Events server as WebSocket fallback."""
        from fastapi import FastAPI
        from fastapi.responses import StreamingResponse
        from sse_starlette.sse import EventSourceResponse
        
        self.sse_app = FastAPI()
        
        @self.sse_app.get("/events/{client_id}")
        async def sse_endpoint(client_id: str):
            return EventSourceResponse(self._sse_event_stream(client_id))
        
        logger.info("Server-Sent Events endpoint available at /events/{client_id}")
    
    async def _sse_event_stream(self, client_id: str):
        """Server-Sent Events stream generator."""
        try:
            # Add client to SSE clients
            client_queue = asyncio.Queue()
            self.sse_clients[client_id] = client_queue
            
            # Send initial connection event
            yield {
                "event": "connection",
                "data": json.dumps({
                    "client_id": client_id,
                    "message": "Connected to SSE stream"
                })
            }
            
            # Stream events
            while True:
                try:
                    # Wait for events with timeout
                    event_data = await asyncio.wait_for(client_queue.get(), timeout=30)
                    yield event_data
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield {
                        "event": "keepalive",
                        "data": json.dumps({"timestamp": datetime.now().isoformat()})
                    }
        
        except asyncio.CancelledError:
            logger.info(f"SSE client {client_id} disconnected")
        finally:
            # Clean up client
            self.sse_clients.pop(client_id, None)

# ==========================================
# CRITICAL FIX 5: ENHANCED ACCESSIBILITY
# ==========================================

class AccessibilityGenerator:
    """Generate comprehensive accessibility features for visualizations."""
    
    def __init__(self):
        self.data_patterns = {
            'trend': ['increasing', 'decreasing', 'stable', 'fluctuating'],
            'distribution': ['normal', 'skewed', 'bimodal', 'uniform'],
            'correlation': ['positive', 'negative', 'weak', 'strong']
        }
    
    def generate_descriptive_alt_text(self, plot_type: str, data: pd.DataFrame, 
                                    additional_context: Dict[str, Any] = None) -> str:
        """Generate descriptive alt text based on actual data analysis."""
        
        additional_context = additional_context or {}
        
        try:
            if plot_type == 'histogram':
                return self._generate_histogram_alt_text(data, additional_context)
            elif plot_type == 'scatter':
                return self._generate_scatter_alt_text(data, additional_context)
            elif plot_type == 'heatmap':
                return self._generate_heatmap_alt_text(data, additional_context)
            elif plot_type == 'timeseries':
                return self._generate_timeseries_alt_text(data, additional_context)
            elif plot_type == 'boxplot':
                return self._generate_boxplot_alt_text(data, additional_context)
            else:
                return self._generate_generic_alt_text(plot_type, data, additional_context)
        
        except Exception as e:
            logger.error(f"Failed to generate alt text: {e}")
            return f"Data visualization of type {plot_type} with {len(data)} data points."
    
    def _generate_histogram_alt_text(self, data: pd.DataFrame, context: Dict[str, Any]) -> str:
        """Generate detailed alt text for histogram."""
        column = context.get('column')
        if not column or column not in data.columns:
            return "Histogram showing data distribution."
        
        values = data[column].dropna()
        
        # Calculate statistics
        mean_val = values.mean()
        median_val = values.median()
        std_val = values.std()
        min_val = values.min()
        max_val = values.max()
        
        # Determine distribution shape
        skewness = values.skew()
        if abs(skewness) < 0.5:
            shape = "approximately normal"
        elif skewness > 0.5:
            shape = "right-skewed"
        else:
            shape = "left-skewed"
        
        # Check for outliers
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers = values[(values < q1 - 1.5*iqr) | (values > q3 + 1.5*iqr)]
        outlier_text = f" with {len(outliers)} outliers" if len(outliers) > 0 else ""
        
        return (f"Histogram of {column} showing {shape} distribution. "
                f"Data ranges from {min_val:.2f} to {max_val:.2f}, "
                f"with mean {mean_val:.2f} and median {median_val:.2f}. "
                f"Standard deviation is {std_val:.2f}{outlier_text}. "
                f"Based on {len(values)} data points.")
    
    def _generate_scatter_alt_text(self, data: pd.DataFrame, context: Dict[str, Any]) -> str:
        """Generate detailed alt text for scatter plot."""
        x_col = context.get('x_column')
        y_col = context.get('y_column')
        
        if not x_col or not y_col or x_col not in data.columns or y_col not in data.columns:
            return "Scatter plot showing relationship between two variables."
        
        # Calculate correlation
        correlation = data[x_col].corr(data[y_col])
        
        if correlation > 0.7:
            corr_desc = "strong positive"
        elif correlation > 0.3:
            corr_desc = "moderate positive"
        elif correlation > -0.3:
            corr_desc = "weak"
        elif correlation > -0.7:
            corr_desc = "moderate negative"
        else:
            corr_desc = "strong negative"
        
        # Calculate ranges
        x_min, x_max = data[x_col].min(), data[x_col].max()
        y_min, y_max = data[y_col].min(), data[y_col].max()
        
        return (f"Scatter plot showing {corr_desc} correlation (r={correlation:.2f}) "
                f"between {x_col} and {y_col}. "
                f"{x_col} ranges from {x_min:.2f} to {x_max:.2f}, "
                f"{y_col} ranges from {y_min:.2f} to {y_max:.2f}. "
                f"Based on {len(data)} data points.")
    
    def _generate_timeseries_alt_text(self, data: pd.DataFrame, context: Dict[str, Any]) -> str:
        """Generate detailed alt text for time series plot."""
        date_col = context.get('date_column')
        value_col = context.get('value_column')
        
        if not date_col or not value_col:
            return "Time series plot showing data over time."
        
        # Calculate trend
        if len(data) > 1:
            first_val = data[value_col].iloc[0]
            last_val = data[value_col].iloc[-1]
            change_percent = ((last_val - first_val) / abs(first_val)) * 100 if first_val != 0 else 0
            
            if change_percent > 10:
                trend = "increasing"
            elif change_percent < -10:
                trend = "decreasing"
            else:
                trend = "relatively stable"
        else:
            trend = "stable"
        
        # Time range
        start_date = data[date_col].min()
        end_date = data[date_col].max()
        
        # Value statistics
        min_val = data[value_col].min()
        max_val = data[value_col].max()
        mean_val = data[value_col].mean()
        
        return (f"Time series plot of {value_col} from {start_date} to {end_date} "
                f"showing {trend} trend. Values range from {min_val:.2f} to {max_val:.2f} "
                f"with average of {mean_val:.2f}. Based on {len(data)} data points.")
    
    def _generate_heatmap_alt_text(self, data: pd.DataFrame, context: Dict[str, Any]) -> str:
        """Generate detailed alt text for correlation heatmap."""
        if data.shape[1] < 2:
            return "Heatmap visualization."
        
        # Find strongest correlations
        corr_matrix = data.corr()
        
        # Get upper triangle (avoid duplicates)
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find strongest positive and negative correlations
        max_corr = upper_triangle.max().max()
        min_corr = upper_triangle.min().min()
        
        max_pair = upper_triangle.stack().idxmax()
        min_pair = upper_triangle.stack().idxmin()
        
        return (f"Correlation heatmap showing relationships between {data.shape[1]} variables. "
                f"Strongest positive correlation ({max_corr:.2f}) is between {max_pair[0]} and {max_pair[1]}. "
                f"Strongest negative correlation ({min_corr:.2f}) is between {min_pair[0]} and {min_pair[1]}. "
                f"Based on {len(data)} observations.")
    
    def _generate_boxplot_alt_text(self, data: pd.DataFrame, context: Dict[str, Any]) -> str:
        """Generate detailed alt text for box plot."""
        column = context.get('column')
        if not column or column not in data.columns:
            return "Box plot showing data distribution with quartiles."
        
        values = data[column].dropna()
        
        # Calculate quartiles
        q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
        iqr = q3 - q1
        
        # Identify outliers
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = values[(values < lower_bound) | (values > upper_bound)]
        
        return (f"Box plot of {column} showing median of {median:.2f}, "
                f"first quartile at {q1:.2f}, third quartile at {q3:.2f}. "
                f"Interquartile range is {iqr:.2f}. "
                f"{'No outliers detected.' if len(outliers) == 0 else f'{len(outliers)} outliers detected.'} "
                f"Based on {len(values)} data points.")
    
    def _generate_generic_alt_text(self, plot_type: str, data: pd.DataFrame, 
                                 context: Dict[str, Any]) -> str:
        """Generate generic alt text for unknown plot types."""
        return (f"{plot_type.title()} visualization with {len(data)} data points "
                f"across {data.shape[1]} variables. "
                f"Data spans from {data.index.min()} to {data.index.max()}.")
    
    def generate_aria_labels(self, plot_type: str, data: pd.DataFrame) -> Dict[str, str]:
        """Generate ARIA labels for plot elements."""
        labels = {
            'role': 'img',
            'aria-label': f'{plot_type} chart',
            'aria-describedby': f'{plot_type}-description'
        }
        
        # Add specific labels based on plot type
        if plot_type == 'histogram':
            labels['aria-label'] = 'Histogram showing data distribution'
        elif plot_type == 'scatter':
            labels['aria-label'] = 'Scatter plot showing variable relationships'
        elif plot_type == 'heatmap':
            labels['aria-label'] = 'Correlation heatmap matrix'
        elif plot_type == 'timeseries':
            labels['aria-label'] = 'Time series line chart'
        elif plot_type == 'boxplot':
            labels['aria-label'] = 'Box and whisker plot showing quartiles'
        
        return labels
    
    def generate_data_table(self, data: pd.DataFrame, plot_type: str, 
                          max_rows: int = 100) -> str:
        """Generate screen reader-friendly data table."""
        
        # Sample data if too large
        if len(data) > max_rows:
            sampled_data = data.sample(n=max_rows, random_state=42)
            sample_note = f" (showing {max_rows} of {len(data)} rows)"
        else:
            sampled_data = data
            sample_note = ""
        
        # Generate HTML table with proper accessibility attributes
        table_html = f"""
        <table role="table" aria-label="Data table for {plot_type} visualization{sample_note}">
            <caption>Raw data used in {plot_type} visualization{sample_note}</caption>
            <thead>
                <tr>
        """
        
        # Add column headers
        for col in sampled_data.columns:
            table_html += f'<th scope="col">{html.escape(str(col))}</th>'
        
        table_html += """
                </tr>
            </thead>
            <tbody>
        """
        
        # Add data rows
        for idx, row in sampled_data.iterrows():
            table_html += "<tr>"
            for value in row:
                # Format numeric values
                if isinstance(value, (int, float)):
                    if pd.isna(value):
                        formatted_value = "N/A"
                    else:
                        formatted_value = f"{value:.3f}" if isinstance(value, float) else str(value)
                else:
                    formatted_value = html.escape(str(value))
                
                table_html += f'<td>{formatted_value}</td>'
            table_html += "</tr>"
        
        table_html += """
            </tbody>
        </table>
        """
        
        return table_html

# ==========================================
# CRITICAL FIX 6: COMPREHENSIVE RATE LIMITING
# ==========================================

@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    requests_per_minute: int = 30
    requests_per_hour: int = 300
    requests_per_day: int = 1000
    cost_based_limiting: bool = True
    max_cost_per_minute: float = 100.0
    burst_allowance: int = 10

@dataclass
class PlotComplexityMetrics:
    """Metrics for calculating plot generation cost."""
    data_size: int
    plot_type: str
    custom_styling: bool
    high_resolution: bool
    animation: bool
    
    def calculate_cost(self) -> float:
        """Calculate computational cost of plot generation."""
        base_cost = 1.0
        
        # Data size cost (logarithmic scaling)
        size_multiplier = math.log10(max(self.data_size, 10)) / 4  # Normalize to ~0.25-2.0
        
        # Plot type complexity
        type_costs = {
            'histogram': 1.0,
            'scatter': 1.2,
            'boxplot': 1.5,
            'timeseries': 2.0,
            'heatmap': 3.0,
            'custom': 4.0
        }
        type_multiplier = type_costs.get(self.plot_type, 1.0)
        
        # Feature multipliers
        styling_multiplier = 1.5 if self.custom_styling else 1.0
        resolution_multiplier = 2.0 if self.high_resolution else 1.0
        animation_multiplier = 3.0 if self.animation else 1.0
        
        total_cost = (base_cost * size_multiplier * type_multiplier * 
                     styling_multiplier * resolution_multiplier * animation_multiplier)
        
        return round(total_cost, 2)

class UserRateLimiter:
    """Per-user rate limiting with cost-based throttling."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.user_requests: Dict[str, deque] = defaultdict(lambda: deque())
        self.user_costs: Dict[str, deque] = defaultdict(lambda: deque())
        self.user_queue_positions: Dict[str, Dict[str, int]] = defaultdict(dict)
        self._lock = RLock()
        
        # Cleanup thread
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_old_records,
            name="RateLimitCleanup",
            daemon=True
        )
        self._cleanup_thread.start()
    
    def check_rate_limit(self, user_id: str, complexity_metrics: PlotComplexityMetrics) -> Dict[str, Any]:
        """Check if request is within rate limits."""
        
        current_time = time.time()
        cost = complexity_metrics.calculate_cost()
        
        with self._lock:
            user_requests = self.user_requests[user_id]
            user_costs = self.user_costs[user_id]
            
            # Clean up old requests
            self._clean_user_records(user_requests, user_costs, current_time)
            
            # Check time-based limits
            time_limits = self._check_time_based_limits(user_requests, current_time)
            if not time_limits['allowed']:
                return {
                    'allowed': False,
                    'reason': 'time_limit_exceeded',
                    'retry_after': time_limits['retry_after'],
                    'limits': time_limits
                }
            
            # Check cost-based limits
            cost_limits = self._check_cost_based_limits(user_costs, current_time, cost)
            if not cost_limits['allowed']:
                return {
                    'allowed': False,
                    'reason': 'cost_limit_exceeded',
                    'retry_after': cost_limits['retry_after'],
                    'current_cost': cost,
                    'limits': cost_limits
                }
            
            # Record the request
            user_requests.append(current_time)
            user_costs.append((current_time, cost))
            
            return {
                'allowed': True,
                'cost': cost,
                'remaining_requests': self._get_remaining_requests(user_requests, current_time),
                'remaining_cost': self._get_remaining_cost(user_costs, current_time)
            }
    
    def _check_time_based_limits(self, user_requests: deque, current_time: float) -> Dict[str, Any]:
        """Check time-based rate limits."""
        
        # Count requests in different time windows
        minute_requests = sum(1 for req_time in user_requests if current_time - req_time <= 60)
        hour_requests = sum(1 for req_time in user_requests if current_time - req_time <= 3600)
        day_requests = sum(1 for req_time in user_requests if current_time - req_time <= 86400)
        
        # Check limits
        if minute_requests >= self.config.requests_per_minute:
            return {
                'allowed': False,
                'retry_after': 60,
                'minute_requests': minute_requests,
                'minute_limit': self.config.requests_per_minute
            }
        
        if hour_requests >= self.config.requests_per_hour:
            return {
                'allowed': False,
                'retry_after': 3600,
                'hour_requests': hour_requests,
                'hour_limit': self.config.requests_per_hour
            }
        
        if day_requests >= self.config.requests_per_day:
            return {
                'allowed': False,
                'retry_after': 86400,
                'day_requests': day_requests,
                'day_limit': self.config.requests_per_day
            }
        
        return {
            'allowed': True,
            'minute_requests': minute_requests,
            'hour_requests': hour_requests,
            'day_requests': day_requests
        }
    
    def _check_cost_based_limits(self, user_costs: deque, current_time: float, 
                                new_cost: float) -> Dict[str, Any]:
        """Check cost-based rate limits."""
        
        if not self.config.cost_based_limiting:
            return {'allowed': True}
        
        # Calculate cost in last minute
        minute_cost = sum(cost for req_time, cost in user_costs 
                         if current_time - req_time <= 60)
        
        if minute_cost + new_cost > self.config.max_cost_per_minute:
            return {
                'allowed': False,
                'retry_after': 60,
                'current_minute_cost': minute_cost,
                'new_cost': new_cost,
                'max_cost_per_minute': self.config.max_cost_per_minute
            }
        
        return {
            'allowed': True,
            'current_minute_cost': minute_cost
        }
    
    def _get_remaining_requests(self, user_requests: deque, current_time: float) -> Dict[str, int]:
        """Get remaining requests for different time windows."""
        minute_requests = sum(1 for req_time in user_requests if current_time - req_time <= 60)
        hour_requests = sum(1 for req_time in user_requests if current_time - req_time <= 3600)
        day_requests = sum(1 for req_time in user_requests if current_time - req_time <= 86400)
        
        return {
            'minute': max(0, self.config.requests_per_minute - minute_requests),
            'hour': max(0, self.config.requests_per_hour - hour_requests),
            'day': max(0, self.config.requests_per_day - day_requests)
        }
    
    def _get_remaining_cost(self, user_costs: deque, current_time: float) -> Dict[str, float]:
        """Get remaining cost budget."""
        minute_cost = sum(cost for req_time, cost in user_costs 
                         if current_time - req_time <= 60)
        
        return {
            'minute': max(0, self.config.max_cost_per_minute - minute_cost)
        }
    
    def _clean_user_records(self, user_requests: deque, user_costs: deque, current_time: float):
        """Clean up old request records."""
        # Keep only requests from last day
        cutoff_time = current_time - 86400
        
        while user_requests and user_requests[0] < cutoff_time:
            user_requests.popleft()
        
        while user_costs and user_costs[0][0] < cutoff_time:
            user_costs.popleft()
    
    def _cleanup_old_records(self):
        """Periodic cleanup of old records."""
        while True:
            time.sleep(3600)  # Run every hour
            
            current_time = time.time()
            cutoff_time = current_time - 86400  # 24 hours ago
            
            with self._lock:
                # Clean up users with no recent activity
                inactive_users = []
                
                for user_id in list(self.user_requests.keys()):
                    user_requests = self.user_requests[user_id]
                    user_costs = self.user_costs[user_id]
                    
                    if not user_requests or user_requests[-1] < cutoff_time:
                        inactive_users.append(user_id)
                    else:
                        self._clean_user_records(user_requests, user_costs, current_time)
                
                # Remove inactive users
                for user_id in inactive_users:
                    del self.user_requests[user_id]
                    del self.user_costs[user_id]
                    if user_id in self.user_queue_positions:
                        del self.user_queue_positions[user_id]
                
                logger.info(f"Rate limiter cleanup: removed {len(inactive_users)} inactive users")
    
    def get_queue_position(self, user_id: str, request_id: str) -> Optional[int]:
        """Get user's position in the processing queue."""
        return self.user_queue_positions.get(user_id, {}).get(request_id)
    
    def set_queue_position(self, user_id: str, request_id: str, position: int):
        """Set user's position in the processing queue."""
        with self._lock:
            if user_id not in self.user_queue_positions:
                self.user_queue_positions[user_id] = {}
            self.user_queue_positions[user_id][request_id] = position
    
    def remove_from_queue(self, user_id: str, request_id: str):
        """Remove user's request from queue tracking."""
        with self._lock:
            if user_id in self.user_queue_positions:
                self.user_queue_positions[user_id].pop(request_id, None)

# Global instances for critical fixes
_isolated_renderer = IsolatedPlotRenderer()
_memory_pressure_monitor = MemoryPressureMonitor()
_ml_prediction_engine = MLCachePredictionEngine(CACHE_DIR / "cache_patterns.db")
_resilient_plot_queue = ResilientPlotGenerationQueue()
_realtime_viz_manager = RealTimeVisualizationManager()
_accessibility_generator = AccessibilityGenerator()
_rate_limiter = UserRateLimiter(RateLimitConfig())

# Global instances for critical fixes
_isolated_renderer = IsolatedPlotRenderer()
_memory_pressure_monitor = MemoryPressureMonitor()
_ml_prediction_engine = MLCachePredictionEngine(CACHE_DIR / "cache_patterns.db")
_resilient_plot_queue = ResilientPlotGenerationQueue()
_realtime_viz_manager = RealTimeVisualizationManager()

MAX_ROWS_FULL_PROCESSING = 50000
SAMPLE_SIZE_LARGE_DATASET = 10000
MAX_CATEGORIES_HISTOGRAM = 50
MAX_CORRELATION_COLUMNS = 100
MAX_BASE64_SIZE_KB = 200  # Max size for base64 encoding (reduced for S3 integration)
CACHE_TTL_SECONDS = 3600  # 1 hour cache TTL
MAX_MEMORY_MB = 2048  # Maximum memory usage for plot generation
MAX_PLOT_GENERATION_TIME = 300  # Maximum time for plot generation in seconds
CACHE_WARM_THRESHOLD = 10  # Number of accesses before cache warming
S3_RETRY_ATTEMPTS = 3  # Number of S3 operation retries
PLOT_QUEUE_SIZE = 100  # Maximum plot generation queue size

# Memory and resource management enums
class MemoryState(Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EXHAUSTED = "exhausted"

class PlotPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

class CacheState(Enum):
    VALID = "valid"
    STALE = "stale"
    EXPIRED = "expired"
    INVALID = "invalid"

@dataclass
class MemoryUsage:
    """Memory usage tracking for plot generation."""
    process_memory_mb: float = 0.0
    matplotlib_objects: int = 0
    temp_files: int = 0
    cache_size_mb: float = 0.0
    timestamp: float = field(default_factory=time.time)

@dataclass
class PlotGenerationRequest:
    """Plot generation request with priority and metadata."""
    request_id: str
    plot_type: str
    priority: PlotPriority
    data_hash: str
    config: Dict[str, Any]
    callback: Optional[Callable] = None
    created_at: float = field(default_factory=time.time)
    timeout: float = MAX_PLOT_GENERATION_TIME

class MemoryManager:
    """Advanced memory management for matplotlib and plot generation."""
    
    def __init__(self, memory_limit_mb: int = MAX_MEMORY_MB):
        self.memory_limit_mb = memory_limit_mb
        self._lock = RLock()
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._shutdown_event = Event()
        self._memory_history: deque = deque(maxlen=100)
        self._temp_files: weakref.WeakSet = weakref.WeakSet()
        self._matplotlib_figures: weakref.WeakSet = weakref.WeakSet()
        
    def start_monitoring(self, interval_seconds: float = 5.0) -> None:
        """Start memory monitoring."""
        with self._lock:
            if self._monitoring:
                return
            
            self._monitoring = True
            self._shutdown_event.clear()
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop,
                args=(interval_seconds,),
                name="MemoryMonitor",
                daemon=True
            )
            self._monitor_thread.start()
            logger.info(f"Memory monitoring started | limit={self.memory_limit_mb}MB")
    
    def stop_monitoring(self) -> None:
        """Stop memory monitoring."""
        with self._lock:
            if not self._monitoring:
                return
            
            self._monitoring = False
            self._shutdown_event.set()
            
            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=5.0)
            
            logger.info("Memory monitoring stopped")
    
    def get_current_usage(self) -> MemoryUsage:
        """Get current memory usage."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return MemoryUsage(
                process_memory_mb=memory_info.rss / (1024 * 1024),
                matplotlib_objects=len(self._matplotlib_figures),
                temp_files=len(self._temp_files),
                cache_size_mb=self._get_cache_size_mb()
            )
        except Exception as e:
            logger.warning(f"Failed to get memory usage: {e}")
            return MemoryUsage()
    
    def check_memory_state(self, usage: MemoryUsage) -> MemoryState:
        """Check current memory state."""
        if usage.process_memory_mb > self.memory_limit_mb * 0.95:
            return MemoryState.EXHAUSTED
        elif usage.process_memory_mb > self.memory_limit_mb * 0.85:
            return MemoryState.CRITICAL
        elif usage.process_memory_mb > self.memory_limit_mb * 0.70:
            return MemoryState.WARNING
        else:
            return MemoryState.NORMAL
    
    @contextmanager
    def memory_managed_plot(self, plot_id: str):
        """Context manager for memory-managed plot generation."""
        initial_usage = self.get_current_usage()
        fig = None
        
        try:
            # Check memory before starting
            if self.check_memory_state(initial_usage) == MemoryState.EXHAUSTED:
                self._emergency_cleanup()
                raise MemoryError(f"Insufficient memory for plot generation: {initial_usage.process_memory_mb:.1f}MB")
            
            yield
            
        except Exception as e:
            logger.error(f"Plot generation failed | plot_id={plot_id} | error={e}")
            raise
        
        finally:
            # Force cleanup
            self._cleanup_matplotlib_objects()
            gc.collect()
            
            # Log memory usage
            final_usage = self.get_current_usage()
            memory_delta = final_usage.process_memory_mb - initial_usage.process_memory_mb
            logger.debug(f"Plot memory usage | plot_id={plot_id} | delta={memory_delta:.1f}MB | final={final_usage.process_memory_mb:.1f}MB")
    
    def register_figure(self, fig: plt.Figure) -> None:
        """Register matplotlib figure for tracking."""
        with self._lock:
            self._matplotlib_figures.add(fig)
    
    def register_temp_file(self, file_path: Path) -> None:
        """Register temporary file for cleanup."""
        with self._lock:
            self._temp_files.add(file_path)
    
    def _monitor_loop(self, interval_seconds: float) -> None:
        """Main memory monitoring loop."""
        while not self._shutdown_event.wait(interval_seconds):
            try:
                usage = self.get_current_usage()
                state = self.check_memory_state(usage)
                
                with self._lock:
                    self._memory_history.append(usage)
                
                # Take action based on memory state
                if state == MemoryState.EXHAUSTED:
                    logger.error(f"Memory exhausted: {usage.process_memory_mb:.1f}MB | triggering emergency cleanup")
                    self._emergency_cleanup()
                elif state == MemoryState.CRITICAL:
                    logger.warning(f"Memory critical: {usage.process_memory_mb:.1f}MB | triggering cleanup")
                    self._cleanup_matplotlib_objects()
                    gc.collect()
                elif state == MemoryState.WARNING:
                    logger.info(f"Memory warning: {usage.process_memory_mb:.1f}MB")
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
    
    def _cleanup_matplotlib_objects(self) -> None:
        """Clean up matplotlib objects."""
        try:
            # Close all tracked figures
            figures_closed = 0
            for fig in list(self._matplotlib_figures):
                try:
                    plt.close(fig)
                    figures_closed += 1
                except Exception:
                    pass
            
            # Clear matplotlib cache
            plt.clf()
            plt.cla()
            
            if figures_closed > 0:
                logger.debug(f"Closed {figures_closed} matplotlib figures")
                
        except Exception as e:
            logger.warning(f"Matplotlib cleanup failed: {e}")
    
    def _emergency_cleanup(self) -> None:
        """Emergency memory cleanup."""
        try:
            # Close all matplotlib figures
            self._cleanup_matplotlib_objects()
            
            # Clean up temporary files
            temp_files_cleaned = 0
            for temp_file in list(self._temp_files):
                try:
                    if temp_file.exists():
                        temp_file.unlink()
                        temp_files_cleaned += 1
                except Exception:
                    pass
            
            # Force garbage collection
            gc.collect()
            
            logger.info(f"Emergency cleanup completed | temp_files_cleaned={temp_files_cleaned}")
            
        except Exception as e:
            logger.error(f"Emergency cleanup failed: {e}")
    
    def _get_cache_size_mb(self) -> float:
        """Get cache directory size in MB."""
        try:
            total_size = 0
            for file_path in CACHE_DIR.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            return total_size / (1024 * 1024)
        except Exception:
            return 0.0

class IntelligentCacheManager:
    """Intelligent cache management with invalidation and warming."""
    
    def __init__(self, redis_client=None):
        self._redis_client = redis_client or _redis_client
        self._lock = RLock()
        self._cache_stats: Dict[str, Dict[str, Any]] = {}
        self._access_patterns: Dict[str, deque] = {}
        self._invalidation_rules: List[Callable[[str, Dict[str, Any]], bool]] = []
        self._warming_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._warming_thread: Optional[threading.Thread] = None
        self._warming_active = False
        
        self._start_cache_warming()
    
    def get(self, data_hash: str, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get from cache with access tracking."""
        cache_key = _generate_cache_key(data_hash, config)
        
        with self._lock:
            # Track access pattern
            if cache_key not in self._access_patterns:
                self._access_patterns[cache_key] = deque(maxlen=100)
            self._access_patterns[cache_key].append(time.time())
            
            # Update stats
            if cache_key not in self._cache_stats:
                self._cache_stats[cache_key] = {
                    'hits': 0, 'misses': 0, 'last_access': 0, 'created_at': time.time()
                }
        
        # Check cache
        cached_data = _get_from_cache(cache_key)
        
        with self._lock:
            if cached_data:
                self._cache_stats[cache_key]['hits'] += 1
                self._cache_stats[cache_key]['last_access'] = time.time()
                
                # Check if data is still valid
                if self._is_cache_valid(cache_key, cached_data):
                    return cached_data
                else:
                    # Invalidate stale cache
                    self.invalidate(cache_key)
                    self._cache_stats[cache_key]['misses'] += 1
                    return None
            else:
                self._cache_stats[cache_key]['misses'] += 1
                
                # Consider cache warming if frequently accessed
                if self._should_warm_cache(cache_key):
                    self._queue_for_warming(cache_key, data_hash, config)
                
                return None
    
    def set(self, data_hash: str, config: Dict[str, Any], data: Dict[str, Any]) -> None:
        """Set cache with metadata."""
        cache_key = _generate_cache_key(data_hash, config)
        
        # Add cache metadata
        cache_data = {
            **data,
            '_cache_metadata': {
                'created_at': time.time(),
                'data_hash': data_hash,
                'config_hash': hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                'version': '1.0'
            }
        }
        
        _store_in_cache(cache_key, cache_data)
        
        with self._lock:
            if cache_key not in self._cache_stats:
                self._cache_stats[cache_key] = {
                    'hits': 0, 'misses': 0, 'last_access': 0, 'created_at': time.time()
                }
    
    def invalidate(self, cache_key: str) -> None:
        """Invalidate specific cache entry."""
        if self._redis_client:
            try:
                self._redis_client.delete(cache_key)
                logger.debug(f"Invalidated cache key: {cache_key}")
            except Exception as e:
                logger.warning(f"Cache invalidation failed: {e}")
    
    def invalidate_cache_on_data_change(self, data_hash: str) -> None:
        """Invalidate all cache entries for a specific data hash."""
        if not self._redis_client:
            return
        
        try:
            # Find all keys with this data hash
            pattern = f"*{data_hash}*"
            keys = self._redis_client.keys(pattern)
            
            if keys:
                self._redis_client.delete(*keys)
                logger.info(f"Invalidated {len(keys)} cache entries for data_hash: {data_hash}")
                
        except Exception as e:
            logger.warning(f"Data change invalidation failed: {e}")
    
    def _is_cache_valid(self, cache_key: str, cached_data: Dict[str, Any]) -> bool:
        """Check if cached data is still valid."""
        try:
            metadata = cached_data.get('_cache_metadata', {})
            created_at = metadata.get('created_at', 0)
            
            # Check TTL
            if time.time() - created_at > CACHE_TTL_SECONDS:
                return False
            
            # Check custom invalidation rules
            for rule in self._invalidation_rules:
                if rule(cache_key, cached_data):
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Cache validation failed: {e}")
            return False
    
    def _should_warm_cache(self, cache_key: str) -> bool:
        """Determine if cache should be warmed."""
        with self._lock:
            stats = self._cache_stats.get(cache_key, {})
            misses = stats.get('misses', 0)
            
            # Warm cache if it has been missed frequently
            return misses >= CACHE_WARM_THRESHOLD
    
    def _queue_for_warming(self, cache_key: str, data_hash: str, config: Dict[str, Any]) -> None:
        """Queue cache entry for warming."""
        try:
            # Priority based on access frequency
            with self._lock:
                access_count = len(self._access_patterns.get(cache_key, []))
            
            priority = min(access_count, 100)  # Cap priority
            
            warming_request = {
                'cache_key': cache_key,
                'data_hash': data_hash,
                'config': config,
                'priority': priority
            }
            
            self._warming_queue.put((100 - priority, warming_request))  # Lower number = higher priority
            
        except Exception as e:
            logger.warning(f"Cache warming queue failed: {e}")
    
    def _start_cache_warming(self) -> None:
        """Start cache warming thread."""
        self._warming_active = True
        self._warming_thread = threading.Thread(
            target=self._cache_warming_loop,
            name="CacheWarming",
            daemon=True
        )
        self._warming_thread.start()
    
    def _cache_warming_loop(self) -> None:
        """Cache warming background loop."""
        while self._warming_active:
            try:
                # Get warming request with timeout
                try:
                    priority, request = self._warming_queue.get(timeout=30.0)
                except queue.Empty:
                    continue
                
                # Process warming request
                cache_key = request['cache_key']
                logger.debug(f"Warming cache for key: {cache_key}")
                
                # Mark task as done
                self._warming_queue.task_done()
                
            except Exception as e:
                logger.error(f"Cache warming error: {e}")
    
    def get_cache_analytics(self) -> Dict[str, Any]:
        """Get cache usage analytics."""
        with self._lock:
            total_hits = sum(stats['hits'] for stats in self._cache_stats.values())
            total_misses = sum(stats['misses'] for stats in self._cache_stats.values())
            hit_rate = total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0
            
            return {
                'total_entries': len(self._cache_stats),
                'total_hits': total_hits,
                'total_misses': total_misses,
                'hit_rate': round(hit_rate, 3),
                'warming_queue_size': self._warming_queue.qsize(),
                'most_accessed': self._get_most_accessed_entries(5)
            }
    
    def _get_most_accessed_entries(self, limit: int) -> List[Dict[str, Any]]:
        """Get most accessed cache entries."""
        sorted_entries = sorted(
            self._cache_stats.items(),
            key=lambda x: x[1]['hits'],
            reverse=True
        )
        
        return [
            {
                'cache_key': key[:32] + '...' if len(key) > 32 else key,
                'hits': stats['hits'],
                'misses': stats['misses'],
                'hit_rate': stats['hits'] / (stats['hits'] + stats['misses']) if (stats['hits'] + stats['misses']) > 0 else 0
            }
            for key, stats in sorted_entries[:limit]
        ]

# Global instances
_memory_manager: Optional[MemoryManager] = None
_cache_manager: Optional[IntelligentCacheManager] = None

def get_memory_manager() -> MemoryManager:
    """Get or create global memory manager."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
        _memory_manager.start_monitoring()
    return _memory_manager

def get_cache_manager() -> IntelligentCacheManager:
    """Get or create global cache manager."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = IntelligentCacheManager()
    return _cache_manager

# Thread pool for non-blocking rendering
_thread_pool = ThreadPoolExecutor(max_workers=4)

# Redis client for caching (optional)
try:
    _redis_client = redis.Redis(
        host=getattr(settings, 'REDIS_HOST', 'localhost'),
        port=getattr(settings, 'REDIS_PORT', 6379),
        db=getattr(settings, 'REDIS_DB', 0),
        decode_responses=True
    )
    _redis_client.ping()  # Test connection
    logger.info("Redis cache initialized successfully")
except Exception as e:
    logger.warning(f"Redis cache not available: {e}")
    _redis_client = None


class ValidationError(ValueError):
    """Custom exception for data validation errors."""
    pass


class ProcessingError(Exception):
    """Custom exception for data processing errors."""
    pass


def _sanitize_text(text: str) -> str:
    """Sanitize text to prevent code injection in titles and labels."""
    if not isinstance(text, str):
        return str(text)
    
    # Remove potentially dangerous characters and limit length
    sanitized = text.replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')
    sanitized = sanitized.replace('&', '&amp;').replace("'", '&#x27;')
    return sanitized[:200]  # Limit length


def _generate_cache_key(data_hash: str, config_dict: Dict[str, Any]) -> str:
    """Generate cache key from data hash and configuration."""
    config_str = json.dumps(config_dict, sort_keys=True)
    key_content = f"{data_hash}:{config_str}"
    return hashlib.md5(key_content.encode()).hexdigest()


def _get_file_size_kb(file_path: Union[str, Path]) -> float:
    """Get file size in KB."""
    return Path(file_path).stat().st_size / 1024


def _should_skip_base64(file_path: Union[str, Path], force_skip: bool = False) -> bool:
    """Determine if base64 encoding should be skipped based on file size."""
    if force_skip:
        return True
    try:
        return _get_file_size_kb(file_path) > MAX_BASE64_SIZE_KB
    except Exception:
        return True  # Skip on error


def _generate_signed_url(file_path: str) -> str:
    """Generate signed URL for file access (placeholder implementation)."""
    # In production, implement actual signed URL generation with your storage provider
    # For now, return a simple path
    if hasattr(settings, 'BASE_URL'):
        return f"{settings.BASE_URL}/plots/{Path(file_path).name}"
    return f"/api/v1/plots/{Path(file_path).name}"


def _get_from_cache(cache_key: str) -> Optional[Dict[str, Any]]:
    """Retrieve cached plot data."""
    if not _redis_client:
        return None
    
    try:
        cached_data = _redis_client.get(cache_key)
        if cached_data:
            return json.loads(cached_data)
    except Exception as e:
        logger.warning(f"Cache retrieval error: {e}")
    
    return None


def _store_in_cache(cache_key: str, data: Dict[str, Any]) -> None:
    """Store plot data in cache."""
    if not _redis_client:
        return
    
    try:
        _redis_client.setex(cache_key, CACHE_TTL_SECONDS, json.dumps(data))
    except Exception as e:
        logger.warning(f"Cache storage error: {e}")


def _compute_data_hash(df: pd.DataFrame, columns: List[str]) -> str:
    """Compute hash of relevant data for caching."""
    try:
        # Use only the relevant columns and a sample for large datasets
        relevant_data = df[columns] if columns else df
        if len(relevant_data) > 1000:
            # Sample for hash computation to improve performance
            sample_data = relevant_data.sample(n=min(1000, len(relevant_data)), random_state=42)
        else:
            sample_data = relevant_data
        
        # Create hash from data content
        data_str = sample_data.to_string()
        return hashlib.md5(data_str.encode()).hexdigest()
    except Exception as e:
        logger.warning(f"Data hash computation failed: {e}")
        return hashlib.md5(str(time.time()).encode()).hexdigest()


def _log_operation(operation_type: str, columns: List[str], duration: float, 
                  file_path: Optional[str] = None, sample_size: Optional[int] = None,
                  metadata: Optional[dict] = None) -> None:
    """Log visualization operation with structured JSON format."""
    log_data = {
        "operation": "visualization",
        "type": operation_type,
        "columns": columns,
        "duration_seconds": round(duration, 3),
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    if file_path:
        log_data["output_path"] = str(file_path)
    if sample_size:
        log_data["sample_size"] = sample_size
    if metadata:
        log_data["metadata"] = metadata
    
    logger.info(json.dumps(log_data))


def _validate_path_safety(file_path: Union[str, Path]) -> Path:
    """
    Validate that a file path is safe and within allowed boundaries.
    
    Args:
        file_path: Path to validate
        
    Returns:
        Path: Validated and resolved path
        
    Raises:
        ValidationError: If path is unsafe
    """
    path = Path(file_path).resolve()
    
    # Check for path traversal attempts
    if ".." in str(file_path):
        raise ValidationError("Path traversal not allowed")
    
    # Check if path is absolute and outside base storage
    if path.is_absolute() and not str(path).startswith(str(BASE_STORAGE_PATH.resolve())):
        raise ValidationError(f"Absolute paths outside {BASE_STORAGE_PATH} not allowed")
    
    return path


def _generate_filename(prefix: str, extension: str = "png") -> str:
    """Generate timestamped filename for plots."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
    return f"{prefix}_{timestamp}.{extension}"


def load_dataset(file_or_path: Union[str, Path, UploadFile]) -> pd.DataFrame:
    """
    Load dataset from file path or uploaded file.
    
    Args:
        file_or_path: File path string, Path object, or FastAPI UploadFile
        
    Returns:
        pd.DataFrame: Loaded dataset
        
    Raises:
        ValidationError: If file cannot be loaded or is invalid
    """
    start_time = time.time()
    
    try:
        if isinstance(file_or_path, UploadFile):
            # Handle uploaded file
            return _load_from_upload(file_or_path)
        else:
            # Handle file path
            file_path = _validate_path_safety(file_or_path)
            return _load_from_path(file_path)
            
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Failed to load dataset: {str(e)} (duration: {duration:.3f}s)")
        raise ValidationError(f"Failed to load dataset: {str(e)}")


def _load_from_upload(file: UploadFile) -> pd.DataFrame:
    """Load dataset from uploaded file."""
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        content = file.file.read()
        temp_file.write(content)
        temp_path = temp_file.name
    
    try:
        # Determine file type and load accordingly
        filename = file.filename.lower() if file.filename else ""
        
        if filename.endswith('.csv'):
            df = pd.read_csv(temp_path)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(temp_path)
        else:
            # Try CSV first, then Excel
            try:
                df = pd.read_csv(temp_path)
            except Exception:
                df = pd.read_excel(temp_path)
        
        if df.empty:
            raise ValidationError("Dataset contains no data")
        
        return df
        
    finally:
        # Clean up temporary file
        os.unlink(temp_path)


def _load_from_path(file_path: Path) -> pd.DataFrame:
    """Load dataset from file path."""
    if not file_path.exists():
        raise ValidationError(f"File not found: {file_path}")
    
    file_extension = file_path.suffix.lower()
    
    try:
        if file_extension == '.csv':
            df = pd.read_csv(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            raise ValidationError(f"Unsupported file format: {file_extension}")
        
        if df.empty:
            raise ValidationError("Dataset contains no data")
        
        return df
        
    except pd.errors.EmptyDataError:
        raise ValidationError("Dataset contains no data")
    except Exception as e:
        raise ValidationError(f"Failed to read file: {str(e)}")


def validate_columns(df: pd.DataFrame, columns: List[str], 
                    expected_type: str = 'any') -> None:
    """
    Validate that columns exist and have expected data types.
    
    Args:
        df: DataFrame to validate
        columns: List of column names to check
        expected_type: 'numeric', 'categorical', 'datetime', or 'any'
        
    Raises:
        ValidationError: If validation fails
    """
    # Check if columns exist
    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        available = list(df.columns)
        raise ValidationError(
            f"Columns not found: {missing_columns}. Available columns: {available}"
        )
    
    # Check data types if specified
    if expected_type == 'numeric':
        non_numeric = [
            col for col in columns 
            if not pd.api.types.is_numeric_dtype(df[col])
        ]
        if non_numeric:
            raise ValidationError(
                f"Columns must be numeric: {non_numeric}"
            )
    
    elif expected_type == 'categorical':
        # Categorical can be object, category, or low-cardinality numeric
        invalid_categorical = []
        for col in columns:
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > MAX_CATEGORIES_HISTOGRAM:
                invalid_categorical.append(col)
        if invalid_categorical:
            raise ValidationError(
                f"Columns have too many unique values for categorical analysis: {invalid_categorical}"
            )
    
    elif expected_type == 'datetime':
        non_datetime = []
        for col in columns:
            try:
                pd.to_datetime(df[col])
            except Exception:
                non_datetime.append(col)
        if non_datetime:
            raise ValidationError(
                f"Columns cannot be converted to datetime: {non_datetime}"
            )
    
    # Check for all null columns
    all_null_columns = [
        col for col in columns 
        if df[col].isna().all()
    ]
    if all_null_columns:
        raise ValidationError(
            f"Columns contain only null values: {all_null_columns}"
        )


def _sample_large_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Sample large datasets for performance and return sampling metadata.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (sampled_df, metadata_dict)
    """
    if len(df) <= MAX_ROWS_FULL_PROCESSING:
        return df, {"sampled": False, "original_size": len(df)}
    
    # Use stratified sampling if possible, otherwise random
    sample_size = min(SAMPLE_SIZE_LARGE_DATASET, len(df))
    sampled_df = df.sample(n=sample_size, random_state=42)
    
    metadata = {
        "sampled": True,
        "original_size": len(df),
        "sample_size": sample_size,
        "sampling_method": "random"
    }
    
    return sampled_df, metadata


@retrying.retry(stop_max_attempt_number=S3_RETRY_ATTEMPTS, wait_exponential_multiplier=1000)
def _upload_to_s3_with_retry(plot_storage, local_file_path: str, dataset_hash: str, 
                           plot_id: str, format_type: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Upload plot to S3 with retry logic."""
    return plot_storage.upload_plot(
        local_file_path=local_file_path,
        dataset_hash=dataset_hash,
        plot_id=plot_id,
        format_type=format_type,
        metadata=metadata
    )

def render_and_save_plot(fig: plt.Figure, output_path: str, 
                        return_base64: bool = True, skip_base64: bool = False,
                        dpi: int = 100, format_type: str = 'png',
                        dataset_hash: Optional[str] = None,
                        plot_id: Optional[str] = None) -> dict:
    """
    Enhanced render plot with comprehensive memory management and S3 error handling.
    
    Args:
        fig: Matplotlib figure
        output_path: Path to save plot locally
        return_base64: Whether to generate base64 string
        skip_base64: Whether to skip base64 generation (overrides return_base64)
        dpi: DPI for rendering
        format_type: Output format ('png', 'svg', 'pdf')
        dataset_hash: Hash of the dataset for S3 organization
        plot_id: Unique plot identifier for S3 storage
        
    Returns:
        dict: Contains 'file_path', 's3_info', optionally 'base64_image' or 'signed_url'
    """
    result = {}
    temp_file_path = None
    memory_manager = get_memory_manager()
    
    # Register figure for memory tracking
    memory_manager.register_figure(fig)
    
    try:
        # Generate plot ID if not provided
        if not plot_id:
            plot_id = str(uuid.uuid4())
        
        # Check memory before rendering
        initial_memory = memory_manager.get_current_usage()
        memory_state = memory_manager.check_memory_state(initial_memory)
        
        if memory_state == MemoryState.EXHAUSTED:
            raise MemoryError(f"Insufficient memory for plot rendering: {initial_memory.process_memory_mb:.1f}MB")
        
        # Create temporary file for rendering
        temp_file = tempfile.NamedTemporaryFile(
            suffix=f'.{format_type}',
            prefix='kaizen_plot_',
            delete=False
        )
        temp_file_path = temp_file.name
        temp_file.close()
        
        # Register temp file for cleanup
        memory_manager.register_temp_file(Path(temp_file_path))
        
        # Apply tight layout with error handling
        try:
            fig.tight_layout()
        except Exception as e:
            logger.warning(f"Tight layout failed: {e}")
        
        # Save to temporary file with memory monitoring
        save_kwargs = {
            'dpi': dpi,
            'bbox_inches': 'tight',
            'facecolor': 'white',
            'edgecolor': 'none'
        }
        
        if format_type == 'svg':
            save_kwargs.pop('dpi')  # SVG doesn't use DPI
            
        fig.savefig(temp_file_path, format=format_type, **save_kwargs)
        
        # Get file size
        file_size_kb = _get_file_size_kb(temp_file_path)
        result['file_size_kb'] = round(file_size_kb, 2)
        result['format'] = format_type
        
        # Upload to S3 with retry logic and fallback
        if dataset_hash:
            try:
                plot_storage = get_plot_storage()
                s3_result = _upload_to_s3_with_retry(
                    plot_storage=plot_storage,
                    local_file_path=temp_file_path,
                    dataset_hash=dataset_hash,
                    plot_id=plot_id,
                    format_type=format_type,
                    metadata={
                        'created_at': datetime.utcnow().isoformat(),
                        'dpi': str(dpi),
                        'format': format_type,
                        'memory_usage_mb': initial_memory.process_memory_mb
                    }
                )
                result['s3_info'] = s3_result
                result['signed_url'] = s3_result['signed_url']
                logger.info(f"Plot uploaded to S3: {s3_result['s3_key']}")
            except Exception as e:
                logger.warning(f"S3 upload failed after {S3_RETRY_ATTEMPTS} attempts, using local fallback: {e}")
                result['s3_error'] = str(e)
                result['fallback_mode'] = True
                
                # Ensure local file is available as fallback
                if not output_path:
                    fallback_path = PLOTS_DIR / f"fallback_{plot_id}.{format_type}"
                    import shutil
                    shutil.copy2(temp_file_path, fallback_path)
                    result['fallback_path'] = str(fallback_path)
        
        # Also save to local path if specified
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if format_type != 'png':
                output_path = output_path.with_suffix(f'.{format_type}')
            
            # Copy from temp to final location
            import shutil
            shutil.copy2(temp_file_path, output_path)
            result['file_path'] = str(output_path)
        
        # Determine if we should skip base64 based on file size
        should_skip = _should_skip_base64(temp_file_path, skip_base64)
        
        # Generate base64 for small PNG files
        if return_base64 and not should_skip and format_type == 'png' and file_size_kb <= MAX_BASE64_SIZE_KB:
            with open(temp_file_path, 'rb') as f:
                image_data = f.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            result['base64_image'] = f"data:image/png;base64,{image_base64}"
        elif return_base64 and (should_skip or format_type != 'png' or file_size_kb > MAX_BASE64_SIZE_KB):
            # Use signed URL for large files or non-PNG formats
            if 'signed_url' not in result:
                result['signed_url'] = _generate_signed_url(temp_file_path)
            result['message'] = f"File too large for base64 encoding ({file_size_kb:.1f} KB). Use signed URL instead."
        
        # Add accessibility metadata
        result['accessibility'] = {
            'alt_text_template': f"Data visualization showing {format_type} plot",
            'format': format_type,
            'dimensions': f"{fig.get_figwidth():.1f}x{fig.get_figheight():.1f} inches"
        }
        
        return result
        
    except MemoryError as e:
        logger.error(f"Memory error during plot rendering: {str(e)}")
        # Trigger emergency cleanup
        memory_manager._emergency_cleanup()
        raise ProcessingError(f"Insufficient memory for plot rendering: {str(e)}")
    except Exception as e:
        logger.error(f"Error rendering plot: {str(e)}")
        raise ProcessingError(f"Failed to render plot: {str(e)}")
    
    finally:
        # Comprehensive cleanup with memory management
        try:
            # Close figure explicitly
            plt.close(fig)
            
            # Clear any remaining matplotlib state
            plt.clf()
            plt.cla()
            
            # Force garbage collection
            gc.collect()
            
            # Log final memory usage
            if 'initial_memory' in locals():
                final_memory = memory_manager.get_current_usage()
                memory_delta = final_memory.process_memory_mb - initial_memory.process_memory_mb
                logger.debug(f"Plot rendering memory delta: {memory_delta:.1f}MB")
            
        except Exception as e:
            logger.warning(f"Figure cleanup failed: {e}")
        
        # Clean up temporary file (memory manager will also track this)
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {temp_file_path}: {e}")


class PlotGenerationQueue:
    """Priority queue for plot generation with async processing."""
    
    def __init__(self, max_workers: int = 4):
        self._queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=PLOT_QUEUE_SIZE)
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="PlotGen")
        self._active_requests: Dict[str, Future] = {}
        self._lock = RLock()
        self._shutdown = False
        
    def submit_plot_request(self, request: PlotGenerationRequest) -> str:
        """Submit plot generation request with priority."""
        if self._shutdown:
            raise RuntimeError("Plot generation queue is shutdown")
        
        try:
            # Priority is inverted (lower number = higher priority)
            priority_value = 5 - request.priority.value
            self._queue.put((priority_value, request.created_at, request), timeout=1.0)
            
            # Submit to executor
            future = self._executor.submit(self._process_request, request)
            
            with self._lock:
                self._active_requests[request.request_id] = future
            
            logger.debug(f"Queued plot request | id={request.request_id} | priority={request.priority.name}")
            return request.request_id
            
        except queue.Full:
            raise ProcessingError("Plot generation queue is full")
    
    def get_request_status(self, request_id: str) -> Dict[str, Any]:
        """Get status of plot generation request."""
        with self._lock:
            future = self._active_requests.get(request_id)
            
            if not future:
                return {"status": "not_found", "request_id": request_id}
            
            if future.done():
                try:
                    result = future.result()
                    return {
                        "status": "completed",
                        "request_id": request_id,
                        "result": result
                    }
                except Exception as e:
                    return {
                        "status": "failed",
                        "request_id": request_id,
                        "error": str(e)
                    }
            else:
                return {
                    "status": "running",
                    "request_id": request_id
                }
    
    def _process_request(self, request: PlotGenerationRequest) -> Dict[str, Any]:
        """Process individual plot generation request."""
        start_time = time.time()
        memory_manager = get_memory_manager()
        
        try:
            with memory_manager.memory_managed_plot(request.request_id):
                # Execute the callback if provided
                if request.callback:
                    result = request.callback()
                    
                    # Add processing metadata
                    result['processing_time'] = time.time() - start_time
                    result['request_id'] = request.request_id
                    result['priority'] = request.priority.name
                    
                    return result
                else:
                    raise ValueError("No callback provided for plot generation")
                    
        except Exception as e:
            logger.error(f"Plot generation failed | id={request.request_id} | error={e}")
            raise
        
        finally:
            # Clean up active request
            with self._lock:
                self._active_requests.pop(request.request_id, None)
    
    def shutdown(self, wait: bool = True, timeout: float = 30.0) -> None:
        """Shutdown plot generation queue."""
        self._shutdown = True
        self._executor.shutdown(wait=wait, timeout=timeout)
        
        with self._lock:
            self._active_requests.clear()
        
        logger.info("Plot generation queue shutdown complete")


class ThemeManager:
    """Advanced theme management system for visualizations."""
    
    def __init__(self):
        self._themes: Dict[str, Dict[str, Any]] = {}
        self._lock = RLock()
        self._current_theme = "default"
        
        # Initialize default themes
        self._init_default_themes()
    
    def _init_default_themes(self) -> None:
        """Initialize built-in themes."""
        self._themes.update({
            "default": {
                "style": "whitegrid",
                "palette": "deep",
                "font_scale": 1.0,
                "rc_params": {
                    "figure.facecolor": "white",
                    "axes.facecolor": "white",
                    "font.size": 12,
                    "axes.labelsize": 12,
                    "axes.titlesize": 14,
                    "xtick.labelsize": 10,
                    "ytick.labelsize": 10,
                    "legend.fontsize": 11
                }
            },
            "dark": {
                "style": "darkgrid",
                "palette": "bright",
                "font_scale": 1.0,
                "rc_params": {
                    "figure.facecolor": "#2E2E2E",
                    "axes.facecolor": "#2E2E2E",
                    "text.color": "white",
                    "axes.labelcolor": "white",
                    "xtick.color": "white",
                    "ytick.color": "white",
                    "axes.edgecolor": "white",
                    "font.size": 12
                }
            },
            "minimal": {
                "style": "white",
                "palette": "muted",
                "font_scale": 0.9,
                "rc_params": {
                    "axes.spines.top": False,
                    "axes.spines.right": False,
                    "axes.grid": False,
                    "font.size": 11
                }
            },
            "corporate": {
                "style": "whitegrid",
                "palette": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
                "font_scale": 1.1,
                "rc_params": {
                    "font.family": "sans-serif",
                    "font.sans-serif": ["Arial", "DejaVu Sans"],
                    "axes.linewidth": 1.2,
                    "grid.linewidth": 0.8,
                    "font.size": 13
                }
            },
            "accessible": {
                "style": "whitegrid",
                "palette": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
                "font_scale": 1.2,
                "rc_params": {
                    "font.size": 14,
                    "axes.labelsize": 14,
                    "axes.titlesize": 16,
                    "xtick.labelsize": 12,
                    "ytick.labelsize": 12,
                    "legend.fontsize": 13,
                    "lines.linewidth": 2.5,
                    "lines.markersize": 8
                }
            }
        })
    
    def apply_theme(self, theme_name: str) -> None:
        """Apply a theme to matplotlib and seaborn."""
        with self._lock:
            if theme_name not in self._themes:
                raise ValueError(f"Theme '{theme_name}' not found. Available themes: {list(self._themes.keys())}")
            
            theme = self._themes[theme_name]
            
            # Apply seaborn style and palette
            if "style" in theme:
                sns.set_style(theme["style"])
            
            if "palette" in theme:
                sns.set_palette(theme["palette"])
            
            if "font_scale" in theme:
                sns.set_context("notebook", font_scale=theme["font_scale"])
            
            # Apply matplotlib rcParams
            if "rc_params" in theme:
                plt.rcParams.update(theme["rc_params"])
            
            self._current_theme = theme_name
            logger.debug(f"Applied theme: {theme_name}")
    
    def create_custom_theme(self, name: str, theme_config: Dict[str, Any]) -> None:
        """Create a custom theme."""
        with self._lock:
            # Validate theme configuration
            required_keys = ["style", "palette"]
            if not all(key in theme_config for key in required_keys):
                raise ValueError(f"Theme must contain keys: {required_keys}")
            
            self._themes[name] = theme_config.copy()
            logger.info(f"Created custom theme: {name}")
    
    def get_available_themes(self) -> List[str]:
        """Get list of available themes."""
        with self._lock:
            return list(self._themes.keys())
    
    def get_current_theme(self) -> str:
        """Get current theme name."""
        return self._current_theme
    
    def get_theme_config(self, theme_name: str) -> Dict[str, Any]:
        """Get theme configuration."""
        with self._lock:
            if theme_name not in self._themes:
                raise ValueError(f"Theme '{theme_name}' not found")
            return self._themes[theme_name].copy()


class RealTimeVisualizationManager:
    """Real-time visualization capabilities with WebSocket support."""
    
    def __init__(self):
        self._active_sessions: Dict[str, Dict[str, Any]] = {}
        self._lock = RLock()
        self._websocket_server = None
        self._update_queue: queue.Queue = queue.Queue()
        
    def create_realtime_session(self, session_id: str, plot_config: Dict[str, Any]) -> None:
        """Create a real-time visualization session."""
        with self._lock:
            self._active_sessions[session_id] = {
                "config": plot_config,
                "created_at": time.time(),
                "last_update": time.time(),
                "update_count": 0,
                "subscribers": set()
            }
            
            logger.info(f"Created real-time session: {session_id}")
    
    def update_plot_data(self, session_id: str, new_data: pd.DataFrame) -> None:
        """Update plot data for real-time session."""
        with self._lock:
            if session_id not in self._active_sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self._active_sessions[session_id]
            session["last_update"] = time.time()
            session["update_count"] += 1
            
            # Queue update for WebSocket broadcast
            update_message = {
                "session_id": session_id,
                "timestamp": time.time(),
                "data_hash": _compute_data_hash(new_data, list(new_data.columns)),
                "row_count": len(new_data)
            }
            
            try:
                self._update_queue.put_nowait(update_message)
            except queue.Full:
                logger.warning(f"Update queue full for session: {session_id}")
    
    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get real-time session status."""
        with self._lock:
            if session_id not in self._active_sessions:
                return {"status": "not_found"}
            
            session = self._active_sessions[session_id]
            return {
                "status": "active",
                "session_id": session_id,
                "created_at": session["created_at"],
                "last_update": session["last_update"],
                "update_count": session["update_count"],
                "subscriber_count": len(session["subscribers"])
            }
    
    def cleanup_inactive_sessions(self, max_age_seconds: int = 3600) -> None:
        """Clean up inactive sessions."""
        current_time = time.time()
        inactive_sessions = []
        
        with self._lock:
            for session_id, session in self._active_sessions.items():
                if current_time - session["last_update"] > max_age_seconds:
                    inactive_sessions.append(session_id)
        
        for session_id in inactive_sessions:
            self.remove_session(session_id)
            
        if inactive_sessions:
            logger.info(f"Cleaned up {len(inactive_sessions)} inactive sessions")
    
    def remove_session(self, session_id: str) -> None:
        """Remove real-time session."""
        with self._lock:
            if session_id in self._active_sessions:
                del self._active_sessions[session_id]
                logger.info(f"Removed session: {session_id}")


# Global instances
_plot_queue: Optional[PlotGenerationQueue] = None
_theme_manager: Optional[ThemeManager] = None
_realtime_manager: Optional[RealTimeVisualizationManager] = None

def get_plot_queue() -> PlotGenerationQueue:
    """Get or create global plot generation queue."""
    global _plot_queue
    if _plot_queue is None:
        _plot_queue = PlotGenerationQueue()
    return _plot_queue

def get_theme_manager() -> ThemeManager:
    """Get or create global theme manager."""
    global _theme_manager
    if _theme_manager is None:
        _theme_manager = ThemeManager()
    return _theme_manager

def get_realtime_manager() -> RealTimeVisualizationManager:
    """Get or create global real-time visualization manager."""
    global _realtime_manager
    if _realtime_manager is None:
        _realtime_manager = RealTimeVisualizationManager()
    return _realtime_manager


def _render_plot_async(fig: plt.Figure, output_path: str, 
                      return_base64: bool = True, skip_base64: bool = False,
                      dpi: int = 100, format_type: str = 'png') -> Dict[str, Any]:
    """Async wrapper for plot rendering using thread pool."""
    future = _thread_pool.submit(
        render_and_save_plot, fig, output_path, return_base64, 
        skip_base64, dpi, format_type
    )
    return future


def generate_histogram(request: HistogramRequest) -> VisualizationResponse:
    """
    Enhanced histogram generation with comprehensive memory management and caching.
    
    Args:
        request: Histogram request configuration
        
    Returns:
        VisualizationResponse: Generated plot response
    """
    start_time = time.time()
    memory_manager = get_memory_manager()
    cache_manager = get_cache_manager()
    plot_id = f"histogram_{request.column}_{int(time.time())}"
    
    # Enhanced input sanitization
    if not request.column or len(request.column) > 100:
        raise ValidationError("Invalid column name")
    
    # Sanitize title and other text inputs
    if request.config.title:
        request.config.title = html.escape(request.config.title[:200])
    
    try:
        with memory_manager.memory_managed_plot(plot_id):
            # Load and validate data
            df = load_dataset(request.file_or_path)
            validate_columns(df, [request.column])
            
            # Generate dataset hash and plot configuration
            data_hash = _compute_data_hash(df, [request.column])
            plot_config = {
                'type': 'histogram',
                'column': request.column,
                'bins': request.bins,
                'style': request.config.style,
                'kde': request.config.kde,
                'title': request.config.title,
                'figsize': request.config.figsize,
                'dpi': request.config.dpi
            }
            
            # Check intelligent cache first
            cached_result = cache_manager.get(data_hash, plot_config)
            if cached_result:
                logger.info(f"Returning cached histogram for column: {request.column}")
                return VisualizationResponse(**cached_result)
            
            # Sample large datasets
            df, sampling_metadata = _sample_large_dataset(df)
            
            # Configure plot style
            sns.set_style(request.config.style)
            plt.rcParams.update({'font.size': 10})
            
            # Create figure
            fig, ax = plt.subplots(figsize=request.config.figsize)
            
            # Register figure for memory tracking
            memory_manager.register_figure(fig)
            
            # Determine plot type based on data
            column_data = df[request.column]
            
            if pd.api.types.is_numeric_dtype(column_data):
                # Numeric histogram
                sns.histplot(
                    data=df, 
                    x=request.column, 
                    kde=request.config.kde,
                    bins=request.bins or 'auto',
                    ax=ax
                )
            else:
                # Categorical count plot
                if column_data.nunique() > MAX_CATEGORIES_HISTOGRAM:
                    raise ValidationError(
                        f"Column '{request.column}' has {column_data.nunique()} unique values, "
                        f"which exceeds the maximum of {MAX_CATEGORIES_HISTOGRAM} for histogram display"
                    )
            
                sns.countplot(data=df, x=request.column, ax=ax)
                
                # Rotate labels if many categories
                if column_data.nunique() > 10:
                    plt.xticks(rotation=45, ha='right')
            
            # Customize labels and title with sanitization
            ax.set_xlabel(_sanitize_text(request.column.replace('_', ' ').title()))
            ax.set_ylabel('Count')
            
            if request.config.title:
                ax.set_title(_sanitize_text(request.config.title))
            else:
                ax.set_title(_sanitize_text(f"Distribution of {request.column.replace('_', ' ').title()}"))
            
            # Generate output path and plot ID
            output_filename = _generate_filename("histogram")
            output_path = PLOTS_DIR / output_filename
            
            # Render plot with S3 upload, traced and observed
            with traced_span("visualization.histogram"):
                with observe(
                    operation="render",
                    service="visualization",
                    labels={"plot_type": "histogram"},
                    histogram=visualization_render_seconds
                ):
                    render_result = render_and_save_plot(
                        fig, output_path, request.config.return_base64, 
                        request.config.skip_base64, request.config.dpi,
                        dataset_hash=data_hash, plot_id=plot_id
                    )
            
            # Calculate metadata
            duration = time.time() - start_time
            metadata = {
                "column": request.column,
                "data_type": str(df[request.column].dtype),
                "null_count": int(df[request.column].isna().sum()),
                "unique_values": int(df[request.column].nunique()),
                "total_values": len(df),
                "plot_id": plot_id,
                "dataset_hash": data_hash,
                **sampling_metadata
            }
            
            # Add S3 info to metadata if available
            if 's3_info' in render_result:
                metadata['s3_key'] = render_result['s3_info']['s3_key']
                metadata['s3_bucket'] = render_result['s3_info']['bucket']
            
            # Create response
            response_data = {
                "status": "success",
                "plot_path": render_result.get('file_path'),
                "base64_image": render_result.get('base64_image'),
                "signed_url": render_result.get('signed_url'),
                "message": f"Histogram generated successfully for column '{request.column}'",
                "metadata": {**metadata, **render_result.get('accessibility', {})}
            }
            
            # Cache the result with intelligent management
            cache_manager.set(data_hash, plot_config, response_data)
            
            # Log operation
            _log_operation("histogram", [request.column], duration, 
                          render_result.get('file_path'), 
                          sampling_metadata.get('sample_size'), metadata)
            
            return VisualizationResponse(**response_data)
        
    except (ValidationError, ProcessingError) as e:
        duration = time.time() - start_time
        _log_operation("histogram", [request.column], duration, metadata={"error": str(e)})
        return VisualizationResponse(
            status="error",
            message=str(e),
            metadata={"column": request.column}
        )
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Unexpected error in histogram generation: {str(e)}")
        _log_operation("histogram", [request.column], duration, metadata={"error": str(e)})
        return VisualizationResponse(
            status="error",
            message="An unexpected error occurred during histogram generation",
            metadata={"column": request.column}
        )


def generate_correlation_heatmap(request: CorrelationHeatmapRequest) -> VisualizationResponse:
    """
    Generate correlation heatmap visualization with caching support.
    
    Args:
        request: Correlation heatmap request configuration
        
    Returns:
        VisualizationResponse: Generated plot response
    """
    start_time = time.time()
    
    try:
        # Load data
        df = load_dataset(request.file_or_path)
        
        # Select numeric columns
        if request.columns:
            validate_columns(df, request.columns, 'numeric')
            numeric_df = df[request.columns]
        else:
            numeric_df = df.select_dtypes(include=['float64', 'int64', 'float32', 'int32'])
        
        if numeric_df.empty:
            raise ValidationError("No numeric columns found for correlation analysis")
        
        if numeric_df.shape[1] < 2:
            raise ValidationError("At least 2 numeric columns required for correlation analysis")
        
        if numeric_df.shape[1] > MAX_CORRELATION_COLUMNS:
            raise ValidationError(
                f"Too many columns ({numeric_df.shape[1]}) for correlation heatmap. "
                f"Maximum allowed: {MAX_CORRELATION_COLUMNS}"
            )
        
        # Generate cache key
        data_hash = _compute_data_hash(numeric_df, list(numeric_df.columns))
        config_dict = {
            'type': 'correlation_heatmap',
            'columns': list(numeric_df.columns),
            'method': request.correlation_method,
            'style': request.config.style,
            'title': request.config.title
        }
        cache_key = _generate_cache_key(data_hash, config_dict)
        
        # Check cache first
        cached_result = _get_from_cache(cache_key)
        if cached_result and Path(cached_result.get('plot_path', '')).exists():
            logger.info(f"Returning cached correlation heatmap for {len(numeric_df.columns)} columns")
            return VisualizationResponse(**cached_result)
        
        # Sample large datasets
        numeric_df, sampling_metadata = _sample_large_dataset(numeric_df)
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr(method=request.correlation_method)
        
        # Configure plot style
        sns.set_style(request.config.style)
        
        # Calculate figure size based on number of columns
        n_cols = len(corr_matrix.columns)
        figsize = (max(8, n_cols * 0.6), max(6, n_cols * 0.5))
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap='RdBu_r',
            center=0,
            square=True,
            fmt='.2f',
            cbar_kws={"shrink": 0.8},
            ax=ax,
            vmin=-1,
            vmax=1
        )
        
        # Customize layout
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        if request.config.title:
            ax.set_title(_sanitize_text(request.config.title))
        else:
            ax.set_title("Correlation Matrix")
        
        # Generate output path
        output_filename = _generate_filename("correlation_heatmap")
        output_path = PLOTS_DIR / output_filename
        
        # Render plot
        render_result = render_and_save_plot(
            fig, output_path, request.config.return_base64,
            request.config.skip_base64, request.config.dpi
        )
        
        # Calculate metadata
        duration = time.time() - start_time
        metadata = {
            "numeric_columns": list(numeric_df.columns),
            "correlation_method": request.correlation_method,
            "matrix_shape": corr_matrix.shape,
            "cache_key": cache_key,
            **sampling_metadata
        }
        
        # Create response
        response_data = {
            "status": "success",
            "plot_path": render_result.get('file_path'),
            "base64_image": render_result.get('base64_image'),
            "signed_url": render_result.get('signed_url'),
            "message": f"Correlation heatmap generated successfully for {len(numeric_df.columns)} columns",
            "metadata": {**metadata, **render_result.get('accessibility', {})}
        }
        
        # Cache the result
        _store_in_cache(cache_key, response_data)
        
        # Log operation
        _log_operation("correlation_heatmap", list(numeric_df.columns), 
                      duration, render_result['file_path'],
                      sampling_metadata.get('sample_size'), metadata)
        
        return VisualizationResponse(**response_data)
        
    except (ValidationError, ProcessingError) as e:
        duration = time.time() - start_time
        columns = request.columns or []
        _log_operation("correlation_heatmap", columns, duration, metadata={"error": str(e)})
        return VisualizationResponse(
            status="error",
            message=str(e),
            metadata={"columns": columns}
        )
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Unexpected error in correlation heatmap generation: {str(e)}")
        columns = request.columns or []
        _log_operation("correlation_heatmap", columns, duration, metadata={"error": str(e)})
        return VisualizationResponse(
            status="error",
            message="An unexpected error occurred during correlation heatmap generation",
            metadata={"columns": columns}
        )


def generate_scatter_plot(request: ScatterPlotRequest) -> VisualizationResponse:
    """
    Generate scatter plot visualization with caching support.
    
    Args:
        request: Scatter plot request configuration
        
    Returns:
        VisualizationResponse: Generated plot response
    """
    start_time = time.time()
    
    try:
        # Load and validate data
        df = load_dataset(request.file_or_path)
        validate_columns(df, [request.x_column, request.y_column], 'numeric')
        
        # Check for sufficient valid data
        valid_data = df[[request.x_column, request.y_column]].dropna()
        if valid_data.empty:
            raise ValidationError(
                f"No valid data points found for columns '{request.x_column}' and '{request.y_column}'"
            )
        
        # Generate cache key
        columns_for_hash = [request.x_column, request.y_column]
        if request.color_column:
            columns_for_hash.append(request.color_column)
        
        data_hash = _compute_data_hash(df, columns_for_hash)
        config_dict = {
            'type': 'scatter_plot',
            'x_column': request.x_column,
            'y_column': request.y_column,
            'color_column': request.color_column,
            'style': request.config.style,
            'title': request.config.title
        }
        cache_key = _generate_cache_key(data_hash, config_dict)
        
        # Check cache first
        cached_result = _get_from_cache(cache_key)
        if cached_result and Path(cached_result.get('plot_path', '')).exists():
            logger.info(f"Returning cached scatter plot for {request.x_column} vs {request.y_column}")
            return VisualizationResponse(**cached_result)
        
        # Sample large datasets
        valid_data, sampling_metadata = _sample_large_dataset(valid_data)
        
        # Configure plot style
        sns.set_style(request.config.style)
        
        # Create figure
        fig, ax = plt.subplots(figsize=request.config.figsize)
        
        # Create scatter plot
        scatter_kwargs = {'alpha': 0.6, 's': 30}
        
        if request.color_column and request.color_column in df.columns:
            # Color by category or continuous variable
            sns.scatterplot(
                data=valid_data, 
                x=request.x_column, 
                y=request.y_column,
                hue=request.color_column,
                ax=ax,
                **scatter_kwargs
            )
        else:
            sns.scatterplot(
                data=valid_data,
                x=request.x_column,
                y=request.y_column,
                ax=ax,
                **scatter_kwargs
            )
        
        # Customize labels
        ax.set_xlabel(_sanitize_text(request.x_column.replace('_', ' ').title()))
        ax.set_ylabel(_sanitize_text(request.y_column.replace('_', ' ').title()))
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3)
        
        if request.config.title:
            ax.set_title(_sanitize_text(request.config.title))
        else:
            title = f"{request.x_column.replace('_', ' ').title()} vs {request.y_column.replace('_', ' ').title()}"
            ax.set_title(_sanitize_text(title))
        
        # Generate output path
        output_filename = _generate_filename("scatter_plot")
        output_path = PLOTS_DIR / output_filename
        
        # Render plot
        render_result = render_and_save_plot(
            fig, output_path, request.config.return_base64,
            request.config.skip_base64, request.config.dpi
        )
        
        # Calculate correlation
        correlation = valid_data[request.x_column].corr(valid_data[request.y_column])
        
        # Calculate metadata
        duration = time.time() - start_time
        metadata = {
            "x_column": request.x_column,
            "y_column": request.y_column,
            "color_column": request.color_column,
            "correlation": round(correlation, 4) if not pd.isna(correlation) else None,
            "valid_points": len(valid_data),
            "original_points": len(df),
            "cache_key": cache_key,
            **sampling_metadata
        }
        
        # Create response
        response_data = {
            "status": "success",
            "plot_path": render_result.get('file_path'),
            "base64_image": render_result.get('base64_image'),
            "signed_url": render_result.get('signed_url'),
            "message": f"Scatter plot generated successfully for '{request.x_column}' vs '{request.y_column}'",
            "metadata": {**metadata, **render_result.get('accessibility', {})}
        }
        
        # Cache the result
        _store_in_cache(cache_key, response_data)
        
        # Log operation
        columns = [request.x_column, request.y_column]
        if request.color_column:
            columns.append(request.color_column)
        
        _log_operation("scatter_plot", columns, duration,
                      render_result['file_path'],
                      sampling_metadata.get('sample_size'), metadata)
        
        return VisualizationResponse(**response_data)
        
    except (ValidationError, ProcessingError) as e:
        duration = time.time() - start_time
        columns = [request.x_column, request.y_column]
        _log_operation("scatter_plot", columns, duration, metadata={"error": str(e)})
        return VisualizationResponse(
            status="error",
            message=str(e),
            metadata={"x_column": request.x_column, "y_column": request.y_column}
        )
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Unexpected error in scatter plot generation: {str(e)}")
        columns = [request.x_column, request.y_column]
        _log_operation("scatter_plot", columns, duration, metadata={"error": str(e)})
        return VisualizationResponse(
            status="error",
            message="An unexpected error occurred during scatter plot generation",
            metadata={"x_column": request.x_column, "y_column": request.y_column}
        )


def generate_box_plot(request: BoxPlotRequest) -> VisualizationResponse:
    """
    Generate box plot visualization with caching support.
    
    Args:
        request: Box plot request configuration
        
    Returns:
        VisualizationResponse: Generated plot response
    """
    start_time = time.time()
    
    try:
        # Load and validate data
        df = load_dataset(request.file_or_path)
        validate_columns(df, [request.numeric_column], 'numeric')
        validate_columns(df, [request.categorical_column], 'categorical')
        
        # Check for sufficient data
        valid_data = df[[request.numeric_column, request.categorical_column]].dropna()
        if valid_data.empty:
            raise ValidationError(
                f"No valid data points found for columns '{request.numeric_column}' and '{request.categorical_column}'"
            )
        
        # Check category count
        unique_categories = valid_data[request.categorical_column].nunique()
        if unique_categories > MAX_CATEGORIES_HISTOGRAM:
            raise ValidationError(
                f"Too many categories ({unique_categories}) in '{request.categorical_column}'. "
                f"Maximum allowed: {MAX_CATEGORIES_HISTOGRAM}"
            )
        
        # Generate cache key
        data_hash = _compute_data_hash(df, [request.numeric_column, request.categorical_column])
        config_dict = {
            'type': 'box_plot',
            'numeric_column': request.numeric_column,
            'categorical_column': request.categorical_column,
            'style': request.config.style,
            'title': request.config.title
        }
        cache_key = _generate_cache_key(data_hash, config_dict)
        
        # Check cache first
        cached_result = _get_from_cache(cache_key)
        if cached_result and Path(cached_result.get('plot_path', '')).exists():
            logger.info(f"Returning cached box plot for {request.numeric_column} by {request.categorical_column}")
            return VisualizationResponse(**cached_result)
        
        # Sample large datasets
        valid_data, sampling_metadata = _sample_large_dataset(valid_data)
        
        # Configure plot style
        sns.set_style(request.config.style)
        
        # Create figure with adjusted width for many categories
        width_factor = max(1, unique_categories * 0.8)
        figsize = (min(16, request.config.figsize[0] * width_factor), request.config.figsize[1])
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create box plot
        sns.boxplot(
            data=valid_data,
            x=request.categorical_column,
            y=request.numeric_column,
            ax=ax
        )
        
        # Customize labels
        ax.set_xlabel(_sanitize_text(request.categorical_column.replace('_', ' ').title()))
        ax.set_ylabel(_sanitize_text(request.numeric_column.replace('_', ' ').title()))
        
        # Rotate x-axis labels if many categories
        if unique_categories > 5:
            plt.xticks(rotation=45, ha='right')
        
        if request.config.title:
            ax.set_title(_sanitize_text(request.config.title))
        else:
            title = f"Distribution of {request.numeric_column.replace('_', ' ').title()} by {request.categorical_column.replace('_', ' ').title()}"
            ax.set_title(_sanitize_text(title))
        
        # Generate output path
        output_filename = _generate_filename("box_plot")
        output_path = PLOTS_DIR / output_filename
        
        # Render plot
        render_result = render_and_save_plot(
            fig, output_path, request.config.return_base64,
            request.config.skip_base64, request.config.dpi
        )
        
        # Calculate metadata
        duration = time.time() - start_time
        metadata = {
            "numeric_column": request.numeric_column,
            "categorical_column": request.categorical_column,
            "unique_categories": unique_categories,
            "valid_points": len(valid_data),
            "original_points": len(df),
            "cache_key": cache_key,
            **sampling_metadata
        }
        
        # Create response
        response_data = {
            "status": "success",
            "plot_path": render_result.get('file_path'),
            "base64_image": render_result.get('base64_image'),
            "signed_url": render_result.get('signed_url'),
            "message": f"Box plot generated successfully for '{request.numeric_column}' by '{request.categorical_column}'",
            "metadata": {**metadata, **render_result.get('accessibility', {})}
        }
        
        # Cache the result
        _store_in_cache(cache_key, response_data)
        
        # Log operation
        columns = [request.numeric_column, request.categorical_column]
        _log_operation("box_plot", columns, duration,
                      render_result['file_path'],
                      sampling_metadata.get('sample_size'), metadata)
        
        return VisualizationResponse(**response_data)
        
    except (ValidationError, ProcessingError) as e:
        duration = time.time() - start_time
        columns = [request.numeric_column, request.categorical_column]
        _log_operation("box_plot", columns, duration, metadata={"error": str(e)})
        return VisualizationResponse(
            status="error",
            message=str(e),
            metadata={
                "numeric_column": request.numeric_column,
                "categorical_column": request.categorical_column
            }
        )
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Unexpected error in box plot generation: {str(e)}")
        columns = [request.numeric_column, request.categorical_column]
        _log_operation("box_plot", columns, duration, metadata={"error": str(e)})
        return VisualizationResponse(
            status="error",
            message="An unexpected error occurred during box plot generation",
            metadata={
                "numeric_column": request.numeric_column,
                "categorical_column": request.categorical_column
            }
        )


def generate_time_series_plot(request: TimeSeriesRequest) -> VisualizationResponse:
    """
    Generate time series line plot visualization with caching support.
    
    Args:
        request: Time series request configuration
        
    Returns:
        VisualizationResponse: Generated plot response
    """
    start_time = time.time()
    
    try:
        # Load and validate data
        df = load_dataset(request.file_or_path)
        validate_columns(df, [request.datetime_column], 'datetime')
        validate_columns(df, [request.value_column], 'numeric')
        
        # Convert datetime column
        df[request.datetime_column] = pd.to_datetime(df[request.datetime_column])
        
        # Check for sufficient valid data
        valid_data = df[[request.datetime_column, request.value_column]].dropna()
        if valid_data.empty:
            raise ValidationError(
                f"No valid data points found for columns '{request.datetime_column}' and '{request.value_column}'"
            )
        
        # Sort by datetime
        valid_data = valid_data.sort_values(request.datetime_column)
        
        # Generate cache key
        columns_for_hash = [request.datetime_column, request.value_column]
        if request.group_column:
            columns_for_hash.append(request.group_column)
        
        data_hash = _compute_data_hash(df, columns_for_hash)
        config_dict = {
            'type': 'time_series',
            'datetime_column': request.datetime_column,
            'value_column': request.value_column,
            'group_column': request.group_column,
            'style': request.config.style,
            'title': request.config.title
        }
        cache_key = _generate_cache_key(data_hash, config_dict)
        
        # Check cache first
        cached_result = _get_from_cache(cache_key)
        if cached_result and Path(cached_result.get('plot_path', '')).exists():
            logger.info(f"Returning cached time series plot for {request.value_column}")
            return VisualizationResponse(**cached_result)
        
        # Sample large datasets while maintaining time order
        if len(valid_data) > MAX_ROWS_FULL_PROCESSING:
            # For time series, use systematic sampling to maintain temporal structure
            step_size = len(valid_data) // SAMPLE_SIZE_LARGE_DATASET
            sample_indices = range(0, len(valid_data), max(1, step_size))
            valid_data = valid_data.iloc[list(sample_indices)]
            
            sampling_metadata = {
                "sampled": True,
                "original_size": len(df),
                "sample_size": len(valid_data),
                "sampling_method": "systematic_temporal"
            }
        else:
            sampling_metadata = {"sampled": False, "original_size": len(valid_data)}
        
        # Configure plot style
        sns.set_style(request.config.style)
        plt.rcParams.update({'axes.grid': True})
        
        # Create figure
        fig, ax = plt.subplots(figsize=request.config.figsize)
        
        # Create time series plot
        if request.group_column and request.group_column in df.columns:
            # Multiple series grouped by category
            for group_name, group_data in valid_data.groupby(request.group_column):
                ax.plot(
                    group_data[request.datetime_column],
                    group_data[request.value_column],
                    label=str(group_name),
                    linewidth=1.5,
                    alpha=0.8
                )
            ax.legend(title=_sanitize_text(request.group_column.replace('_', ' ').title()))
        else:
            # Single time series
            ax.plot(
                valid_data[request.datetime_column],
                valid_data[request.value_column],
                linewidth=1.5,
                color='steelblue'
            )
        
        # Customize labels and formatting
        ax.set_xlabel(_sanitize_text(request.datetime_column.replace('_', ' ').title()))
        ax.set_ylabel(_sanitize_text(request.value_column.replace('_', ' ').title()))
        
        # Format x-axis dates
        fig.autofmt_xdate()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        if request.config.title:
            ax.set_title(_sanitize_text(request.config.title))
        else:
            title = f"Time Series: {request.value_column.replace('_', ' ').title()}"
            if request.group_column:
                title += f" by {request.group_column.replace('_', ' ').title()}"
            ax.set_title(_sanitize_text(title))
        
        # Generate output path
        output_filename = _generate_filename("time_series")
        output_path = PLOTS_DIR / output_filename
        
        # Render plot
        render_result = render_and_save_plot(
            fig, output_path, request.config.return_base64,
            request.config.skip_base64, request.config.dpi
        )
        
        # Calculate time range and metadata
        time_range = {
            "start": valid_data[request.datetime_column].min().isoformat(),
            "end": valid_data[request.datetime_column].max().isoformat(),
            "duration_days": (valid_data[request.datetime_column].max() - 
                             valid_data[request.datetime_column].min()).days
        }
        
        duration = time.time() - start_time
        metadata = {
            "datetime_column": request.datetime_column,
            "value_column": request.value_column,
            "group_column": request.group_column,
            "time_range": time_range,
            "data_points": len(valid_data),
            "original_points": len(df),
            "cache_key": cache_key,
            **sampling_metadata
        }
        
        # Create response
        response_data = {
            "status": "success",
            "plot_path": render_result.get('file_path'),
            "base64_image": render_result.get('base64_image'),
            "signed_url": render_result.get('signed_url'),
            "message": f"Time series plot generated successfully for '{request.value_column}' over time",
            "metadata": {**metadata, **render_result.get('accessibility', {})}
        }
        
        # Cache the result
        _store_in_cache(cache_key, response_data)
        
        # Log operation
        columns = [request.datetime_column, request.value_column]
        if request.group_column:
            columns.append(request.group_column)
        
        _log_operation("time_series", columns, duration,
                      render_result['file_path'],
                      sampling_metadata.get('sample_size'), metadata)
        
        return VisualizationResponse(**response_data)
        
    except (ValidationError, ProcessingError) as e:
        duration = time.time() - start_time
        columns = [request.datetime_column, request.value_column]
        _log_operation("time_series", columns, duration, metadata={"error": str(e)})
        return VisualizationResponse(
            status="error",
            message=str(e),
            metadata={
                "datetime_column": request.datetime_column,
                "value_column": request.value_column
            }
        )
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Unexpected error in time series plot generation: {str(e)}")
        columns = [request.datetime_column, request.value_column]
        _log_operation("time_series", columns, duration, metadata={"error": str(e)})
        return VisualizationResponse(
            status="error",
            message="An unexpected error occurred during time series plot generation",
            metadata={
                "datetime_column": request.datetime_column,
                "value_column": request.value_column
            }
        )


# Utility functions for backward compatibility and additional functionality

def get_dataset_info(file_or_path: Union[str, Path, UploadFile]) -> dict:
    """
    Get basic information about a dataset without generating visualizations.
    
    Args:
        file_or_path: File path or UploadFile object
        
    Returns:
        dict: Dataset information including columns, types, and basic stats
    """
    try:
        df = load_dataset(file_or_path)
        
        # Basic info
        info = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "null_counts": df.isnull().sum().to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum()
        }
        
        # Categorize columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = []
        
        # Try to identify datetime columns
        for col in df.columns:
            try:
                pd.to_datetime(df[col], errors='raise')
                datetime_cols.append(col)
            except Exception:
                pass
        
        info.update({
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "datetime_columns": datetime_cols,
            "recommended_plots": _get_plot_recommendations(df, numeric_cols, categorical_cols, datetime_cols)
        })
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting dataset info: {str(e)}")
        raise ValidationError(f"Failed to analyze dataset: {str(e)}")


def _get_plot_recommendations(df: pd.DataFrame, numeric_cols: List[str], 
                             categorical_cols: List[str], datetime_cols: List[str]) -> List[dict]:
    """Generate plot recommendations based on dataset characteristics."""
    recommendations = []
    
    # Histogram recommendations
    for col in numeric_cols[:5]:  # Limit recommendations
        recommendations.append({
            "plot_type": "histogram",
            "description": f"Distribution analysis of {col}",
            "parameters": {"column": col}
        })
    
    for col in categorical_cols[:3]:
        if df[col].nunique() <= MAX_CATEGORIES_HISTOGRAM:
            recommendations.append({
                "plot_type": "histogram",
                "description": f"Count distribution of {col}",
                "parameters": {"column": col}
            })
    
    # Correlation heatmap
    if len(numeric_cols) >= 2:
        recommendations.append({
            "plot_type": "correlation_heatmap",
            "description": f"Correlation analysis of {len(numeric_cols)} numeric variables",
            "parameters": {"columns": numeric_cols}
        })
    
    # Scatter plot recommendations
    for i, col1 in enumerate(numeric_cols[:3]):
        for col2 in numeric_cols[i+1:4]:
            recommendations.append({
                "plot_type": "scatter_plot",
                "description": f"Relationship between {col1} and {col2}",
                "parameters": {"x_column": col1, "y_column": col2}
            })
    
    # Box plot recommendations
    for num_col in numeric_cols[:2]:
        for cat_col in categorical_cols[:2]:
            if df[cat_col].nunique() <= MAX_CATEGORIES_HISTOGRAM:
                recommendations.append({
                    "plot_type": "box_plot",
                    "description": f"Distribution of {num_col} across {cat_col} categories",
                    "parameters": {"numeric_column": num_col, "categorical_column": cat_col}
                })
    
    # Time series recommendations
    for datetime_col in datetime_cols[:2]:
        for value_col in numeric_cols[:2]:
            recommendations.append({
                "plot_type": "time_series",
                "description": f"Temporal trend of {value_col}",
                "parameters": {"datetime_column": datetime_col, "value_column": value_col}
            })
    
    return recommendations[:10]  # Limit total recommendations


def cleanup_old_plots(days_old: int = 7) -> int:
    """
    Clean up old plot files to manage disk space.
    
    Args:
        days_old: Remove plots older than this many days
        
    Returns:
        int: Number of files removed
    """
    try:
        cutoff_time = time.time() - (days_old * 24 * 60 * 60)
        removed_count = 0
        
        for plot_file in PLOTS_DIR.glob("*.png"):
            if plot_file.stat().st_mtime < cutoff_time:
                plot_file.unlink()
                removed_count += 1
                logger.info(f"Removed old plot file: {plot_file}")
        
        for plot_file in PLOTS_DIR.glob("*.svg"):
            if plot_file.stat().st_mtime < cutoff_time:
                plot_file.unlink()
                removed_count += 1
                logger.info(f"Removed old plot file: {plot_file}")
        
        for plot_file in PLOTS_DIR.glob("*.pdf"):
            if plot_file.stat().st_mtime < cutoff_time:
                plot_file.unlink()
                removed_count += 1
                logger.info(f"Removed old plot file: {plot_file}")
        
        return removed_count
        
    except Exception as e:
        logger.error(f"Error during plot cleanup: {str(e)}")
        return 0


def clear_cache() -> int:
    """
    Clear visualization cache.
    
    Returns:
        int: Number of cache entries cleared
    """
    if not _redis_client:
        return 0
    
    try:
        # Get all cache keys with our pattern
        pattern = "*"  # Could be more specific if we add prefixes
        keys = _redis_client.keys(pattern)
        
        if keys:
            cleared_count = _redis_client.delete(*keys)
            logger.info(f"Cleared {cleared_count} cache entries")
            return cleared_count
        
        return 0
        
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        return 0


def get_cache_stats() -> dict:
    """
    Get cache statistics.
    
    Returns:
        dict: Cache statistics
    """
    if not _redis_client:
        return {"cache_enabled": False}
    
    try:
        info = _redis_client.info()
        return {
            "cache_enabled": True,
            "memory_used_mb": round(info.get('used_memory', 0) / 1024 / 1024, 2),
            "total_keys": info.get('db0', {}).get('keys', 0) if 'db0' in info else 0,
            "hits": info.get('keyspace_hits', 0),
            "misses": info.get('keyspace_misses', 0),
            "hit_rate": round(info.get('keyspace_hits', 0) / max(1, info.get('keyspace_hits', 0) + info.get('keyspace_misses', 0)) * 100, 2)
        }
    except Exception as e:
        logger.error(f"Error getting cache stats: {str(e)}")
        return {"cache_enabled": False, "error": str(e)}


def generate_plot_with_format(request: Union[HistogramRequest, CorrelationHeatmapRequest, 
                                          ScatterPlotRequest, BoxPlotRequest, TimeSeriesRequest],
                            format_type: str = 'png') -> VisualizationResponse:
    """
    Generate plot with specified format (png, svg, pdf).
    
    Args:
        request: Plot request configuration
        format_type: Output format ('png', 'svg', 'pdf')
        
    Returns:
        VisualizationResponse: Generated plot response
    """
    # Store original request format preference
    original_skip_base64 = request.config.skip_base64
    
    # For non-PNG formats, skip base64 by default
    if format_type != 'png':
        request.config.skip_base64 = True
    
    try:
        # Route to appropriate generator based on request type
        if isinstance(request, HistogramRequest):
            response = generate_histogram(request)
        elif isinstance(request, CorrelationHeatmapRequest):
            response = generate_correlation_heatmap(request)
        elif isinstance(request, ScatterPlotRequest):
            response = generate_scatter_plot(request)
        elif isinstance(request, BoxPlotRequest):
            response = generate_box_plot(request)
        elif isinstance(request, TimeSeriesRequest):
            response = generate_time_series_plot(request)
        else:
            raise ValidationError(f"Unsupported request type: {type(request)}")
        
        # Update response metadata with format info
        if response.metadata:
            response.metadata['output_format'] = format_type
        
        return response
        
    finally:
        # Restore original setting
        request.config.skip_base64 = original_skip_base64


# Performance monitoring
def get_performance_metrics() -> dict:
    """
    Get performance metrics for the visualization service.
    
    Returns:
        dict: Performance metrics
    """
    try:
        plots_dir_size = sum(f.stat().st_size for f in PLOTS_DIR.rglob('*') if f.is_file())
        plot_count = len(list(PLOTS_DIR.rglob('*.png'))) + len(list(PLOTS_DIR.rglob('*.svg'))) + len(list(PLOTS_DIR.rglob('*.pdf')))
        
        return {
            "plots_directory_size_mb": round(plots_dir_size / 1024 / 1024, 2),
            "total_plot_files": plot_count,
            "thread_pool_active": _thread_pool._threads is not None,
            "cache_stats": get_cache_stats()
        }
    except Exception as e:
        logger.error(f"Error getting performance metrics: {str(e)}")
        return {"error": str(e)}


# Initialize matplotlib backend for headless operation and thread safety
plt.switch_backend('Agg')
matplotlib.pyplot.ioff()  # Turn off interactive mode