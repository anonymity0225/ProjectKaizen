"""
Production-ready data transformation module for FastAPI backend.
Provides comprehensive data transformation capabilities with enterprise-grade security,
performance optimization, resource management, and observability.
"""

import logging
import threading
import time
import json
import hashlib
import subprocess
import tempfile
import os
import pickle
import signal
import multiprocessing
import ast
import gc
import resource
import uuid
import weakref
import sqlite3
import math
import mmap
import struct
import socket
import re
import ctypes
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Callable, Any, Iterator, Set
from datetime import datetime, timezone, timedelta
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from collections import deque, defaultdict, Counter
from threading import RLock, Event, Condition
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from queue import Queue, Empty, Full, PriorityQueue
import copy
import pickle
import warnings
import io

import pandas as pd
import numpy as np
import docker
from docker.errors import DockerException
import psutil
import magic
try:
    import clamav
    CLAMAV_AVAILABLE = True
except ImportError:
    CLAMAV_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.instrumentation.logging import LoggingInstrumentor
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.decomposition import PCA, IncrementalPCA

# Optional imports
try:
    import dask.dataframe as dd
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

try:
    from prometheus_client import Counter, Histogram, Gauge
    METRICS_AVAILABLE = True
    # Metrics
    transformation_counter = Counter('transformations_total', 'Total transformations applied', ['method', 'status'])
    transformation_duration = Histogram('transformation_duration_seconds', 'Time spent on transformations', ['method'])
    dataset_size_gauge = Gauge('dataset_size_rows', 'Number of rows in dataset being processed')
except ImportError:
    METRICS_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# MVP-1: Simplified sandboxing will rely on Docker only within SafeFunctionExecutor

# ==========================================
# CRITICAL FIX 2: ADVANCED CODE COMPLEXITY ANALYSIS
# ==========================================

class ComplexityType(IntEnum):
    """Types of complexity analysis."""
    CYCLOMATIC = 1  # McCabe complexity
    TEMPORAL = 2    # Time complexity
    SPATIAL = 3     # Space complexity
    HALSTEAD = 4    # Halstead complexity

@dataclass
class ComplexityAnalysis:
    """Comprehensive code complexity analysis results."""
    cyclomatic_complexity: int = 0
    temporal_complexity_class: str = "O(1)"
    spatial_complexity_class: str = "O(1)"
    halstead_difficulty: float = 0.0
    recursive_depth: int = 0
    loop_nesting_depth: int = 0
    infinite_loop_risk: bool = False
    recursive_bomb_risk: bool = False
    memory_allocation_risk: bool = False
    algorithmic_complexity_score: float = 0.0
    
class AdvancedComplexityAnalyzer:
    """Advanced static analysis for algorithmic complexity and security risks."""
    
    def __init__(self):
        self._loop_patterns = {
            'for': r'for\s+\w+\s+in\s+',
            'while': r'while\s+.*:',
            'nested_loop': r'for\s+.*:\s*\n\s*for\s+',
            'infinite_while': r'while\s+True\s*:',
            'infinite_for': r'for\s+.*\s+in\s+itertools\.count'
        }
        
        self._recursive_patterns = {
            'direct_recursion': r'def\s+(\w+).*:\s*.*\1\s*\(',
            'mutual_recursion': r'def\s+\w+.*:\s*.*def\s+\w+.*:\s*.*'
        }
        
        self._memory_patterns = {
            'list_comprehension': r'\[.*for.*in.*\]',
            'generator_abuse': r'.*\.join\s*\(\s*.*for.*in.*\)',
            'large_allocation': r'(np\.zeros|np\.ones|np\.empty)\s*\(\s*\d{6,}',
            'memory_copy': r'(copy\.deepcopy|\.copy\(\))',
            'string_concat': r'\s*\+\s*.*\s*\+\s*'
        }
    
    def analyze_complexity(self, code: str) -> ComplexityAnalysis:
        """Perform comprehensive complexity analysis."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            logger.error(f"Syntax error in code analysis: {e}")
            return ComplexityAnalysis()
        
        analysis = ComplexityAnalysis()
        
        # McCabe cyclomatic complexity
        analysis.cyclomatic_complexity = self._calculate_cyclomatic_complexity(tree)
        
        # Detect algorithmic complexity patterns
        analysis.temporal_complexity_class = self._analyze_temporal_complexity(code, tree)
        analysis.spatial_complexity_class = self._analyze_spatial_complexity(code, tree)
        
        # Detect dangerous patterns
        analysis.infinite_loop_risk = self._detect_infinite_loops(code)
        analysis.recursive_bomb_risk = self._detect_recursive_bombs(code, tree)
        analysis.memory_allocation_risk = self._detect_memory_risks(code)
        
        # Calculate nesting depths
        analysis.loop_nesting_depth = self._calculate_loop_nesting(tree)
        analysis.recursive_depth = self._calculate_recursive_depth(tree)
        
        # Halstead complexity
        analysis.halstead_difficulty = self._calculate_halstead_difficulty(tree)
        
        # Overall algorithmic complexity score
        analysis.algorithmic_complexity_score = self._calculate_overall_score(analysis)
        
        return analysis
    
    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """Calculate McCabe cyclomatic complexity."""
        complexity = 1
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, (ast.And, ast.Or)):
                complexity += 1
            elif isinstance(node, ast.comprehension):
                complexity += 1
        
        return complexity
    
    def _analyze_temporal_complexity(self, code: str, tree: ast.AST) -> str:
        """Analyze time complexity patterns."""
        # Look for nested loops
        nested_loops = len(re.findall(self._loop_patterns['nested_loop'], code, re.MULTILINE))
        if nested_loops >= 3:
            return "O(n³) or worse"
        elif nested_loops >= 2:
            return "O(n²)"
        elif nested_loops >= 1:
            return "O(n²)"
        
        # Look for single loops
        loops = len(re.findall(self._loop_patterns['for'], code)) + len(re.findall(self._loop_patterns['while'], code))
        if loops > 0:
            return "O(n)"
        
        # Look for recursive patterns
        if self._detect_recursive_patterns(code):
            return "O(2ⁿ) - exponential"
        
        return "O(1)"
    
    def _analyze_spatial_complexity(self, code: str, tree: ast.AST) -> str:
        """Analyze space complexity patterns."""
        # Look for large data structure creation
        if re.search(self._memory_patterns['large_allocation'], code):
            return "O(n) - large allocation"
        
        # Look for recursive calls (stack space)
        recursive_depth = self._calculate_recursive_depth(tree)
        if recursive_depth > 100:
            return "O(n) - deep recursion"
        
        # Look for data structure copying
        if re.search(self._memory_patterns['memory_copy'], code):
            return "O(n) - copying overhead"
        
        return "O(1)"
    
    def _detect_infinite_loops(self, code: str) -> bool:
        """Detect potential infinite loop patterns."""
        # Check for obvious infinite loops
        if re.search(self._loop_patterns['infinite_while'], code):
            return True
        if re.search(self._loop_patterns['infinite_for'], code):
            return True
        
        # Check for loops without obvious termination
        while_loops = re.findall(r'while\s+([^:]+):', code)
        for condition in while_loops:
            if 'True' in condition and 'break' not in code:
                return True
        
        return False
    
    def _detect_recursive_bombs(self, code: str, tree: ast.AST) -> bool:
        """Detect recursive bomb patterns."""
        # Find function definitions
        functions = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions[node.name] = node
        
        # Check for functions that call themselves without proper base case
        for func_name, func_node in functions.items():
            calls_self = False
            has_base_case = False
            
            for node in ast.walk(func_node):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id == func_name:
                        calls_self = True
                elif isinstance(node, ast.Return):
                    has_base_case = True
            
            if calls_self and not has_base_case:
                return True
        
        return False
    
    def _detect_memory_risks(self, code: str) -> bool:
        """Detect memory allocation risks."""
        # Check for large allocations
        if re.search(self._memory_patterns['large_allocation'], code):
            return True
        
        # Check for problematic string concatenation in loops
        if re.search(r'for.*:.*\w+\s*\+=\s*.*', code):
            return True
        
        # Check for generator abuse
        if re.search(self._memory_patterns['generator_abuse'], code):
            return True
        
        return False
    
    def _calculate_loop_nesting(self, tree: ast.AST) -> int:
        """Calculate maximum loop nesting depth."""
        max_depth = 0
        
        def visit_node(node, current_depth=0):
            nonlocal max_depth
            if isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            
            for child in ast.iter_child_nodes(node):
                visit_node(child, current_depth)
        
        visit_node(tree)
        return max_depth
    
    def _calculate_recursive_depth(self, tree: ast.AST) -> int:
        """Calculate potential recursive depth."""
        # Simplified heuristic - count recursive calls
        recursive_calls = 0
        
        functions = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions[node.name] = node
        
        for func_name, func_node in functions.items():
            for node in ast.walk(func_node):
                if (isinstance(node, ast.Call) and 
                    isinstance(node.func, ast.Name) and 
                    node.func.id == func_name):
                    recursive_calls += 1
        
        return recursive_calls * 10  # Heuristic multiplier
    
    def _calculate_halstead_difficulty(self, tree: ast.AST) -> float:
        """Calculate Halstead difficulty metric."""
        operators = set()
        operands = set()
        operator_count = 0
        operand_count = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, 
                                ast.Pow, ast.LShift, ast.RShift, ast.BitOr, 
                                ast.BitXor, ast.BitAnd, ast.FloorDiv)):
                operators.add(type(node).__name__)
                operator_count += 1
            elif isinstance(node, (ast.Name, ast.Constant)):
                if isinstance(node, ast.Name):
                    operands.add(node.id)
                operand_count += 1
        
        n1 = len(operators)  # number of distinct operators
        n2 = len(operands)   # number of distinct operands
        N1 = operator_count  # total occurrences of operators
        N2 = operand_count   # total occurrences of operands
        
        if n2 == 0 or N2 == 0:
            return 0.0
        
        vocabulary = n1 + n2
        length = N1 + N2
        volume = length * math.log2(vocabulary) if vocabulary > 1 else 0
        difficulty = (n1 / 2) * (N2 / n2) if n2 > 0 else 0
        
        return difficulty
    
    def _calculate_overall_score(self, analysis: ComplexityAnalysis) -> float:
        """Calculate overall algorithmic complexity risk score."""
        score = 0.0
        
        # Cyclomatic complexity weight
        score += min(analysis.cyclomatic_complexity / 10.0, 1.0) * 0.2
        
        # Temporal complexity weight
        temporal_weights = {
            "O(1)": 0.0,
            "O(log n)": 0.1,
            "O(n)": 0.3,
            "O(n log n)": 0.5,
            "O(n²)": 0.7,
            "O(n³) or worse": 0.9,
            "O(2ⁿ) - exponential": 1.0
        }
        score += temporal_weights.get(analysis.temporal_complexity_class, 0.5) * 0.3
        
        # Risk factors
        if analysis.infinite_loop_risk:
            score += 0.3
        if analysis.recursive_bomb_risk:
            score += 0.2
        if analysis.memory_allocation_risk:
            score += 0.2
        
        # Nesting depth penalty
        score += min(analysis.loop_nesting_depth / 5.0, 1.0) * 0.1
        score += min(analysis.recursive_depth / 100.0, 1.0) * 0.1
        
        return min(score, 1.0)
    
    def _detect_recursive_patterns(self, code: str) -> bool:
        """Detect recursive call patterns."""
        return bool(re.search(self._recursive_patterns['direct_recursion'], code, re.MULTILINE))

# Global complexity analyzer instance
_advanced_complexity_analyzer = AdvancedComplexityAnalyzer()

# MVP-1: SimpleResourceMonitor with 1s interval and psutil only
@dataclass
class ResourceSample:
    timestamp: float
    memory_mb: float
    cpu_percent: float
    threads: int

class SimpleResourceMonitor:
    def __init__(self, interval_seconds: float = 1.0):
        self.interval_seconds = interval_seconds
        self._monitoring = False
        self._shutdown_event = Event()
        self._lock = RLock()
        self._monitor_thread: Optional[threading.Thread] = None
        self.sample_history: deque = deque(maxlen=60)

    def start(self) -> None:
        with self._lock:
            if self._monitoring:
                return
            self._monitoring = True
            self._shutdown_event.clear()
            self._monitor_thread = threading.Thread(target=self._loop, name="SimpleResourceMonitor", daemon=True)
            self._monitor_thread.start()
    
    def stop(self) -> None:
        with self._lock:
            if not self._monitoring:
                return
            self._monitoring = False
            self._shutdown_event.set()
            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=5.0)
            
    def _loop(self) -> None:
        while not self._shutdown_event.wait(self.interval_seconds):
            try:
                cpu_percent = psutil.cpu_percent(interval=None)
                mem = psutil.virtual_memory()
                threads = len(psutil.pids())
                sample = ResourceSample(
                    timestamp=time.time(),
                    memory_mb=mem.used / (1024 * 1024),
                    cpu_percent=cpu_percent,
                    threads=threads
                )
                with self._lock:
                    self.sample_history.append(sample)
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")

# ==========================================
# CRITICAL FIX 4: PERSISTENT TRANSFORMATION REGISTRY
# ==========================================

@dataclass
class TransformationRecord:
    """Persistent transformation record."""
    transformation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    operation_type: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    input_fingerprint: str = ""
    output_fingerprint: str = ""
    execution_time_ms: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"  # pending, running, completed, failed
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class PersistentTransformationRegistry:
    """Database-backed transformation registry with distributed locking and DAG validation."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        self._lock = RLock()
        
        # Distributed locking (Redis-based in production)
        self._distributed_locks = {}
        self._local_locks = defaultdict(RLock)
        
    def _init_database(self):
        """Initialize SQLite database for transformation registry."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS transformations (
                    transformation_id TEXT PRIMARY KEY,
                    operation_type TEXT,
                    parameters TEXT,
                    input_fingerprint TEXT,
                    output_fingerprint TEXT,
                    execution_time_ms REAL,
                    created_at TIMESTAMP,
                    dependencies TEXT,
                    status TEXT,
                    error_message TEXT,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_transformations_status 
                ON transformations(status)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_transformations_type 
                ON transformations(operation_type)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_transformations_fingerprint 
                ON transformations(input_fingerprint)
            """)
            
            # DAG validation table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS transformation_dag (
                    parent_id TEXT,
                    child_id TEXT,
                    created_at TIMESTAMP,
                    PRIMARY KEY (parent_id, child_id)
                )
            """)
    
    def register_transformation(self, record: TransformationRecord) -> bool:
        """Register a transformation with DAG validation."""
        try:
            # Validate DAG before insertion
            if not self._validate_dag(record):
                logger.error(f"DAG validation failed for transformation: {record.transformation_id}")
                return False
            
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO transformations 
                        (transformation_id, operation_type, parameters, input_fingerprint,
                         output_fingerprint, execution_time_ms, created_at, dependencies,
                         status, error_message, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        record.transformation_id,
                        record.operation_type,
                        json.dumps(record.parameters),
                        record.input_fingerprint,
                        record.output_fingerprint,
                        record.execution_time_ms,
                        record.created_at.isoformat(),
                        json.dumps(record.dependencies),
                        record.status,
                        record.error_message,
                        json.dumps(record.metadata)
                    ))
                    
                    # Update DAG edges
                    for dep_id in record.dependencies:
                        conn.execute("""
                            INSERT OR IGNORE INTO transformation_dag 
                            (parent_id, child_id, created_at)
                            VALUES (?, ?, ?)
                        """, (dep_id, record.transformation_id, datetime.now(timezone.utc).isoformat()))
            
            return True
        except Exception as e:
            logger.error(f"Failed to register transformation: {e}")
            return False
    
    def _validate_dag(self, record: TransformationRecord) -> bool:
        """Validate that adding this transformation won't create cycles."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check if adding this transformation would create a cycle
                for dep_id in record.dependencies:
                    if self._creates_cycle(conn, dep_id, record.transformation_id):
                        return False
            return True
        except Exception as e:
            logger.error(f"DAG validation error: {e}")
            return False
    
    def _creates_cycle(self, conn, parent_id: str, child_id: str) -> bool:
        """Check if adding edge parent->child creates a cycle."""
        if parent_id == child_id:
            return True
        
        # BFS to check if child_id can reach parent_id
        visited = set()
        queue = [child_id]
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            
            if current == parent_id:
                return True
            
            # Get children of current node
            cursor = conn.execute("""
                SELECT child_id FROM transformation_dag WHERE parent_id = ?
            """, (current,))
            
            for (child,) in cursor.fetchall():
                if child not in visited:
                    queue.append(child)
        
        return False
    
    def get_transformation_chain(self, transformation_id: str) -> List[TransformationRecord]:
        """Get complete transformation chain (dependencies)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Use recursive CTE to get dependency chain
                cursor = conn.execute("""
                    WITH RECURSIVE dependency_chain AS (
                        -- Base case: start with the given transformation
                        SELECT transformation_id, dependencies, 0 as level
                        FROM transformations 
                        WHERE transformation_id = ?
                        
                        UNION ALL
                        
                        -- Recursive case: find dependencies
                        SELECT t.transformation_id, t.dependencies, dc.level + 1
                        FROM transformations t
                        JOIN dependency_chain dc ON t.transformation_id IN (
                            SELECT value FROM json_each(dc.dependencies)
                        )
                        WHERE dc.level < 10  -- Prevent infinite recursion
                    )
                    SELECT DISTINCT t.* FROM transformations t
                    JOIN dependency_chain dc ON t.transformation_id = dc.transformation_id
                    ORDER BY dc.level
                """, (transformation_id,))
                
                records = []
                for row in cursor.fetchall():
                    record = TransformationRecord(
                        transformation_id=row[0],
                        operation_type=row[1],
                        parameters=json.loads(row[2]),
                        input_fingerprint=row[3],
                        output_fingerprint=row[4],
                        execution_time_ms=row[5],
                        created_at=datetime.fromisoformat(row[6]),
                        dependencies=json.loads(row[7]),
                        status=row[8],
                        error_message=row[9],
                        metadata=json.loads(row[10])
                    )
                    records.append(record)
                
                return records
        except Exception as e:
            logger.error(f"Failed to get transformation chain: {e}")
            return []
    
    def acquire_distributed_lock(self, lock_key: str, timeout_seconds: int = 30) -> bool:
        """Acquire distributed lock for concurrent transformation protection."""
        # In production, this would use Redis SETNX with expiration
        # For now, use local locks as fallback
        try:
            lock = self._local_locks[lock_key]
            acquired = lock.acquire(timeout=timeout_seconds)
            if acquired:
                self._distributed_locks[lock_key] = time.time() + timeout_seconds
            return acquired
        except Exception as e:
            logger.error(f"Failed to acquire distributed lock: {e}")
            return False
    
    def release_distributed_lock(self, lock_key: str) -> bool:
        """Release distributed lock."""
        try:
            if lock_key in self._local_locks:
                self._local_locks[lock_key].release()
                self._distributed_locks.pop(lock_key, None)
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to release distributed lock: {e}")
            return False

# ==========================================
# CRITICAL FIX 5: ENHANCED DATA SAMPLING STRATEGIES
# ==========================================

class SamplingStrategy(Enum):
    """Advanced sampling strategies for different data types."""
    RANDOM = "random"
    STRATIFIED = "stratified"

@dataclass
class SamplingConfig:
    """Configuration for data sampling."""
    strategy: SamplingStrategy = SamplingStrategy.STRATIFIED
    sample_size: int = 10000
    preserve_distributions: bool = True
    time_column: Optional[str] = None
    stratify_columns: List[str] = field(default_factory=list)
    random_seed: int = 42
    temporal_windows: int = 0
    min_samples_per_stratum: int = 5
    reservoir_k: int = 0

class AdvancedDataSampler:
    """(Deprecated) Placeholder to keep file structure. Not used in MVP-1."""
    def __init__(self, config: SamplingConfig = None):
        self.config = config or SamplingConfig()
        self._random_state = np.random.RandomState(self.config.random_seed)
        
    def sample_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        # MVP-1: Only random and simple stratified sampling
        if len(df) <= self.config.sample_size:
            return df.copy()
        if self.config.strategy == SamplingStrategy.STRATIFIED:
            cat_cols = self.config.stratify_columns or df.select_dtypes(include=['object', 'category']).columns.tolist()[:1]
            if not cat_cols:
                return df.sample(n=self.config.sample_size, random_state=self._random_state)
            col = cat_cols[0]
            groups = df.groupby(col)
            sampled = []
            for _, g in groups:
                take = max(1, int(len(g) / len(df) * self.config.sample_size))
                sampled.append(g.sample(n=min(take, len(g)), random_state=self._random_state))
            return pd.concat(sampled).head(self.config.sample_size)
        return df.sample(n=self.config.sample_size, random_state=self._random_state)

# ==========================================
# CRITICAL FIX 6: IDEMPOTENCY AND CACHING SYSTEM
# ==========================================

@dataclass
class TransformationFingerprint:
    """Unique fingerprint for transformation operations."""
    operation_type: str
    parameters_hash: str
    input_data_hash: str
    code_hash: str
    environment_hash: str
    random_seed: int
    version: str = "1.0.0"

@dataclass 
class CachedResult:
    """Cached transformation result with metadata."""
    fingerprint: TransformationFingerprint
    result_data: Any
    created_at: datetime
    ttl_seconds: int
    hit_count: int = 0
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

class IdempotentTransformationEngine:
    """(Deprecated) Placeholder for MVP-1; persistent caching removed."""
    def __init__(self, cache_dir: Path, default_ttl_seconds: int = 3600):
        self.cache_dir = cache_dir
        self.default_ttl_seconds = default_ttl_seconds
        self._memory_cache: Dict[str, CachedResult] = {}
        self._cache_lock = RLock()
        self._random_states: Dict[str, np.random.RandomState] = {}
        self._random_lock = RLock()
        
    def get_deterministic_random_state(self, seed: int) -> np.random.RandomState:
        with self._random_lock:
            if seed not in self._random_states:
                self._random_states[seed] = np.random.RandomState(seed)
            return self._random_states[seed]

# Global instances for critical fixes
# MVP-1: Remove complex global instances; create lazily when needed

# NLTK imports with error handling
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    # Create dummy classes to prevent errors
    class DummyTokenizer:
        def stem(self, word): return word
        def lemmatize(self, word): return word
    
    PorterStemmer = DummyTokenizer
    WordNetLemmatizer = DummyTokenizer

# Configure logging with structured format
logger = logging.getLogger(__name__)

# Initialize OpenTelemetry tracing if available
if OPENTELEMETRY_AVAILABLE:
    tracer = trace.get_tracer(__name__)
    LoggingInstrumentor().instrument(set_logging_format=True)
else:
    tracer = None

# Enhanced Constants
LARGE_DATASET_THRESHOLD = 100_000  # rows
SAMPLE_SIZE_FOR_VALIDATION = 1000
MAX_CUSTOM_FUNCTION_TIME = 30  # seconds
MAX_MEMORY_MB = 1024  # MB for custom functions
NLTK_DOWNLOAD_TIMEOUT = 30  # seconds
MAX_CONTAINER_RUNTIME = 300  # seconds
MAX_CPU_PERCENT = 80.0  # CPU usage limit
MAX_DISK_USAGE_PERCENT = 90.0  # Disk usage limit
CODE_COMPLEXITY_LIMIT = 10  # Maximum cyclomatic complexity
MAX_AST_NODES = 1000  # Maximum AST nodes in user code

# Security and performance enums
class SecurityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ResourceState(Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EXHAUSTED = "exhausted"

class TransformationState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"

# Type aliases for better readability
TransformationResult = Tuple[pd.DataFrame, Dict[str, Any]]
TransformationMetadata = Dict[str, Any]
TransformationPipeline = List[Dict[str, Any]]

class TransformationError(Exception):
    """Custom exception for transformation errors."""
    pass

class ResourceLimitError(Exception):
    """Exception for resource limit violations."""
    pass

class SandboxExecutionError(Exception):
    """Exception for sandbox execution failures."""
    pass

class SecurityViolationError(Exception):
    """Exception for security violations."""
    pass

class CodeComplexityError(Exception):
    """Exception for code complexity violations."""
    pass

@dataclass
class ResourceUsage:
    """Resource usage metrics."""
    memory_mb: float = 0.0
    cpu_percent: float = 0.0
    disk_usage_percent: float = 0.0
    execution_time: float = 0.0
    timestamp: float = field(default_factory=time.time)

@dataclass
class SecurityAnalysis:
    """Security analysis results."""
    level: SecurityLevel
    violations: List[str] = field(default_factory=list)
    complexity_score: int = 0
    ast_node_count: int = 0
    forbidden_imports: List[str] = field(default_factory=list)
    dangerous_calls: List[str] = field(default_factory=list)

class EnhancedResourceMonitor:  # Deprecated leftover, keep minimal to satisfy references if any
    pass

class BasicCodeAnalyzer:
    """MVP-1: Basic analyzer that checks only forbidden imports and dangerous calls."""
    def __init__(self):
        self.forbidden_modules = {
            'os', 'sys', 'subprocess', 'socket', 'urllib', 'http', 'ftplib',
            'smtplib', 'telnetlib', 'webbrowser', 'multiprocessing', 'threading',
            'ctypes', 'pickle', 'marshal', 'shelve', 'dbm', 'sqlite3', 'zlib',
            'gzip', 'bz2', 'lzma', 'tarfile', 'zipfile', 'shutil', 'glob',
            'pathlib', 'tempfile', 'signal', 'resource'
        }
        self.dangerous_functions = {'exec', 'eval', 'compile', '__import__', 'open'}
    
    def analyze_code(self, code: str) -> SecurityAnalysis:
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return SecurityAnalysis(level=SecurityLevel.CRITICAL, violations=[f"Syntax error: {str(e)}"])
        
        analysis = SecurityAnalysis(level=SecurityLevel.LOW)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name.split('.')[0] for alias in node.names]
                else:
                    names = [node.module.split('.')[0]] if node.module else []
                for name in names:
                    if name in self.forbidden_modules:
                        analysis.level = SecurityLevel.CRITICAL
                        analysis.forbidden_imports.append(name)
                        analysis.violations.append(f"Forbidden import: {name}")
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in self.dangerous_functions:
                    analysis.level = SecurityLevel.CRITICAL
                    analysis.dangerous_calls.append(node.func.id)
                    analysis.violations.append(f"Dangerous call: {node.func.id}")
        return analysis

# Global resource monitor
_resource_monitor: Optional[SimpleResourceMonitor] = None
_code_analyzer = BasicCodeAnalyzer()

def get_resource_monitor() -> SimpleResourceMonitor:
    """Get or create global resource monitor."""
    global _resource_monitor
    if _resource_monitor is None:
        _resource_monitor = SimpleResourceMonitor()
        _resource_monitor.start()
    return _resource_monitor

class NLTKResourceManager:
    """Manages NLTK resources with fallback mechanisms."""
    
    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path or os.path.join(os.getcwd(), 'nltk_data')
        self._downloaded = set()
        self._lock = threading.Lock()
        
        # Set NLTK data path if NLTK is available
        if NLTK_AVAILABLE:
            if self.data_path not in nltk.data.path:
                nltk.data.path.insert(0, self.data_path)
    
    def ensure_resource(self, resource: str, download_name: str = None) -> bool:
        """Ensure NLTK resource is available with timeout and fallback."""
        if not NLTK_AVAILABLE:
            return False
            
        with self._lock:
            download_name = download_name or resource.split('/')[-1]
            
            if download_name in self._downloaded:
                return True
            
            try:
                nltk.data.find(resource)
                self._downloaded.add(download_name)
                return True
            except LookupError:
                pass
            
            try:
                logger.info(f"Downloading NLTK resource: {download_name}")
                
                def download_with_timeout():
                    return nltk.download(download_name, download_dir=self.data_path, quiet=True)
                
                # Use multiprocessing to enforce timeout
                with multiprocessing.Pool(processes=1) as pool:
                    try:
                        result = pool.apply_async(download_with_timeout)
                        success = result.get(timeout=NLTK_DOWNLOAD_TIMEOUT)
                        if success:
                            self._downloaded.add(download_name)
                            return True
                    except multiprocessing.TimeoutError:
                        logger.warning(f"NLTK download timeout for {download_name}")
                    except Exception as e:
                        logger.warning(f"NLTK download error for {download_name}: {e}")
                        
            except Exception as e:
                logger.warning(f"Failed to download NLTK resource {download_name}: {e}")
            
            return False
    
    def get_fallback_stopwords(self) -> set:
        """Return hardcoded stopwords as fallback."""
        return {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
            'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
            'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
            'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
            'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
            'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after', 'above',
            'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once'
        }
    
    def simple_tokenize(self, text: str) -> List[str]:
        """Simple tokenization fallback when NLTK is unavailable."""
        import re
        # Simple regex-based tokenization
        tokens = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
        return tokens

class TransformationRegistry:
    """Registry to track all transformations applied to a dataset with persistence."""
    
    def __init__(self):
        self.operations: List[TransformationMetadata] = []
        self.pipeline_config: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._pipeline_hash: Optional[str] = None
    
    def add_operation(self, metadata: TransformationMetadata) -> None:
        """Add a transformation operation to the registry (thread-safe)."""
        with self._lock:
            # Add timestamp
            metadata['timestamp'] = datetime.now().isoformat()
            self.operations.append(metadata)
            
            # Update metrics if available
            if METRICS_AVAILABLE:
                transformation_counter.labels(
                    method=metadata.get('method', 'unknown'),
                    status=metadata.get('status', 'unknown')
                ).inc()
    
    def add_to_pipeline(self, config: Dict[str, Any]) -> None:
        """Add transformation configuration to reproducible pipeline."""
        with self._lock:
            config_copy = config.copy()
            config_copy['timestamp'] = datetime.now().isoformat()
            self.pipeline_config.append(config_copy)
            self._pipeline_hash = None  # Invalidate hash
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of all applied transformations (thread-safe)."""
        with self._lock:
            return {
                "total_operations": len(self.operations),
                "operations": self.operations.copy(),
                "pipeline_config": self.pipeline_config.copy(),
                "pipeline_hash": self.get_pipeline_hash(),
            "columns_modified": list(set().union(*[op.get("columns_affected", []) for op in self.operations]) if self.operations else []),
            "columns_added": [],
            "columns_dropped": []
            }
    
    def get_pipeline_hash(self) -> str:
        """Get deterministic hash of the transformation pipeline."""
        if self._pipeline_hash is None:
            pipeline_str = json.dumps(self.pipeline_config, sort_keys=True, default=str)
            self._pipeline_hash = hashlib.sha256(pipeline_str.encode()).hexdigest()[:16]
        return self._pipeline_hash
    
    def export_pipeline(self, filepath: str) -> None:
        """Export transformation pipeline to file for reproducibility."""
        with self._lock:
            pipeline_data = {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "pipeline_hash": self.get_pipeline_hash(),
                "operations": self.operations.copy(),
                "pipeline_config": self.pipeline_config.copy()
            }
            
            with open(filepath, 'w') as f:
                json.dump(pipeline_data, f, indent=2, default=str)
            
            logger.info(f"Pipeline exported to {filepath}")
    
    def import_pipeline(self, filepath: str) -> None:
        """Import transformation pipeline from file."""
        with self._lock:
            with open(filepath, 'r') as f:
                pipeline_data = json.load(f)
            
            self.operations = pipeline_data.get('operations', [])
            self.pipeline_config = pipeline_data.get('pipeline_config', [])
            self._pipeline_hash = None
            
            logger.info(f"Pipeline imported from {filepath}")
    
    def clear(self) -> None:
        """Clear all recorded operations (thread-safe)."""
        with self._lock:
            self.operations.clear()
            self.pipeline_config.clear()
            self._pipeline_hash = None

# Global instances
transformation_registry = TransformationRegistry()
nltk_manager = NLTKResourceManager()

@contextmanager
def transformation_timer(method_name: str):
    """Context manager for timing transformations."""
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        logger.debug(f"Transformation '{method_name}' took {duration:.2f}s")
        if METRICS_AVAILABLE:
            transformation_duration.labels(method=method_name).observe(duration)

def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Operation timed out")

@contextmanager
def execution_timeout(seconds: int):
    """Context manager for execution timeout."""
    if os.name != 'nt':  # Unix-like systems
        # Set up signal handler
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        # Windows doesn't support SIGALRM, just yield without timeout
        yield

def _validate_dataframe(df: pd.DataFrame) -> None:
    """Validate input DataFrame with size tracking."""
    if not isinstance(df, pd.DataFrame):
        raise TransformationError("Input must be a pandas DataFrame")
    if df.empty:
        raise TransformationError("DataFrame cannot be empty")
    
    # Track dataset size
    if METRICS_AVAILABLE:
        dataset_size_gauge.set(len(df))

def _validate_column_exists(df: pd.DataFrame, column: str) -> None:
    """Validate that column exists in DataFrame."""
    if column not in df.columns:
        raise TransformationError(f"Column '{column}' not found in DataFrame")

def _is_large_dataset(df: pd.DataFrame) -> bool:
    """Check if dataset is considered large."""
    return len(df) > LARGE_DATASET_THRESHOLD

def _create_success_metadata(
    method: str,
    message: str,
    transformed_columns: List[str] = None,
    added_columns: List[str] = None,
    dropped_columns: List[str] = None,
    scaled_columns: List[str] = None,
    affected_columns: List[str] = None,
    additional_info: Dict[str, Any] = None,
    random_state: Optional[int] = None
) -> TransformationMetadata:
    """Create standardized success metadata with reproducibility info."""
    metadata = {
        "status": "success",
        "method": method,
        "message": message,
        "transformed_columns": transformed_columns or [],
        "added_columns": added_columns or [],
        "dropped_columns": dropped_columns or [],
        "scaled_columns": scaled_columns or [],
        "affected_columns": affected_columns or []
    }
    
    if random_state is not None:
        metadata["random_state"] = random_state
    
    if additional_info:
        metadata.update(additional_info)
    
    return metadata

def _create_error_metadata(method: str, message: str, error: str = None) -> TransformationMetadata:
    """Create standardized error metadata."""
    return {
        "status": "error",
        "method": method,
        "message": message,
        "error": error or message,
        "transformed_columns": [],
        "added_columns": [],
        "dropped_columns": [],
        "scaled_columns": [],
        "affected_columns": []
    }

class SandboxError(Exception):
    """Exception for sandbox security violations."""
    pass

class SafeFunctionExecutor:
    """Enterprise-grade secure executor for user-defined functions with Docker isolation."""
    
    def __init__(self, 
                 max_time_seconds: int = 5,
                 max_memory_mb: int = 256,
                 use_docker: bool = True,
                 docker_image: str = "python:3.11-slim"):
        self.max_time_seconds = max_time_seconds
        self.max_memory_mb = max_memory_mb
        self.use_docker = use_docker
        self.docker_image = docker_image
        self.allowed_modules = {
            'pandas', 'numpy', 'math', 'datetime', 're', 'json', 'collections',
            'itertools', 'functools', 'operator', 'typing', 'abc'
        }
        self.forbidden_modules = {
            'os', 'sys', 'subprocess', 'socket', 'urllib', 'http', 'ftplib',
            'smtplib', 'telnetlib', 'webbrowser', 'multiprocessing', 'threading',
            'ctypes', 'pickle', 'marshal', 'shelve', 'dbm', 'sqlite3', 'zlib',
            'gzip', 'bz2', 'lzma', 'tarfile', 'zipfile', 'shutil', 'glob',
            'pathlib', 'tempfile', 'fileinput', 'linecache', 'codecs', 'io',
            'mmap', 'signal', 'resource', 'pwd', 'grp', 'crypt', 'termios',
            'tty', 'pty', 'fcntl', 'pipes', 'posix', 'nt', 'ntpath', 'posixpath',
            'stat', 'filecmp', 'fnmatch', 'glob', 'linecache', 'shlex', 'traceback',
            'warnings', 'contextlib', 'weakref', 'gc', 'inspect', 'ast', 'dis',
            'trace', 'profile', 'pstats', 'hotshot', 'timeit', 'doctest', 'unittest',
            'test', 'distutils', 'setuptools', 'pip', 'pkg_resources', 'importlib',
            'imp', 'runpy', 'pkgutil', 'modulefinder', 'py_compile', 'compileall',
            'pyclbr', 'tabnanny', 'tokenize', 'keyword', 'token', 'symbol',
            'parser', 'symbol', 'keyword', 'token', 'tokenize', 'ast', 'compiler',
            'dis', 'pickletools', 'pickle', 'copy', 'copyreg', 'shelve', 'dbm',
            'sqlite3', 'zlib', 'gzip', 'bz2', 'lzma', 'tarfile', 'zipfile',
            'shutil', 'glob', 'pathlib', 'tempfile', 'fileinput', 'linecache',
            'codecs', 'io', 'mmap', 'signal', 'resource', 'pwd', 'grp', 'crypt',
            'termios', 'tty', 'pty', 'fcntl', 'pipes', 'posix', 'nt', 'ntpath',
            'posixpath', 'stat', 'filecmp', 'fnmatch', 'glob', 'linecache',
            'shlex', 'traceback', 'warnings', 'contextlib', 'weakref', 'gc',
            'inspect', 'ast', 'dis', 'trace', 'profile', 'pstats', 'hotshot',
            'timeit', 'doctest', 'unittest', 'test', 'distutils', 'setuptools',
            'pip', 'pkg_resources', 'importlib', 'imp', 'runpy', 'pkgutil',
            'modulefinder', 'py_compile', 'compileall', 'pyclbr', 'tabnanny',
            'tokenize', 'keyword', 'token', 'symbol', 'parser', 'symbol',
            'keyword', 'token', 'tokenize', 'ast', 'compiler', 'dis', 'pickletools'
        }
        
        # Check Docker availability
        self.docker_available = self._check_docker_availability()
        if use_docker and not self.docker_available:
            logger.warning("Docker not available, falling back to restricted subprocess execution")
            self.use_docker = False
        
        # Initialize Docker client for advanced operations
        self._docker_client = None
        if self.docker_available:
            try:
                self._docker_client = docker.from_env()
            except Exception as e:
                logger.warning(f"Failed to initialize Docker client: {e}")
                self.docker_available = False
                self.use_docker = False
    
    def _check_docker_availability(self) -> bool:
        """Check if Docker is available and accessible."""
        try:
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _sanitize_code(self, code: str) -> str:
        """MVP-1: Basic sanitization using basic analyzer only."""
        analysis = _code_analyzer.analyze_code(code)
        if analysis.level == SecurityLevel.CRITICAL:
            raise SecurityViolationError(f"Security violations: {'; '.join(analysis.violations)}")
        return code
    
    def _create_docker_script(self, sanitized_code: str) -> str:
        """Create the script to run inside Docker container."""
        return f"""
import sys
import pickle
import pandas as pd
import numpy as np
import json
import time
import psutil
import signal
import os

# Set up memory monitoring
def memory_monitor():
    process = psutil.Process()
    memory_mb = process.memory_info().rss / (1024 * 1024)
    if memory_mb > {self.max_memory_mb}:
        raise MemoryError(f"Memory limit exceeded: {{memory_mb:.1f}}MB > {self.max_memory_mb}MB")

# Set up timeout handler
def timeout_handler(signum, frame):
    raise TimeoutError("Execution timeout exceeded")

# Set up signal handlers
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm({self.max_time_seconds})

try:
    # Load input data
    with open('/tmp/input.pkl', 'rb') as f:
        df = pickle.load(f)
    
    # Monitor memory before execution
    memory_monitor()
    
    # Define user function
    {sanitized_code}
    
    # Execute user function with memory monitoring
    start_time = time.time()
    memory_monitor()
    
    result = user_function(df)
    
    execution_time = time.time() - start_time
    memory_monitor()
    
    # Validate result
    if not isinstance(result, pd.DataFrame):
        raise ValueError("Function must return a pandas DataFrame")
    
    # Save result
    with open('/tmp/output.pkl', 'wb') as f:
        pickle.dump(result, f)
    
    # Save execution metadata
    metadata = {{
        "execution_time": execution_time,
        "memory_usage_mb": psutil.Process().memory_info().rss / (1024 * 1024),
        "exit_status": "success",
        "result_shape": result.shape
    }}
    
    with open('/tmp/metadata.json', 'w') as f:
        json.dump(metadata, f)
        
except Exception as e:
    # Save error information
    error_info = {{
        "error_type": type(e).__name__,
        "error_message": str(e),
        "exit_status": "error"
    }}
    
    with open('/tmp/error.json', 'w') as f:
        json.dump(error_info, f)
    
    sys.exit(1)
finally:
    signal.alarm(0)  # Cancel timeout
"""
    
    def _create_restricted_script(self, sanitized_code: str) -> str:
        """Create the script for restricted subprocess execution."""
        return f"""
import sys
import pickle
import pandas as pd
import numpy as np
import json
import time
import signal
import resource

# Set memory limit
resource.setrlimit(resource.RLIMIT_AS, ({self.max_memory_mb} * 1024 * 1024, -1))

# Set up timeout handler
def timeout_handler(signum, frame):
    raise TimeoutError("Execution timeout exceeded")

# Set up signal handlers
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm({self.max_time_seconds})

try:
    # Load input data
    with open(sys.argv[1], 'rb') as f:
        df = pickle.load(f)
    
    # Define user function
    {sanitized_code}
    
    # Execute user function
    start_time = time.time()
    result = user_function(df)
    execution_time = time.time() - start_time
    
    # Validate result
    if not isinstance(result, pd.DataFrame):
        raise ValueError("Function must return a pandas DataFrame")
    
    # Save result
    with open(sys.argv[2], 'wb') as f:
        pickle.dump(result, f)
    
    # Save execution metadata
    memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB to MB
    metadata = {{
        "execution_time": execution_time,
        "memory_usage_mb": memory_usage,
        "exit_status": "success",
        "result_shape": result.shape
    }}
    
    with open(sys.argv[3], 'w') as f:
        json.dump(metadata, f)
        
except Exception as e:
    # Save error information
    error_info = {{
        "error_type": type(e).__name__,
        "error_message": str(e),
        "exit_status": "error"
    }}
    
    with open(sys.argv[4], 'w') as f:
        json.dump(error_info, f)
    
    sys.exit(1)
finally:
    signal.alarm(0)  # Cancel timeout
"""
    
    def execute_in_docker(self, func_code: str, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute function in Docker container with strict isolation."""
        if not self.docker_available:
            raise SandboxExecutionError("Docker not available")
        
        # Sanitize code
        sanitized_code = self._sanitize_code(func_code)
        
        # Create temporary directory for data exchange
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Prepare input data
            input_file = temp_path / "input.pkl"
            with open(input_file, 'wb') as f:
                pickle.dump(df, f)
            
            # Create execution script
            script_content = self._create_docker_script(sanitized_code)
            script_file = temp_path / "script.py"
            with open(script_file, 'w') as f:
                f.write(script_content)
            
            # Enhanced Docker run command with comprehensive security
            request_id = str(uuid.uuid4())
            container_name = f"transform-{request_id[:8]}-{int(time.time())}"
            
            docker_cmd = [
                'docker', 'run',
                '--rm',  # Remove container after execution
                f'--name={container_name}',
                '--network=none',  # No network access
                f'--memory={self.max_memory_mb}m',  # Memory limit
                f'--memory-swap={self.max_memory_mb}m',  # No swap
                f'--memory-swappiness=0',  # Disable swapping
                f'--cpus={min(1.0, psutil.cpu_count() * 0.5)}',  # CPU limit
                f'--cpu-shares=512',  # Lower CPU priority
                '--pids-limit=100',  # Process limit
                '--ulimit=nofile=1024:1024',  # File descriptor limit
                '--ulimit=nproc=50:50',  # Process limit
                '--user=nobody:nogroup',  # Non-root user
                '--read-only',  # Read-only filesystem
                '--tmpfs=/tmp:rw,size=100m,noexec,nosuid,nodev',  # Secure temp space
                '--tmpfs=/var/tmp:rw,size=50m,noexec,nosuid,nodev',
                '--security-opt=no-new-privileges:true',  # Prevent privilege escalation
                '--security-opt=seccomp=default',  # Enable seccomp
                '--security-opt=apparmor=docker-default',  # Enable AppArmor
                '--cap-drop=ALL',  # Drop all capabilities
                '--cap-add=CHOWN',  # Minimal required capabilities
                '--cap-add=SETUID',
                '--cap-add=SETGID',
                '--restart=no',  # Never restart
                '--init',  # Use init process
                f'--volume={temp_path}:/workspace:ro',  # Mount workspace read-only
                f'--workdir=/workspace',
                '--env=PYTHONPATH=/workspace',
                '--env=PYTHONDONTWRITEBYTECODE=1',
                '--env=PYTHONUNBUFFERED=1',
                '--env=HOME=/tmp',
                '--label=kaizen.service=transformation',
                '--label=kaizen.security=high',
                f'--label=kaizen.request_id={request_id}',
                self.docker_image,
                'timeout', str(self.max_time_seconds),
                'python', '/workspace/script.py'
            ]
            
            logger.info(f"Executing user function in Docker container with {self.max_memory_mb}MB memory limit")
            start_time = time.time()
            
            try:
                result = subprocess.run(
                    docker_cmd,
                    timeout=self.max_time_seconds + 10,  # Extra time for container startup
                    capture_output=True,
                    text=True
                )
                
                execution_duration = time.time() - start_time
                
                if result.returncode != 0:
                    # Read error information
                    error_file = temp_path / "error.json"
                    if error_file.exists():
                        with open(error_file, 'r') as f:
                            error_info = json.load(f)
                        error_type = error_info.get('error_type', 'UnknownError')
                        error_message = error_info.get('error_message', 'Unknown error')
                        
                        if error_type == 'TimeoutError':
                            raise TimeoutError(f"Function execution timed out after {self.max_time_seconds} seconds")
                        elif error_type == 'MemoryError':
                            raise MemoryError(f"Function exceeded memory limit of {self.max_memory_mb}MB")
                        else:
                            raise SandboxExecutionError(f"Function execution failed: {error_message}")
                    else:
                        raise SandboxExecutionError(f"Docker execution failed with return code {result.returncode}")
                
                # Read result
                output_file = temp_path / "output.pkl"
                if not output_file.exists():
                    raise SandboxExecutionError("No output file generated")
                
                with open(output_file, 'rb') as f:
                    result_df = pickle.load(f)
                
                # Read metadata
                metadata_file = temp_path / "metadata.json"
                metadata = {}
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                
                metadata.update({
                    "execution_method": "docker",
                    "container_duration": execution_duration,
                    "memory_limit_mb": self.max_memory_mb,
                    "time_limit_seconds": self.max_time_seconds
                })
                
                logger.info(f"Docker execution completed successfully in {execution_duration:.2f}s")
                return result_df, metadata
                
            except subprocess.TimeoutExpired:
                raise TimeoutError(f"Docker execution timed out after {self.max_time_seconds + 10} seconds")
            except Exception as e:
                if isinstance(e, (TimeoutError, MemoryError, SandboxExecutionError)):
                    raise
                raise SandboxExecutionError(f"Docker execution failed: {str(e)}")
    
    def _monitor_container_security(self, container_name: str, request_id: str) -> None:
        """Monitor container for security violations and resource usage."""
        if not self._docker_client:
            return
        
        try:
            # Wait a moment for container to start
            time.sleep(0.5)
            container = self._docker_client.containers.get(container_name)
            
            while container.status == 'running':
                try:
                    # Check for privilege escalation attempts
                    exec_result = container.exec_run('ps aux', demux=True)
                    if exec_result.exit_code == 0:
                        processes = exec_result.output[0].decode() if exec_result.output[0] else ''
                        
                        # Look for suspicious processes
                        suspicious_patterns = ['sudo', 'su', 'chmod', 'chown', 'mount', 'umount', 'docker']
                        for pattern in suspicious_patterns:
                            if pattern in processes.lower():
                                logger.error(f"Security violation detected | container={container.id[:12]} | pattern={pattern} | request_id={request_id}")
                                container.kill()
                                return
                
                except Exception as e:
                    logger.debug(f"Container monitoring check failed: {e}")
                
                # Check resource usage
                try:
                    stats = container.stats(stream=False)
                    if 'memory_stats' in stats and 'usage' in stats['memory_stats']:
                        memory_usage = stats['memory_stats']['usage'] / (1024 * 1024)  # MB
                        
                        if memory_usage > self.max_memory_mb * 0.95:
                            logger.warning(f"Container approaching memory limit | usage={memory_usage:.1f}MB | limit={self.max_memory_mb}MB | request_id={request_id}")
                
                except Exception as e:
                    logger.debug(f"Container stats check failed: {e}")
                
                time.sleep(1)  # Check every second
                
                # Refresh container status
                try:
                    container.reload()
                except Exception:
                    break  # Container no longer exists
                    
        except docker.errors.NotFound:
            # Container doesn't exist or has been removed
            pass
        except Exception as e:
            logger.warning(f"Container monitoring failed for {container_name}: {e}")
    
    def execute_in_subprocess(self, func_code: str, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """MVP-1: Keep restricted subprocess fallback for environments without Docker."""
        sanitized_code = self._sanitize_code(func_code)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_file = temp_path / "input.pkl"
            output_file = temp_path / "output.pkl"
            metadata_file = temp_path / "metadata.json"
            error_file = temp_path / "error.json"
            with open(input_file, 'wb') as f:
                pickle.dump(df, f)
            script_content = self._create_restricted_script(sanitized_code)
            script_file = temp_path / "script.py"
            with open(script_file, 'w') as f:
                f.write(script_content)
            start_time = time.time()
            result = subprocess.run([
                'python', str(script_file),
                str(input_file), str(output_file), str(metadata_file), str(error_file)
            ], timeout=self.max_time_seconds + 5, capture_output=True, text=True)
            execution_duration = time.time() - start_time
            if result.returncode != 0:
                if error_file.exists():
                    with open(error_file, 'r') as f:
                        error_info = json.load(f)
                    error_type = error_info.get('error_type', 'UnknownError')
                    error_message = error_info.get('error_message', 'Unknown error')
                    if error_type == 'TimeoutError':
                        raise TimeoutError(f"Function execution timed out after {self.max_time_seconds} seconds")
                    elif error_type == 'MemoryError':
                        raise MemoryError(f"Function exceeded memory limit of {self.max_memory_mb}MB")
                    else:
                        raise SandboxExecutionError(f"Function execution failed: {error_message}")
                else:
                    raise SandboxExecutionError(f"Subprocess execution failed with return code {result.returncode}")
            if not output_file.exists():
                raise SandboxExecutionError("No output file generated")
            with open(output_file, 'rb') as f:
                result_df = pickle.load(f)
            metadata = {}
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            metadata.update({
                "execution_method": "subprocess",
                "subprocess_duration": execution_duration,
                "memory_limit_mb": self.max_memory_mb,
                "time_limit_seconds": self.max_time_seconds
            })
            return result_df, metadata
    
    def execute_function(self, func: Callable, df: pd.DataFrame) -> pd.DataFrame:
        """Execute function with safety checks and resource limits (legacy method)."""
        try:
            with execution_timeout(self.max_time_seconds):
                result_df = func(df.copy())
        except TimeoutError:
            raise TimeoutError(f"Custom function exceeded time limit of {self.max_time_seconds} seconds")
        except Exception as e:
            raise TransformationError(f"Custom function failed: {str(e)}")
        
        if not isinstance(result_df, pd.DataFrame):
            raise TransformationError("Custom function must return a pandas DataFrame")
        
        return result_df
    
    def execute_user_code(self, func_code: str, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute user code string with enterprise security measures."""
        logger.info(f"Executing user code with security isolation (Docker: {self.use_docker and self.docker_available})")
        
        try:
            if self.use_docker and self.docker_available:
                return self.execute_in_docker(func_code, df)
            else:
                return self.execute_in_subprocess(func_code, df)
        except Exception as e:
            logger.error(f"User code execution failed: {str(e)}")
            raise

# Initialize safe executor
safe_executor = SafeFunctionExecutor()

def validate_datetime_column(df: pd.DataFrame, column: str) -> bool:
    """
    Validate if a column can be converted to datetime.
    
    Args:
        df: Input DataFrame
        column: Column name to validate
        
    Returns:
        bool: True if column is valid datetime or can be converted
    """
    try:
        _validate_dataframe(df)
        _validate_column_exists(df, column)
        
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            return True
        
        # Try to convert a sample
        sample_size = min(SAMPLE_SIZE_FOR_VALIDATION, len(df))
        test_series = pd.to_datetime(df[column].head(sample_size), errors='coerce')
        return not test_series.isna().all()
    except Exception:
        return False

def validate_text_column(df: pd.DataFrame, column: str) -> bool:
    """
    Validate if a column contains text data suitable for processing.
    
    Args:
        df: Input DataFrame
        column: Column name to validate
        
    Returns:
        bool: True if column contains valid text data
    """
    try:
        _validate_dataframe(df)
        _validate_column_exists(df, column)
        
        # Check if column is string-like
        if not pd.api.types.is_string_dtype(df[column]) and not pd.api.types.is_object_dtype(df[column]):
            return False
        
        # Check if column has meaningful text content
        sample_size = min(SAMPLE_SIZE_FOR_VALIDATION, len(df))
        sample_text = df[column].dropna().astype(str).head(sample_size)
        if sample_text.empty:
            return False
        
        # Check average length to ensure it's not just single characters or numbers
        avg_length = sample_text.str.len().mean()
        return avg_length > 2
    except Exception:
        return False

def extract_date_components(
    df: pd.DataFrame,
    column: str,
    components: List[str],
    random_state: Optional[int] = None,
    preview_mode: bool = False
) -> TransformationResult:
    """
    Extract specified components from a datetime column.
    
    Args:
        df: Input DataFrame
        column: Name of the datetime column
        components: List of components to extract (year, month, day, hour, minute, second, weekday)
        random_state: Random state for reproducibility (unused but kept for API consistency)
        
    Returns:
        Tuple of (updated_df, transformation_metadata)
    """
    with transformation_timer("extract_date_components"):
        _validate_dataframe(df)
        _validate_column_exists(df, column)
        start_time = time.time()
        
        # Validate components
        valid_components = {"year", "month", "day", "hour", "minute", "second", "weekday"}
        invalid_components = set(components) - valid_components
        if invalid_components:
            raise TransformationError(f"Invalid components: {invalid_components}. Valid options: {valid_components}")
        
        df_copy = df if preview_mode else df.copy()
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df_copy[column]):
            df_copy[column] = pd.to_datetime(df_copy[column], errors='coerce')
            if df_copy[column].isna().all():
                raise TransformationError(f"Column '{column}' cannot be converted to datetime")
        
        added_columns = []
        
        for component in components:
            new_col_name = f"{column}_{component}"
            if component == "year":
                df_copy[new_col_name] = df_copy[column].dt.year
            elif component == "month":
                df_copy[new_col_name] = df_copy[column].dt.month
            elif component == "day":
                df_copy[new_col_name] = df_copy[column].dt.day
            elif component == "hour":
                df_copy[new_col_name] = df_copy[column].dt.hour
            elif component == "minute":
                df_copy[new_col_name] = df_copy[column].dt.minute
            elif component == "second":
                df_copy[new_col_name] = df_copy[column].dt.second
            elif component == "weekday":
                df_copy[new_col_name] = df_copy[column].dt.dayofweek
            
            added_columns.append(new_col_name)
        
        duration_ms = (time.time() - start_time) * 1000.0
        metadata = {
            "status": "success",
            "method": "extract_date_components",
            "duration_ms": duration_ms,
            "rows_affected": len(df_copy),
            "columns_affected": [column] + added_columns
        }
        
        # Add to pipeline configuration
        transformation_registry.add_to_pipeline({
            "function": "extract_date_components",
            "args": {
                "column": column,
                "components": components,
                "random_state": random_state
            }
        })
        
        transformation_registry.add_operation(metadata)
        logger.info(f"Successfully extracted date components from column '{column}': {components}")
        return df_copy, metadata

def encode_categorical(
    df: pd.DataFrame,
    column: str,
    method: str,
    custom_mapping: Optional[Dict[str, Any]] = None,
    random_state: Optional[int] = None,
    preview_mode: bool = False
) -> TransformationResult:
    """
    Encode categorical features using various methods with scalability improvements.
    
    Args:
        df: Input DataFrame
        column: Column to encode
        method: Encoding method (onehot, label, frequency, custom)
        custom_mapping: Custom mapping dictionary for custom encoding
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (updated_df, transformation_metadata)
    """
    with transformation_timer("encode_categorical"):
        _validate_dataframe(df)
        _validate_column_exists(df, column)
        start_time = time.time()
        
        valid_methods = {"onehot", "label", "frequency", "custom"}
        if method not in valid_methods:
            raise TransformationError(f"Invalid encoding method '{method}'. Valid options: {valid_methods}")
        
        if method == "custom" and not custom_mapping:
            raise TransformationError("Custom mapping required for custom encoding method")
        
        df_copy = df if preview_mode else df.copy()
        added_columns = []
        dropped_columns = []
        transformed_columns = []
        
        # Check for high cardinality
        unique_count = df_copy[column].nunique()
        is_large = _is_large_dataset(df_copy)
        
        if method == "onehot":
            # Warn about high cardinality
            if unique_count > 100:
                logger.warning(f"One-hot encoding column '{column}' with {unique_count} unique values may create memory issues")
            
            # Handle missing values
            if df_copy[column].isna().any():
                df_copy[column] = df_copy[column].fillna('missing')
            
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded = encoder.fit_transform(df_copy[[column]])
            
            # Create feature names
            feature_names = [f"{column}_{cat}" for cat in encoder.categories_[0]]
            encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df_copy.index)
            
            df_copy = pd.concat([df_copy.drop(column, axis=1), encoded_df], axis=1)
            added_columns = feature_names
            dropped_columns = [column]
            
        elif method == "label":
            encoder = LabelEncoder()
            # Handle missing values
            non_null_mask = df_copy[column].notna()
            if non_null_mask.any():
                df_copy.loc[non_null_mask, column] = encoder.fit_transform(df_copy.loc[non_null_mask, column])
            transformed_columns = [column]
            
        elif method == "frequency":
            # Use original series to compute map to avoid mutation effects
            freq_map = df[column].value_counts(normalize=True).to_dict()
            df_copy[column] = df_copy[column].map(freq_map)
            transformed_columns = [column]
            
        elif method == "custom":
            df_copy[column] = df_copy[column].map(custom_mapping)
            # Check for unmapped values
            unmapped_count = df_copy[column].isna().sum() - df[column].isna().sum()
            if unmapped_count > 0:
                logger.warning(f"Custom encoding left {unmapped_count} values unmapped in column '{column}'")
            transformed_columns = [column]
        
        affected_columns = transformed_columns + added_columns
        message = f"Applied {method} encoding to '{column}'"
        if added_columns:
            message += f", created {len(added_columns)} new columns"
        
        duration_ms = (time.time() - start_time) * 1000.0
        rows_affected = len(df_copy)
        columns_affected = affected_columns
        metadata = {
            "status": "success",
            "method": f"{method}_encoding",
            "duration_ms": duration_ms,
            "rows_affected": rows_affected,
            "columns_affected": columns_affected,
                "unique_count": unique_count,
                "is_large_dataset": is_large
            }
        
        # Add to pipeline configuration
        transformation_registry.add_to_pipeline({
            "function": "encode_categorical",
            "args": {
                "column": column,
                "method": method,
                "custom_mapping": custom_mapping,
                "random_state": random_state
            }
        })
        
        transformation_registry.add_operation(metadata)
        logger.info(f"Successfully applied {method} encoding to column '{column}'")
        return df_copy, metadata

def scale_numeric(
    df: pd.DataFrame,
    column: str,
    method: str,
    custom_range: Tuple[float, float] = (0, 1),
    random_state: Optional[int] = None,
    preview_mode: bool = False
) -> TransformationResult:
    """
    Scale numerical features using various methods.
    
    Args:
        df: Input DataFrame
        column: Column to scale
        method: Scaling method (minmax, standard, log, custom)
        custom_range: Custom range for minmax scaling
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (updated_df, transformation_metadata)
    """
    with transformation_timer("scale_numeric"):
        _validate_dataframe(df)
        _validate_column_exists(df, column)
        start_time = time.time()
        
        valid_methods = {"minmax", "standard", "log", "custom"}
        if method not in valid_methods:
            raise TransformationError(f"Invalid scaling method '{method}'. Valid options: {valid_methods}")
        
        # Check if column is numeric
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise TransformationError(f"Column '{column}' is not numeric")
        
        df_copy = df if preview_mode else df.copy()
        
        if method == "minmax":
            scaler = MinMaxScaler(feature_range=custom_range)
            df_copy[column] = scaler.fit_transform(df_copy[[column]]).flatten()
            
        elif method == "standard":
            scaler = StandardScaler()
            df_copy[column] = scaler.fit_transform(df_copy[[column]]).flatten()
            
        elif method == "log":
            if (df_copy[column] <= 0).any():
                raise TransformationError("Log scaling requires all values to be positive")
            df_copy[column] = np.log1p(df_copy[column])
            
        elif method == "custom":
            min_val, max_val = custom_range
            col_min, col_max = df_copy[column].min(), df_copy[column].max()
            if col_max == col_min:
                raise TransformationError("Cannot scale column with constant values")
            df_copy[column] = min_val + (df_copy[column] - col_min) * (max_val - min_val) / (col_max - col_min)
        
        message = f"Applied {method} scaling to '{column}'"
        if method in ["minmax", "custom"]:
            message += f" with range {custom_range}"
        
        duration_ms = (time.time() - start_time) * 1000.0
        metadata = {
            "status": "success",
            "method": f"{method}_scaling",
            "duration_ms": duration_ms,
            "rows_affected": len(df_copy),
            "columns_affected": [column]
        }
        
        # Add to pipeline configuration
        transformation_registry.add_to_pipeline({
            "function": "scale_numeric",
            "args": {
                "column": column,
                "method": method,
                "custom_range": custom_range,
                "random_state": random_state
            }
        })
        
        transformation_registry.add_operation(metadata)
        logger.info(f"Successfully applied {method} scaling to column '{column}'")
        return df_copy, metadata

def tokenize_text(
    df: pd.DataFrame,
    column: str,
    method: str = "stemming",
    random_state: Optional[int] = None,
    preview_mode: bool = False
) -> TransformationResult:
    """
    Process text features via tokenization with fallback mechanisms.
    
    Args:
        df: Input DataFrame
        column: Text column to process
        method: Text processing method (stemming, lemmatization, none)
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (updated_df, transformation_metadata)
    """
    with transformation_timer("tokenize_text"):
        _validate_dataframe(df)
        _validate_column_exists(df, column)
        start_time = time.time()
        
        valid_methods = {"stemming", "lemmatization", "none"}
        if method not in valid_methods:
            raise TransformationError(f"Invalid text processing method '{method}'. Valid options: {valid_methods}")
        
        df_copy = df if preview_mode else df.copy()
        processed_texts = []
        
        # Try to ensure NLTK resources with fallbacks
        punkt_available = False
        stopwords_available = False
        wordnet_available = False
        
        if NLTK_AVAILABLE:
            punkt_available = nltk_manager.ensure_resource('tokenizers/punkt')
            stopwords_available = nltk_manager.ensure_resource('corpora/stopwords')
            
            if method == "lemmatization":
                wordnet_available = nltk_manager.ensure_resource('corpora/wordnet')
            else:
                wordnet_available = True
        
        # Get stopwords with fallback
        if stopwords_available and NLTK_AVAILABLE:
            try:
                stop_words = set(stopwords.words("english"))
            except Exception:
                logger.warning("Failed to load NLTK stopwords, using fallback")
                stop_words = nltk_manager.get_fallback_stopwords()
        else:
            stop_words = nltk_manager.get_fallback_stopwords()
        
        # Initialize processors
        stemmer = None
        lemmatizer = None
        
        if NLTK_AVAILABLE:
            if method == "stemming":
                stemmer = PorterStemmer()
            elif method == "lemmatization" and wordnet_available:
                try:
                    lemmatizer = WordNetLemmatizer()
                except Exception:
                    logger.warning("Failed to initialize lemmatizer, falling back to stemming")
                    stemmer = PorterStemmer()
                    method = "stemming"
        
        for text in df_copy[column].astype(str):
            if pd.isna(text) or text.lower() in ['nan', 'none', '']:
                processed_texts.append("")
                continue
            
            # Tokenize with fallback
            if punkt_available and NLTK_AVAILABLE:
                try:
                    tokens = word_tokenize(text.lower())
                except Exception:
                    logger.warning("NLTK tokenization failed, using simple tokenizer")
                    tokens = nltk_manager.simple_tokenize(text)
            else:
                tokens = nltk_manager.simple_tokenize(text)
            
            # Clean tokens
            tokens = [word for word in tokens if word.isalpha() and word not in stop_words and len(word) > 1]
            
            # Apply stemming or lemmatization
            if method == "stemming" and stemmer:
                tokens = [stemmer.stem(word) for word in tokens]
            elif method == "lemmatization" and lemmatizer:
                try:
                    tokens = [lemmatizer.lemmatize(word) for word in tokens]
                except Exception:
                    logger.warning("Lemmatization failed for some tokens, using stemming fallback")
                    if stemmer:
                        tokens = [stemmer.stem(word) for word in tokens]
            
            processed_texts.append(" ".join(tokens))
        
        df_copy[column] = processed_texts
        
        duration_ms = (time.time() - start_time) * 1000.0
        metadata = {
            "status": "success",
            "method": f"text_tokenization_{method}",
            "duration_ms": duration_ms,
            "rows_affected": len(df_copy),
            "columns_affected": [column],
                "nltk_resources_available": {
                    "punkt": punkt_available,
                    "stopwords": stopwords_available,
                    "wordnet": wordnet_available if method == "lemmatization" else None
                }
            }
        
        # Add to pipeline configuration
        transformation_registry.add_to_pipeline({
            "function": "tokenize_text",
            "args": {
                "column": column,
                "method": method,
                "random_state": random_state
            }
        })
        
        transformation_registry.add_operation(metadata)
        logger.info(f"Successfully applied {method} tokenization to column '{column}'")
        return df_copy, metadata

def apply_tfidf_vectorization(
    df: pd.DataFrame,
    column: str,
    max_features: int = 1000,
    use_hashing: bool = None,
    random_state: Optional[int] = None,
    preview_mode: bool = False
) -> TransformationResult:
    """
    Apply TF-IDF vectorization with scalability improvements for large datasets.
    
    Args:
        df: Input DataFrame
        column: Text column to vectorize
        max_features: Maximum number of features for the TF-IDF matrix
        use_hashing: Force use of HashingVectorizer for large datasets (auto-detected if None)
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (updated_df, transformation_metadata)
    """
    with transformation_timer("apply_tfidf_vectorization"):
        _validate_dataframe(df)
        _validate_column_exists(df, column)
        start_time = time.time()
        
        if max_features <= 0:
            raise TransformationError("max_features must be positive")
        
        df_copy = df if preview_mode else df.copy()
        is_large = _is_large_dataset(df_copy)
        
        # Auto-detect if we should use hashing vectorizer
        if use_hashing is None:
            use_hashing = is_large
        
        # Prepare text data
        text_data = df_copy[column].astype(str).fillna("")
        
        if text_data.str.strip().eq("").all():
            raise TransformationError(f"Column '{column}' contains no meaningful text data")
        
        # Choose vectorizer based on dataset size and settings
        if use_hashing:
            logger.info(f"Using HashingVectorizer for large dataset ({len(df_copy)} rows)")
            vectorizer = HashingVectorizer(
                n_features=max_features,
                stop_words='english',
                lowercase=True,
                token_pattern=r'\b[a-zA-Z]{2,}\b'
            )
            # HashingVectorizer doesn't have get_feature_names_out, so create generic names
            feature_names = [f"hash_feature_{i}" for i in range(max_features)]
        else:
            vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                lowercase=True,
                token_pattern=r'\b[a-zA-Z]{2,}\b',
                random_state=random_state
            )
        
        # Apply vectorization
        tfidf_matrix = vectorizer.fit_transform(text_data)
        
        # Create feature names
        if not use_hashing:
            feature_names = [f"tfidf_{feature}" for feature in vectorizer.get_feature_names_out()]
        
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=feature_names,
            index=df_copy.index
        )
        
        # For large datasets, consider sparse representation
        if is_large and tfidf_matrix.nnz / tfidf_matrix.size < 0.1:  # Less than 10% non-zero
            logger.info("Dataset is large and sparse, consider using sparse matrix operations")
        
        # Combine with original DataFrame (drop original text column)
        df_result = pd.concat([df_copy.drop(column, axis=1), tfidf_df], axis=1)
        
        additional_info = {
            "max_features": max_features,
            "use_hashing": use_hashing,
            "is_large_dataset": is_large,
            "sparsity": float(1 - (tfidf_matrix.nnz / tfidf_matrix.size))
        }
        
        if not use_hashing:
            additional_info["vocabulary_size"] = len(vectorizer.vocabulary_)
        
        duration_ms = (time.time() - start_time) * 1000.0
        metadata = {
            "status": "success",
            "method": "tfidf_vectorization",
            "duration_ms": duration_ms,
            "rows_affected": len(df_result),
            "columns_affected": feature_names,
            **additional_info
        }
        
        # Add to pipeline configuration
        transformation_registry.add_to_pipeline({
            "function": "apply_tfidf_vectorization",
            "args": {
                "column": column,
                "max_features": max_features,
                "use_hashing": use_hashing,
                "random_state": random_state
            }
        })
        
        transformation_registry.add_operation(metadata)
        logger.info(f"Successfully applied {'Hashing' if use_hashing else 'TF-IDF'} vectorization to column '{column}', created {len(feature_names)} features")
        return df_result, metadata

def apply_pca(
    df: pd.DataFrame,
    n_components: int,
    columns: Optional[List[str]] = None,
    use_incremental: bool = None,
    random_state: Optional[int] = None,
    preview_mode: bool = False
) -> TransformationResult:
    """
    Apply PCA with incremental processing for large datasets.
    
    Args:
        df: Input DataFrame
        n_components: Number of principal components to keep
        columns: Specific columns to apply PCA to (if None, uses all numeric columns)
        use_incremental: Force use of IncrementalPCA (auto-detected if None)
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (updated_df, transformation_metadata)
    """
    with transformation_timer("apply_pca"):
        _validate_dataframe(df)
        start_time = time.time()
        
        if n_components <= 0:
            raise TransformationError("n_components must be positive")
        
        df_copy = df if preview_mode else df.copy()
        is_large = _is_large_dataset(df_copy)
        
        # Auto-detect if we should use incremental PCA
        if use_incremental is None:
            use_incremental = is_large
        
        # Select columns for PCA
        if columns is None:
            numeric_columns = df_copy.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_columns:
                raise TransformationError("No numeric columns found for PCA")
            columns = numeric_columns
        else:
            # Validate specified columns exist and are numeric
            for col in columns:
                _validate_column_exists(df_copy, col)
                if not pd.api.types.is_numeric_dtype(df_copy[col]):
                    raise TransformationError(f"Column '{col}' is not numeric")
        
        if n_components > len(columns):
            raise TransformationError(f"n_components ({n_components}) cannot be greater than number of features ({len(columns)})")
        
        # Extract data for PCA
        pca_data = df_copy[columns].fillna(0)  # Handle NaN values
        
        # Choose PCA implementation
        if use_incremental:
            logger.info(f"Using IncrementalPCA for large dataset ({len(df_copy)} rows)")
            pca = IncrementalPCA(n_components=n_components)
            
            # Process in batches
            batch_size = min(1000, len(df_copy) // 10) if len(df_copy) > 10000 else len(df_copy)
            for start in range(0, len(pca_data), batch_size):
                end = min(start + batch_size, len(pca_data))
                batch = pca_data.iloc[start:end]
                pca.partial_fit(batch)
            
            pca_result = pca.transform(pca_data)
        else:
            pca = PCA(n_components=n_components, random_state=random_state)
            pca_result = pca.fit_transform(pca_data)
        
        # Create new column names
        pca_columns = [f"PC{i+1}" for i in range(n_components)]
        pca_df = pd.DataFrame(pca_result, columns=pca_columns, index=df_copy.index)
        
        # Combine with original DataFrame (drop original columns used for PCA)
        df_result = pd.concat([df_copy.drop(columns, axis=1), pca_df], axis=1)
        
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        duration_ms = (time.time() - start_time) * 1000.0
        metadata = {
            "status": "success",
            "method": "pca",
            "duration_ms": duration_ms,
            "rows_affected": len(df_result),
            "columns_affected": pca_columns,
                "explained_variance_ratio": explained_variance_ratio.tolist(),
                "cumulative_variance_explained": cumulative_variance.tolist(),
                "total_variance_explained": float(cumulative_variance[-1]),
                "use_incremental": use_incremental,
                "is_large_dataset": is_large
            }
        
        # Add to pipeline configuration
        transformation_registry.add_to_pipeline({
            "function": "apply_pca",
            "args": {
                "n_components": n_components,
                "columns": columns,
                "use_incremental": use_incremental,
                "random_state": random_state
            }
        })
        
        transformation_registry.add_operation(metadata)
        logger.info(f"Successfully applied {'Incremental' if use_incremental else 'Standard'} PCA to {len(columns)} columns, created {n_components} components")
        return df_result, metadata

def apply_custom_user_function(
    df: pd.DataFrame,
    func: Callable[[pd.DataFrame], pd.DataFrame],
    function_name: str = "custom_function",
    use_sandbox: bool = True,
    random_state: Optional[int] = None,
    preview_mode: bool = False
) -> TransformationResult:
    """
    Apply a custom user-defined function with enhanced security and safety checks.
    
    Args:
        df: Input DataFrame
        func: User-defined function that takes and returns a DataFrame
        function_name: Name of the function for logging purposes
        use_sandbox: Whether to execute function in secure sandbox
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (updated_df, transformation_metadata)
    """
    with transformation_timer(f"custom_function_{function_name}"):
        _validate_dataframe(df)
        start_time = time.time()
        
        if not callable(func):
            raise TransformationError("Provided function must be callable")
        
        # Safety check: try function on a small sample first
        sample_size = min(5, len(df))
        sample_df = (df.head(sample_size).copy() if not preview_mode else df.head(sample_size))
        
        try:
            if use_sandbox:
                # Test in sandbox with small sample
                result_sample = safe_executor.execute_function(func, sample_df)
            else:
                # Direct execution with timeout
                with execution_timeout(MAX_CUSTOM_FUNCTION_TIME):
                    result_sample = func(sample_df)
            
            if not isinstance(result_sample, pd.DataFrame):
                raise TransformationError("Custom function must return a pandas DataFrame")
        except Exception as e:
            raise TransformationError(f"Custom function failed on sample data: {str(e)}")
        
        # Apply function to full dataset
        try:
            if use_sandbox:
                df_result = safe_executor.execute_function(func, df.copy() if not preview_mode else df)
            else:
                with execution_timeout(MAX_CUSTOM_FUNCTION_TIME):
                    df_result = func(df.copy() if not preview_mode else df)
        except TimeoutError:
            raise TransformationError(f"Custom function exceeded time limit of {MAX_CUSTOM_FUNCTION_TIME} seconds")
        except Exception as e:
            raise TransformationError(f"Custom function failed: {str(e)}")
        
        # Analyze changes
        original_columns = set(df.columns)
        result_columns = set(df_result.columns)
        
        added_columns = list(result_columns - original_columns)
        dropped_columns = list(original_columns - result_columns)
        transformed_columns = list(original_columns & result_columns)
        
        duration_ms = (time.time() - start_time) * 1000.0
        metadata = {
            "status": "success",
            "method": f"custom_function_{function_name}",
            "duration_ms": duration_ms,
            "rows_affected": len(df_result),
            "columns_affected": list(result_columns),
                "use_sandbox": use_sandbox,
                "execution_safe": True
            }
        
        # Add to pipeline configuration (Note: cannot serialize arbitrary functions)
        transformation_registry.add_to_pipeline({
            "function": "apply_custom_user_function",
            "args": {
                "function_name": function_name,
                "use_sandbox": use_sandbox,
                "random_state": random_state
            },
            "warning": "Custom functions cannot be fully serialized for reproducibility"
        })
        
        transformation_registry.add_operation(metadata)
        logger.info(f"Successfully applied custom function '{function_name}'")
        return df_result, metadata


def apply_custom_user_code(
    df: pd.DataFrame,
    func_code: str,
    function_name: str = "custom_code",
    use_docker: bool = True,
    max_time_seconds: int = 5,
    max_memory_mb: int = 256,
    random_state: Optional[int] = None,
    preview_mode: bool = False
) -> TransformationResult:
    """
    Apply custom user code string with enterprise security isolation.
    
    Args:
        df: Input DataFrame
        func_code: User-defined code string that defines a 'user_function' function
        function_name: Name of the function for logging purposes
        use_docker: Whether to use Docker isolation (falls back to subprocess if unavailable)
        max_time_seconds: Maximum execution time in seconds
        max_memory_mb: Maximum memory usage in MB
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (updated_df, transformation_metadata)
    """
    with transformation_timer(f"custom_code_{function_name}"):
        _validate_dataframe(df)
        
        if not func_code.strip():
            raise TransformationError("Function code cannot be empty")
        
        # Create secure executor with specified limits
        executor = SafeFunctionExecutor(
            max_time_seconds=max_time_seconds,
            max_memory_mb=max_memory_mb,
            use_docker=use_docker
        )
        
        try:
            # Execute user code with security isolation
            df_result, execution_metadata = executor.execute_user_code(func_code, df.copy())
            
            # Analyze changes
            original_columns = set(df.columns)
            result_columns = set(df_result.columns)
            
            added_columns = list(result_columns - original_columns)
            dropped_columns = list(original_columns - result_columns)
            transformed_columns = list(original_columns & result_columns)
            
            metadata = _create_success_metadata(
                method=f"custom_code_{function_name}",
                message=f"Applied custom code '{function_name}' with security isolation",
                transformed_columns=transformed_columns,
                added_columns=added_columns,
                dropped_columns=dropped_columns,
                affected_columns=list(result_columns),
                random_state=random_state,
                additional_info={
                    "use_docker": use_docker and executor.docker_available,
                    "execution_safe": True,
                    "security_isolation": True,
                    **execution_metadata
                }
            )
            
            # Add to pipeline configuration
            transformation_registry.add_to_pipeline({
                "function": "apply_custom_user_code",
                "args": {
                    "function_name": function_name,
                    "use_docker": use_docker,
                    "max_time_seconds": max_time_seconds,
                    "max_memory_mb": max_memory_mb,
                    "random_state": random_state
                },
                "warning": "Custom code execution with security isolation"
            })
            
            transformation_registry.add_operation(metadata)
            logger.info(f"Successfully applied custom code '{function_name}' with security isolation")
            return df_result, metadata
            
        except (TimeoutError, MemoryError, SandboxError) as e:
            error_metadata = _create_error_metadata(
                f"custom_code_{function_name}",
                f"Custom code execution failed: {str(e)}",
                str(e)
            )
            transformation_registry.add_operation(error_metadata)
            raise TransformationError(f"Custom code execution failed: {str(e)}")
        except Exception as e:
            error_metadata = _create_error_metadata(
                f"custom_code_{function_name}",
                f"Custom code execution failed: {str(e)}",
                str(e)
            )
            transformation_registry.add_operation(error_metadata)
            raise TransformationError(f"Custom code execution failed: {str(e)}")

def get_transformation_summary() -> Dict[str, Any]:
    """
    Get a comprehensive summary of all transformations applied.
    
    Returns:
        Dictionary containing transformation summary with pipeline information
    """
    return transformation_registry.get_summary()

# ==============================
# MVP-1 Core Transformations
# ==============================

def extract_time_deltas(
    df: pd.DataFrame,
    date_col: str,
    reference_col: str = 'today',
    units: List[str] = None,
    preview_mode: bool = False
) -> TransformationResult:
    """Compute time deltas between date_col and reference_col or today.
    Returns (df_with_new_cols, metadata).
    """
    units = units or ['days', 'hours', 'minutes']
    start_time = time.time()
    _validate_dataframe(df)
    _validate_column_exists(df, date_col)
    df_copy = df if preview_mode else df.copy()

    base = pd.to_datetime(df[date_col], errors='coerce')
    if reference_col == 'today':
        ref_series = pd.Timestamp.now(tz=None)
        delta = ref_series - base
    else:
        _validate_column_exists(df, reference_col)
        ref_series = pd.to_datetime(df[reference_col], errors='coerce')
        delta = ref_series - base

    added_columns = []
    if 'days' in units:
        col = f"{date_col}_delta_days"
        df_copy[col] = delta.dt.days
        added_columns.append(col)
    if 'hours' in units:
        col = f"{date_col}_delta_hours"
        df_copy[col] = delta.dt.total_seconds() / 3600.0
        added_columns.append(col)
    if 'minutes' in units:
        col = f"{date_col}_delta_minutes"
        df_copy[col] = delta.dt.total_seconds() / 60.0
        added_columns.append(col)
    if 'seconds' in units:
        col = f"{date_col}_delta_seconds"
        df_copy[col] = delta.dt.total_seconds()
        added_columns.append(col)

    duration_ms = (time.time() - start_time) * 1000.0
    metadata = {
        "status": "success",
        "method": "extract_time_deltas",
        "duration_ms": duration_ms,
        "rows_affected": len(df_copy),
        "columns_affected": added_columns
    }
    transformation_registry.add_to_pipeline({
        "function": "extract_time_deltas",
        "args": {"date_col": date_col, "reference_col": reference_col, "units": units}
    })
    transformation_registry.add_operation(metadata)
    return df_copy, metadata

def apply_frequency_encoding(
    df: pd.DataFrame,
    column: str,
    preview_mode: bool = False
) -> TransformationResult:
    """Apply frequency encoding to a categorical column in-place when preview_mode, else on a copy."""
    start_time = time.time()
    _validate_dataframe(df)
    _validate_column_exists(df, column)
    df_copy = df if preview_mode else df.copy()
    freq_map = df[column].value_counts(normalize=True).to_dict()
    df_copy[column] = df_copy[column].map(freq_map)
    duration_ms = (time.time() - start_time) * 1000.0
    metadata = {
        "status": "success",
        "method": "frequency_encoding",
        "duration_ms": duration_ms,
        "rows_affected": len(df_copy),
        "columns_affected": [column]
    }
    transformation_registry.add_to_pipeline({
        "function": "apply_frequency_encoding",
        "args": {"column": column}
    })
    transformation_registry.add_operation(metadata)
    return df_copy, metadata

def clear_transformation_history() -> None:
    """Clear the transformation history and pipeline."""
    transformation_registry.clear()
    logger.info("Transformation history and pipeline cleared")


class TransformationPipeline:
    """MVP-1: Simple transformation pipeline executor with preview and hashing."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.steps: List[Dict[str, Any]] = config.get('steps', [])
        self.continue_on_error: bool = config.get('continue_on_error', False)

    def _hash_df(self, df: pd.DataFrame) -> str:
        try:
            head = df.head(100)
            payload = head.to_csv(index=False).encode()
            return hashlib.sha256(payload).hexdigest()[:16]
        except Exception:
            return ""

    def preview(self, df: pd.DataFrame, sample_size: int = 100) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        sample_df = df.head(sample_size)
        return self._apply(sample_df, preview_mode=True)

    def apply(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        return self._apply(df, preview_mode=False)

    def _apply(self, df: pd.DataFrame, preview_mode: bool) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        func_map = {
            "extract_date_components": extract_date_components,
            "extract_time_deltas": extract_time_deltas,
            "encode_categorical": encode_categorical,
            "apply_frequency_encoding": apply_frequency_encoding,
            "scale_numeric": scale_numeric,
            "tokenize_text": tokenize_text,
            "apply_tfidf_vectorization": apply_tfidf_vectorization,
            "apply_pca": apply_pca,
            "apply_custom_user_code": apply_custom_user_code
        }
        current = df
        step_meta: List[Dict[str, Any]] = []
        dataset_hashes: Dict[str, str] = {}
        for idx, step in enumerate(self.steps):
            name = step.get('action') or step.get('function')
            params = step.get('params') or step.get('args', {})
            if not name or name not in func_map:
                if not self.continue_on_error:
                    break
                continue
            params = {**params, 'preview_mode': preview_mode}
            before_hash = self._hash_df(current)
            try:
                current, meta = func_map[name](current, **params)
                after_hash = self._hash_df(current)
                dataset_hashes[name or f"step_{idx}"] = after_hash
                step_meta.append(meta)
            except Exception as e:
                step_meta.append({"status": "error", "method": name, "error": str(e)})
                if not self.continue_on_error:
                    break
        pipeline_metadata = {
            "status": "success",
            "method": "transformation_pipeline",
            "duration_ms": None,
            "rows_affected": len(current),
            "columns_affected": list(current.columns),
            "step_count": len(step_meta)
        }
        return current, {
            "metadata": pipeline_metadata,
            "step_metadata": step_meta,
            "dataset_hashes": dataset_hashes
        }

def export_transformation_pipeline(filepath: str) -> None:
    """Export the transformation pipeline for reproducibility."""
    transformation_registry.export_pipeline(filepath)

def import_transformation_pipeline(filepath: str) -> None:
    """Import a transformation pipeline for reproducibility."""
    transformation_registry.import_pipeline(filepath)

def apply_multiple_transformations(
    df: pd.DataFrame,
    transformations: List[Dict[str, Any]],
    continue_on_error: bool = True
) -> Tuple[pd.DataFrame, List[TransformationMetadata]]:
    """
    Apply multiple transformations in sequence with enhanced error handling.
    
    Args:
        df: Input DataFrame
        transformations: List of transformation configurations
        continue_on_error: Whether to continue processing after errors
        
    Returns:
        Tuple of (final_df, list_of_metadata)
    """
    with transformation_timer("batch_transformations"):
        _validate_dataframe(df)
        
        current_df = df.copy()
        all_metadata = []
        
        # Map function names to actual functions
        func_mapping = {
            "extract_date_components": extract_date_components,
            "extract_time_deltas": extract_time_deltas,
            "encode_categorical": encode_categorical,
            "apply_frequency_encoding": apply_frequency_encoding,
            "scale_numeric": scale_numeric,
            "tokenize_text": tokenize_text,
            "apply_tfidf_vectorization": apply_tfidf_vectorization,
            "apply_pca": apply_pca,
            "apply_custom_user_code": apply_custom_user_code
        }
        
        successful_transformations = 0
        
        for i, transform_config in enumerate(transformations):
            # Normalize configuration format
            func_name = transform_config.get("action") or transform_config.get("function")
            params = transform_config.get("params") or transform_config.get("args", {})
            
            if not func_name:
                error_metadata = _create_error_metadata(
                    "batch_transformation",
                    f"No action/function specified in transformation {i+1}"
                )
                all_metadata.append(error_metadata)
                logger.error(f"No action/function specified in transformation {i+1}")
                if not continue_on_error:
                    break
                continue
            
            if func_name not in func_mapping:
                error_metadata = _create_error_metadata(
                    func_name,
                    f"Unknown transformation action: {func_name}"
                )
                all_metadata.append(error_metadata)
                logger.error(f"Unknown transformation action: {func_name}")
                if not continue_on_error:
                    break
                continue
            
            func = func_mapping[func_name]
            
            try:
                logger.debug(f"Applying transformation {i+1}/{len(transformations)}: {func_name}")
                # Ensure preview_mode is respected if passed
                current_df, metadata = func(current_df, **params)
                all_metadata.append(metadata)
                successful_transformations += 1
                logger.info(f"Successfully completed transformation {i+1}: {func_name}")
            except Exception as e:
                error_metadata = _create_error_metadata(
                    func_name,
                    f"Transformation failed: {str(e)}",
                    str(e)
                )
                all_metadata.append(error_metadata)
                logger.error(f"Batch transformation failed at {func_name}: {e}")
                
                if not continue_on_error:
                    logger.error("Stopping batch transformation due to error")
                    break
                # Continue with remaining transformations
        
        logger.info(f"Batch transformation completed. Applied {len(transformations)} transformations, "
                    f"{successful_transformations} successful, {len(transformations) - successful_transformations} failed")
        
        return current_df, all_metadata

# Utility functions for monitoring and diagnostics
def get_system_info() -> Dict[str, Any]:
    """Get system information for debugging and optimization."""
    import platform
    
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "dask_available": DASK_AVAILABLE,
        "metrics_available": METRICS_AVAILABLE,
        "nltk_available": NLTK_AVAILABLE,
        "large_dataset_threshold": LARGE_DATASET_THRESHOLD
    }
    
    if PSUTIL_AVAILABLE:
        import psutil
        info.update({
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2)
        })
    
    return info

def optimize_for_dataset_size(df: pd.DataFrame) -> Dict[str, Any]:
    """Provide optimization recommendations based on dataset characteristics."""
    size_mb = df.memory_usage(deep=True).sum() / (1024**2)
    row_count = len(df)
    col_count = len(df.columns)
    
    recommendations = []
    
    if row_count > LARGE_DATASET_THRESHOLD:
        recommendations.append("Consider using incremental processing for PCA and streaming for TF-IDF")
    
    if size_mb > 500:  # More than 500MB
        recommendations.append("Dataset is large, consider using Dask for distributed processing")
    
    # Check for high cardinality categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio > 0.5:
            recommendations.append(f"Column '{col}' has high cardinality ({unique_ratio:.2%}), consider alternative encoding methods")
    
    return {
        "dataset_size_mb": round(size_mb, 2),
        "row_count": row_count,
        "column_count": col_count,
        "is_large": row_count > LARGE_DATASET_THRESHOLD,
        "recommendations": recommendations
    }

# Configure NLTK data path on module import
def setup_nltk_environment():
    """Setup NLTK environment with proper data paths."""
    if not NLTK_AVAILABLE:
        logger.warning("NLTK not available, text processing will use fallback methods")
        return
        
    try:
        # Try to download essential resources during deployment
        essential_resources = [
            ('tokenizers/punkt', 'punkt'),
            ('corpora/stopwords', 'stopwords'),
            ('corpora/wordnet', 'wordnet')
        ]
        
        for resource, download_name in essential_resources:
            success = nltk_manager.ensure_resource(resource, download_name)
            if success:
                logger.info(f"NLTK resource '{download_name}' is available")
            else:
                logger.warning(f"NLTK resource '{download_name}' not available, will use fallbacks")
    except Exception as e:
        logger.warning(f"Error setting up NLTK environment: {e}")

# Initialize on import
setup_nltk_environment()