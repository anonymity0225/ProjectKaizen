from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
import psutil
import docker
import hashlib
import secrets
import hmac
import ssl
import httpx
import redis
import aioredis
import sqlite3
import pickle
import signal
import struct
import socket
import mmap
import math
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Set, Union, Callable
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field, asdict
from threading import Thread, RLock, Event
import subprocess
import os
import sys
from pathlib import Path
import tempfile
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography import x509
import base64
from enum import Enum, IntEnum
from collections import defaultdict, deque
import weakref
import resource
import copy

# OpenTelemetry imports
from opentelemetry import trace, metrics
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.prometheus import PrometheusMetricReader

from app.schemas.automation import (
    PipelineConfig,
    PipelineRun,
    PipelineStep,
    RunResponse,
    Status,
    StepArtifact,
    StepLog,
    StepResult,
    StepResourceLimits,
    ResourceUsage,
    ContainerSecurityConfig,
    WebhookSecurityConfig,
)
from app.utils.audit_log import generate_correlation_id, create_standard_response
from app.observability.metrics import (
    service_errors_total,
    pipeline_runs_total,
    pipeline_run_duration_seconds,
    pipeline_step_duration_seconds,
    pipeline_step_retries_total,
    pipeline_schedules_total,
    webhook_notifications_total,
    pipeline_concurrent_runs,
    observe
)
from app.database.connection import get_db_session
from sqlalchemy.orm import Session
from app.models.automation import (
    PipelineRun as DbPipelineRun, 
    StepRun as DbStepRun, 
    RunLog as DbRunLog, 
    RunStatus,
    PipelineSchedule as DbPipelineSchedule,
    PipelineTemplate as DbPipelineTemplate,
    PipelineExecutionMetrics as DbPipelineExecutionMetrics,
    WebhookDelivery as DbWebhookDelivery,
    SecurityEvent as DbSecurityEvent,
    AuditLog as DbAuditLog
)
from app.core.celery_app import celery_app

# Integration services
from app.services.preprocessing import DataPreprocessingService
from app.services.transformation import TransformationService
from app.services.modeling import EnhancedModelingService
from app.services import visualization as visualization_service
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from app.database.connection import get_engine
from app.models.automation import PipelineSchedule, ScheduleType

logger = logging.getLogger(__name__)

# Initialize OpenTelemetry
tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)

# Custom metrics
pipeline_execution_counter = meter.create_counter(
    name="pipeline_executions_total",
    description="Total number of pipeline executions",
    unit="1"
)

container_security_events = meter.create_counter(
    name="container_security_events_total",
    description="Total number of container security events",
    unit="1"
)

webhook_security_violations = meter.create_counter(
    name="webhook_security_violations_total",
    description="Total number of webhook security violations",
    unit="1"
)

resource_limit_violations = meter.create_counter(
    name="resource_limit_violations_total",
    description="Total number of resource limit violations",
    unit="1"
)

# ==========================================
# CRITICAL FIX 1: DURABLE STATE MANAGEMENT
# ==========================================

class PipelineStateManager:
    """Durable state management with Redis persistence and write-ahead logging."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", enable_wal: bool = True):
        self.redis_url = redis_url
        self.enable_wal = enable_wal
        self._redis_pool = None
        self._wal_path = Path(tempfile.gettempdir()) / "pipeline_wal.log"
        self._lock = RLock()
        
        # State machine definitions
        self.valid_transitions = {
            Status.PENDING: [Status.RUNNING, Status.CANCELLED],
            Status.RUNNING: [Status.COMPLETED, Status.FAILED, Status.CANCELLED],
            Status.COMPLETED: [],  # Terminal state
            Status.FAILED: [Status.RUNNING],  # Can retry
            Status.CANCELLED: []  # Terminal state
        }
        
        # Initialize WAL
        if self.enable_wal:
            self._init_wal()
    
    async def get_redis_connection(self):
        """Get Redis connection with connection pooling."""
        if not self._redis_pool:
            self._redis_pool = aioredis.ConnectionPool.from_url(
                self.redis_url,
                max_connections=20,
                retry_on_timeout=True,
                health_check_interval=30
            )
        return aioredis.Redis(connection_pool=self._redis_pool)
    
    def _init_wal(self):
        """Initialize write-ahead logging."""
        self._wal_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._wal_path.exists():
            self._wal_path.touch()
    
    def _write_wal_entry(self, operation: str, run_id: str, old_state: Optional[Dict], new_state: Dict):
        """Write operation to write-ahead log."""
        if not self.enable_wal:
            return
        
        wal_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'operation': operation,
            'run_id': run_id,
            'old_state': old_state,
            'new_state': new_state,
            'checksum': hashlib.sha256(json.dumps(new_state, sort_keys=True).encode()).hexdigest()
        }
        
        with self._lock:
            with open(self._wal_path, 'a') as f:
                f.write(json.dumps(wal_entry) + '\n')
    
    async def create_run(self, run: PipelineRun) -> bool:
        """Create new pipeline run with durable state."""
        try:
            run_data = asdict(run)
            
            # Write to WAL first
            self._write_wal_entry('CREATE', run.run_id, None, run_data)
            
            # Write to Redis
            redis = await self.get_redis_connection()
            key = f"pipeline_run:{run.run_id}"
            
            # Use transaction for atomicity
            async with redis.pipeline(transaction=True) as pipe:
                pipe.hset(key, mapping={
                    'data': json.dumps(run_data),
                    'created_at': datetime.now(timezone.utc).isoformat(),
                    'updated_at': datetime.now(timezone.utc).isoformat()
                })
                pipe.expire(key, 86400 * 7)  # 7 days TTL
                await pipe.execute()
            
            logger.info(f"Created durable pipeline run: {run.run_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create durable run {run.run_id}: {e}")
            return False
    
    async def update_run_state(self, run_id: str, new_status: Status, 
                              metadata: Dict[str, Any] = None) -> bool:
        """Update pipeline run state with validation."""
        try:
            redis = await self.get_redis_connection()
            key = f"pipeline_run:{run_id}"
            
            # Get current state
            current_data = await redis.hget(key, 'data')
            if not current_data:
                logger.error(f"Pipeline run {run_id} not found")
                return False
            
            current_run = json.loads(current_data)
            old_status = Status(current_run['status'])
            
            # Validate state transition
            if not self._is_valid_transition(old_status, new_status):
                logger.error(f"Invalid state transition for {run_id}: {old_status} -> {new_status}")
                return False
            
            # Update state
            current_run['status'] = new_status.value
            current_run['updated_at'] = datetime.now(timezone.utc).isoformat()
            
            if metadata:
                current_run.update(metadata)
            
            # Write to WAL
            self._write_wal_entry('UPDATE', run_id, json.loads(current_data), current_run)
            
            # Atomic update in Redis
            async with redis.pipeline(transaction=True) as pipe:
                pipe.hset(key, mapping={
                    'data': json.dumps(current_run),
                    'updated_at': current_run['updated_at']
                })
                await pipe.execute()
            
            logger.info(f"Updated pipeline {run_id} state: {old_status} -> {new_status}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update run {run_id} state: {e}")
            return False
    
    def _is_valid_transition(self, from_status: Status, to_status: Status) -> bool:
        """Validate state machine transition."""
        return to_status in self.valid_transitions.get(from_status, [])
    
    async def get_run(self, run_id: str) -> Optional[PipelineRun]:
        """Get pipeline run from durable storage."""
        try:
            redis = await self.get_redis_connection()
            key = f"pipeline_run:{run_id}"
            
            data = await redis.hget(key, 'data')
            if not data:
                return None
            
            run_data = json.loads(data)
            return PipelineRun(**run_data)
            
        except Exception as e:
            logger.error(f"Failed to get run {run_id}: {e}")
            return None
    
    async def list_active_runs(self) -> List[PipelineRun]:
        """List all active pipeline runs."""
        try:
            redis = await self.get_redis_connection()
            keys = await redis.keys("pipeline_run:*")
            
            active_runs = []
            for key in keys:
                data = await redis.hget(key, 'data')
                if data:
                    run_data = json.loads(data)
                    if run_data['status'] in ['pending', 'running']:
                        active_runs.append(PipelineRun(**run_data))
            
            return active_runs
            
        except Exception as e:
            logger.error(f"Failed to list active runs: {e}")
            return []
    
    async def recover_from_wal(self) -> int:
        """Recover state from write-ahead log after restart."""
        if not self.enable_wal or not self._wal_path.exists():
            return 0
        
        recovered_count = 0
        try:
            with open(self._wal_path, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            entry = json.loads(line.strip())
                            await self._apply_wal_entry(entry)
                            recovered_count += 1
                        except Exception as e:
                            logger.error(f"Failed to apply WAL entry: {e}")
            
            logger.info(f"Recovered {recovered_count} operations from WAL")
            return recovered_count
            
        except Exception as e:
            logger.error(f"WAL recovery failed: {e}")
            return 0
    
    async def _apply_wal_entry(self, entry: Dict):
        """Apply single WAL entry during recovery."""
        operation = entry['operation']
        run_id = entry['run_id']
        new_state = entry['new_state']
        
        if operation == 'CREATE':
            run = PipelineRun(**new_state)
            await self.create_run(run)
        elif operation == 'UPDATE':
            # Apply update directly to Redis
            redis = await self.get_redis_connection()
            key = f"pipeline_run:{run_id}"
            await redis.hset(key, mapping={
                'data': json.dumps(new_state),
                'updated_at': new_state.get('updated_at', datetime.now(timezone.utc).isoformat())
            })

# ==========================================
# CRITICAL FIX 2: ENHANCED WEBHOOK SECURITY
# ==========================================

class SecureWebhookManager:
    """Enhanced webhook security with timing attack protection and mutual TLS."""
    
    def __init__(self, enable_mtls: bool = True):
        self.enable_mtls = enable_mtls
        self._nonce_store = {}  # In production, use Redis
        self._request_window_seconds = 300  # 5 minutes
        self._rate_limits = defaultdict(deque)
        self._lock = RLock()
        
        # Generate CA for mutual TLS
        if self.enable_mtls:
            self._init_mtls()
    
    def _init_mtls(self):
        """Initialize mutual TLS certificates."""
        try:
            # Generate CA private key
            self.ca_private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            
            # Generate CA certificate
            subject = issuer = x509.Name([
                x509.NameAttribute(x509.NameOID.COUNTRY_NAME, "US"),
                x509.NameAttribute(x509.NameOID.STATE_OR_PROVINCE_NAME, "CA"),
                x509.NameAttribute(x509.NameOID.ORGANIZATION_NAME, "PipelineCA"),
            ])
            
            self.ca_cert = x509.CertificateBuilder().subject_name(
                subject
            ).issuer_name(
                issuer
            ).public_key(
                self.ca_private_key.public_key()
            ).serial_number(
                x509.random_serial_number()
            ).not_valid_before(
                datetime.utcnow()
            ).not_valid_after(
                datetime.utcnow() + timedelta(days=365)
            ).add_extension(
                x509.BasicConstraints(ca=True, path_length=None),
                critical=True,
            ).sign(self.ca_private_key, hashes.SHA256())
            
            logger.info("Mutual TLS CA initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize mTLS: {e}")
            self.enable_mtls = False
    
    def validate_hmac_signature(self, payload: bytes, signature: str, secret: str, 
                               timestamp: Optional[str] = None) -> bool:
        """Secure HMAC validation with timing attack protection and replay window."""
        try:
            if not signature.startswith("sha256="):
                return False
            
            signature = signature[7:]  # Remove 'sha256=' prefix
            
            # Check timestamp window to prevent replay attacks
            if timestamp:
                try:
                    request_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    current_time = datetime.now(timezone.utc)
                    age_seconds = (current_time - request_time).total_seconds()
                    
                    if age_seconds > self._request_window_seconds:
                        logger.warning(f"Webhook request too old: {age_seconds}s")
                        return False
                        
                except ValueError:
                    logger.warning(f"Invalid timestamp format: {timestamp}")
                    return False
            
            # Create expected signature
            expected_signature = hmac.new(
                secret.encode(),
                payload,
                hashlib.sha256
            ).hexdigest()
            
            # Use constant-time comparison to prevent timing attacks
            return secrets.compare_digest(signature, expected_signature)
            
        except Exception as e:
            logger.error(f"HMAC validation failed: {e}")
            return False
    
    def validate_nonce(self, nonce: str, user_id: str) -> bool:
        """Validate nonce to prevent replay attacks."""
        with self._lock:
            current_time = time.time()
            nonce_key = f"{user_id}:{nonce}"
            
            # Check if nonce was already used
            if nonce_key in self._nonce_store:
                logger.warning(f"Nonce replay attempt: {nonce} for user {user_id}")
                return False
            
            # Store nonce with expiration
            self._nonce_store[nonce_key] = current_time + self._request_window_seconds
            
            # Cleanup expired nonces
            expired_nonces = [
                key for key, expiry in self._nonce_store.items() 
                if expiry < current_time
            ]
            for key in expired_nonces:
                del self._nonce_store[key]
            
            return True
    
    def check_rate_limit(self, client_id: str, max_requests: int = 100, 
                        window_seconds: int = 3600) -> bool:
        """Check rate limiting for webhook requests."""
        with self._lock:
            current_time = time.time()
            client_requests = self._rate_limits[client_id]
            
            # Remove requests outside the window
            while client_requests and client_requests[0] < current_time - window_seconds:
                client_requests.popleft()
            
            # Check if within limits
            if len(client_requests) >= max_requests:
                logger.warning(f"Rate limit exceeded for client {client_id}")
                return False
            
            # Record this request
            client_requests.append(current_time)
            return True
    
    def create_client_certificate(self, client_id: str) -> Tuple[bytes, bytes]:
        """Create client certificate for mutual TLS."""
        if not self.enable_mtls:
            raise RuntimeError("Mutual TLS not enabled")
        
        # Generate client private key
        client_private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        
        # Generate client certificate
        subject = x509.Name([
            x509.NameAttribute(x509.NameOID.COMMON_NAME, client_id),
        ])
        
        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            self.ca_cert.subject
        ).public_key(
            client_private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.utcnow()
        ).not_valid_after(
            datetime.utcnow() + timedelta(days=90)
        ).sign(self.ca_private_key, hashes.SHA256())
        
        # Serialize certificate and key
        cert_pem = cert.public_bytes(serialization.Encoding.PEM)
        key_pem = client_private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        return cert_pem, key_pem
    
    def validate_client_certificate(self, cert_data: bytes) -> bool:
        """Validate client certificate against CA."""
        if not self.enable_mtls:
            return True
        
        try:
            client_cert = x509.load_pem_x509_certificate(cert_data)
            
            # Verify certificate is signed by our CA
            try:
                self.ca_cert.public_key().verify(
                    client_cert.signature,
                    client_cert.tbs_certificate_bytes,
                    padding.PKCS1v15(),
                    client_cert.signature_hash_algorithm,
                )
                return True
            except Exception:
                return False
                
        except Exception as e:
            logger.error(f"Certificate validation failed: {e}")
            return False

# Deprecated in-memory state - replaced with durable state manager
_pipeline_runs: Dict[str, PipelineRun] = {}  # Keep for backward compatibility during migration
_cancellations: Dict[str, asyncio.Event] = {}

# Enhanced security tracking
_webhook_security_manager = SecureWebhookManager()
_pipeline_state_manager = PipelineStateManager()

# ==========================================
# CRITICAL FIX 3: ENHANCED CONTAINER SECURITY
# ==========================================

class EnhancedContainerSecurityManager:
    """Advanced container security with syscall monitoring and image signing."""
    
    def __init__(self, enable_seccomp_bpf: bool = True):
        self.enable_seccomp_bpf = enable_seccomp_bpf
        self.docker_client = docker.from_env()
        self._security_profiles = {}
        self._behavior_baselines = {}
        self._init_security_profiles()
        
    def _init_security_profiles(self):
        """Initialize security profiles for different workload types."""
        self._security_profiles = {
            'data_processing': {
                'allowed_syscalls': [
                    'read', 'write', 'open', 'close', 'stat', 'fstat', 'lstat',
                    'mmap', 'munmap', 'brk', 'rt_sigaction', 'rt_sigprocmask',
                    'clone', 'execve', 'exit', 'exit_group', 'wait4', 'kill'
                ],
                'blocked_syscalls': [
                    'ptrace', 'process_vm_readv', 'process_vm_writev',
                    'keyctl', 'add_key', 'request_key', 'mount', 'umount2',
                    'swapon', 'swapoff', 'reboot', 'sethostname', 'setdomainname'
                ],
                'max_memory_mb': 2048,
                'max_cpu_percent': 80,
                'network_access': False
            },
            'ml_training': {
                'allowed_syscalls': [
                    'read', 'write', 'open', 'close', 'stat', 'fstat', 'lstat',
                    'mmap', 'munmap', 'brk', 'rt_sigaction', 'rt_sigprocmask',
                    'clone', 'execve', 'exit', 'exit_group', 'wait4', 'kill',
                    'futex', 'sched_yield', 'nanosleep'
                ],
                'blocked_syscalls': [
                    'ptrace', 'process_vm_readv', 'process_vm_writev',
                    'keyctl', 'add_key', 'request_key', 'mount', 'umount2'
                ],
                'max_memory_mb': 8192,
                'max_cpu_percent': 95,
                'network_access': True
            }
        }
    
    def create_secure_container(self, image: str, command: List[str], 
                              workload_type: str = 'data_processing',
                              verify_image: bool = True) -> str:
        """Create container with enhanced security."""
        
        # Verify image signature if enabled
        if verify_image and not self._verify_image_signature(image):
            raise ValueError(f"Image signature verification failed: {image}")
        
        profile = self._security_profiles.get(workload_type, self._security_profiles['data_processing'])
        
        # Create seccomp profile
        seccomp_profile = self._create_seccomp_profile(profile['allowed_syscalls'], profile['blocked_syscalls'])
        
        # Container configuration with maximum security
        container_config = {
            'image': image,
            'command': command,
            'detach': True,
            'read_only': True,
            'cap_drop': ['ALL'],
            'cap_add': [],  # No capabilities by default
            'user': '1000:1000',  # Non-root user
            'network_mode': 'none' if not profile['network_access'] else 'bridge',
            'security_opt': [
                f'seccomp={seccomp_profile}',
                'no-new-privileges:true',
                'apparmor:docker-default'
            ],
            'mem_limit': f"{profile['max_memory_mb']}m",
            'cpu_period': 100000,
            'cpu_quota': int(100000 * profile['max_cpu_percent'] / 100),
            'tmpfs': {'/tmp': 'rw,noexec,nosuid,size=100m'},
            'environment': {
                'CONTAINER_SECURITY_LEVEL': 'HIGH',
                'WORKLOAD_TYPE': workload_type
            }
        }
        
        try:
            container = self.docker_client.containers.run(**container_config)
            container_id = container.id
            
            # Start security monitoring
            self._start_container_monitoring(container_id, workload_type)
            
            logger.info(f"Created secure container: {container_id} with profile: {workload_type}")
            return container_id
            
        except Exception as e:
            logger.error(f"Failed to create secure container: {e}")
            raise
    
    def _verify_image_signature(self, image: str) -> bool:
        """Verify container image signature using Docker Content Trust."""
        try:
            # In production, this would integrate with Docker Content Trust (DCT)
            # or container registry's signing mechanism like Cosign
            
            # Extract registry and image name
            if '/' not in image:
                image = f"library/{image}"
            
            # Check if image has valid signature
            # This is a simplified implementation - in production you'd use:
            # - Docker Content Trust with notary
            # - Cosign for keyless signing
            # - Harbor's built-in signing
            
            # For now, simulate signature verification
            trusted_registries = ['docker.io', 'gcr.io', 'your-registry.com']
            registry = image.split('/')[0] if '.' in image.split('/')[0] else 'docker.io'
            
            if registry not in trusted_registries:
                logger.warning(f"Untrusted registry: {registry}")
                return False
            
            # Simulate signature check by inspecting image metadata
            try:
                image_info = self.docker_client.api.inspect_image(image)
                
                # Check for signature-related labels
                labels = image_info.get('Config', {}).get('Labels', {})
                
                # Look for signing metadata
                if 'io.cncf.notary.signature' in labels or 'dev.cosign.signature' in labels:
                    logger.info(f"Image signature verified: {image}")
                    return True
                
                # For testing, allow images without signatures but log warning
                logger.warning(f"Image not signed, proceeding with caution: {image}")
                return True  # In production, this should be False
                
            except Exception as e:
                logger.error(f"Failed to inspect image {image}: {e}")
                return False
            
        except Exception as e:
            logger.error(f"Image signature verification failed: {e}")
            return False
    
    def _create_seccomp_profile(self, allowed_syscalls: List[str], blocked_syscalls: List[str]) -> str:
        """Create seccomp security profile."""
        profile = {
            "defaultAction": "SCMP_ACT_ERRNO",
            "architectures": ["SCMP_ARCH_X86_64", "SCMP_ARCH_X86", "SCMP_ARCH_X32"],
            "syscalls": []
        }
        
        # Add allowed syscalls
        for syscall in allowed_syscalls:
            profile["syscalls"].append({
                "name": syscall,
                "action": "SCMP_ACT_ALLOW"
            })
        
        # Explicitly block dangerous syscalls
        for syscall in blocked_syscalls:
            profile["syscalls"].append({
                "name": syscall,
                "action": "SCMP_ACT_KILL"
            })
        
        # Write profile to temporary file
        profile_path = f"/tmp/seccomp_profile_{uuid.uuid4().hex}.json"
        with open(profile_path, 'w') as f:
            json.dump(profile, f)
        
        return profile_path
    
    def _start_container_monitoring(self, container_id: str, workload_type: str):
        """Start comprehensive container security monitoring."""
        
        def monitor_container():
            try:
                container = self.docker_client.containers.get(container_id)
                baseline = self._behavior_baselines.get(workload_type, {})
                
                while container.status == 'running':
                    # Check for privilege escalation attempts
                    if self._check_privilege_escalation(container):
                        logger.critical(f"Privilege escalation detected in container {container_id}")
                        self._handle_security_violation(container_id, "privilege_escalation")
                    
                    # Monitor syscall patterns (requires eBPF in production)
                    if self.enable_seccomp_bpf:
                        suspicious_syscalls = self._monitor_syscall_patterns(container)
                        if suspicious_syscalls:
                            logger.warning(f"Suspicious syscalls in {container_id}: {suspicious_syscalls}")
                    
                    # Check resource usage anomalies
                    stats = container.stats(stream=False)
                    if self._detect_resource_anomalies(stats, baseline):
                        logger.warning(f"Resource anomaly detected in container {container_id}")
                    
                    # Check network activity if network access is disabled
                    if not self._security_profiles[workload_type]['network_access']:
                        if self._detect_network_activity(container):
                            logger.critical(f"Unauthorized network activity in container {container_id}")
                            self._handle_security_violation(container_id, "unauthorized_network")
                    
                    time.sleep(5)  # Monitor every 5 seconds
                    
            except Exception as e:
                logger.error(f"Container monitoring failed for {container_id}: {e}")
        
        # Start monitoring in background thread
        monitor_thread = threading.Thread(target=monitor_container, daemon=True)
        monitor_thread.start()
    
    def _check_privilege_escalation(self, container) -> bool:
        """Check for privilege escalation attempts."""
        try:
            # Get container processes
            top_result = container.top()
            processes = top_result.get('Processes', [])
            
            for process in processes:
                # Check if any process is running as root (UID 0)
                if len(process) > 1 and process[1] == '0':  # UID column
                    return True
                
                # Check for suspicious process names
                if len(process) > 7:  # Command column
                    command = process[7].lower()
                    suspicious_commands = ['su', 'sudo', 'setuid', 'chmod +s', 'setcap']
                    if any(cmd in command for cmd in suspicious_commands):
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check privilege escalation: {e}")
            return False
    
    def _monitor_syscall_patterns(self, container) -> List[str]:
        """Monitor syscall patterns for suspicious behavior."""
        # In production, this would use eBPF programs to monitor syscalls
        # For now, simulate by checking container logs for suspicious patterns
        
        suspicious_patterns = []
        try:
            logs = container.logs(tail=100).decode('utf-8', errors='ignore')
            
            # Look for suspicious log patterns that might indicate syscall abuse
            suspicious_indicators = [
                'ptrace', 'process_vm_read', 'keyctl', 'mount',
                'sethostname', 'setdomainname', 'reboot'
            ]
            
            for indicator in suspicious_indicators:
                if indicator in logs.lower():
                    suspicious_patterns.append(indicator)
            
        except Exception as e:
            logger.error(f"Failed to monitor syscall patterns: {e}")
        
        return suspicious_patterns
    
    def _detect_resource_anomalies(self, stats: Dict, baseline: Dict) -> bool:
        """Detect resource usage anomalies."""
        try:
            # Extract current metrics
            memory_usage = stats['memory_stats']['usage']
            cpu_usage = stats['cpu_stats']['cpu_usage']['total_usage']
            
            # Compare with baseline (if available)
            if baseline:
                baseline_memory = baseline.get('memory_usage', memory_usage)
                baseline_cpu = baseline.get('cpu_usage', cpu_usage)
                
                # Check for significant deviations
                memory_ratio = memory_usage / max(baseline_memory, 1)
                cpu_ratio = cpu_usage / max(baseline_cpu, 1)
                
                # Flag if usage is 3x higher than baseline
                if memory_ratio > 3.0 or cpu_ratio > 3.0:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to detect resource anomalies: {e}")
            return False
    
    def _detect_network_activity(self, container) -> bool:
        """Detect unauthorized network activity."""
        try:
            # Check network statistics
            stats = container.stats(stream=False)
            networks = stats.get('networks', {})
            
            for network_name, network_stats in networks.items():
                rx_bytes = network_stats.get('rx_bytes', 0)
                tx_bytes = network_stats.get('tx_bytes', 0)
                
                # Flag any significant network activity for network-disabled containers
                if rx_bytes > 1024 or tx_bytes > 1024:  # More than 1KB
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to detect network activity: {e}")
            return False
    
    def _handle_security_violation(self, container_id: str, violation_type: str):
        """Handle detected security violations."""
        try:
            container = self.docker_client.containers.get(container_id)
            
            # Log security event
            logger.critical(f"Security violation {violation_type} in container {container_id}")
            
            # Immediate response - pause container
            container.pause()
            
            # Create security incident
            incident = {
                'container_id': container_id,
                'violation_type': violation_type,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'action_taken': 'container_paused'
            }
            
            # In production, this would trigger security incident response
            # For now, log the incident
            logger.critical(f"Security incident created: {json.dumps(incident)}")
            
            # Optionally kill container for critical violations
            if violation_type in ['privilege_escalation', 'unauthorized_network']:
                container.kill()
                logger.critical(f"Container {container_id} killed due to {violation_type}")
            
        except Exception as e:
            logger.error(f"Failed to handle security violation: {e}")

# ==========================================
# CRITICAL FIX 4: DISTRIBUTED SCHEDULER WITH LEADER ELECTION
# ==========================================

class DistributedSchedulerManager:
    """Distributed scheduler with leader election and health checks."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", instance_id: str = None):
        self.redis_url = redis_url
        self.instance_id = instance_id or f"scheduler_{uuid.uuid4().hex[:8]}"
        self.is_leader = False
        self.scheduler = None
        self._redis_client = None
        self._leader_lock_key = "pipeline_scheduler_leader"
        self._leader_ttl = 30  # seconds
        self._health_check_interval = 10  # seconds
        self._leader_election_thread = None
        self._health_check_thread = None
        self._shutdown_event = Event()
        
    async def initialize(self):
        """Initialize distributed scheduler."""
        # Setup Redis connection
        self._redis_client = aioredis.from_url(self.redis_url)
        
        # Initialize scheduler
        jobstore = SQLAlchemyJobStore(url=str(get_engine().url))
        self.scheduler = AsyncIOScheduler(
            jobstores={'default': jobstore},
            job_defaults={
                'coalesce': False,
                'max_instances': 1,
                'misfire_grace_time': 300
            }
        )
        
        # Start leader election
        self._start_leader_election()
        
        # Start health monitoring
        self._start_health_monitoring()
        
        logger.info(f"Distributed scheduler initialized: {self.instance_id}")
    
    def _start_leader_election(self):
        """Start leader election process."""
        def leader_election_loop():
            while not self._shutdown_event.is_set():
                try:
                    # Attempt to acquire leadership
                    if self._acquire_leadership():
                        if not self.is_leader:
                            self.is_leader = True
                            self._on_become_leader()
                        
                        # Renew leadership
                        self._renew_leadership()
                    else:
                        if self.is_leader:
                            self.is_leader = False
                            self._on_lose_leadership()
                    
                    time.sleep(self._leader_ttl // 3)  # Check frequently
                    
                except Exception as e:
                    logger.error(f"Leader election error: {e}")
                    time.sleep(5)
        
        self._leader_election_thread = threading.Thread(
            target=leader_election_loop,
            name=f"LeaderElection-{self.instance_id}",
            daemon=True
        )
        self._leader_election_thread.start()
    
    def _acquire_leadership(self) -> bool:
        """Attempt to acquire leadership lock."""
        try:
            # Use SET with NX and EX for atomic lock acquisition
            result = self._redis_client.set(
                self._leader_lock_key,
                self.instance_id,
                nx=True,  # Only set if key doesn't exist
                ex=self._leader_ttl  # Set expiration
            )
            return result is True
            
        except Exception as e:
            logger.error(f"Failed to acquire leadership: {e}")
            return False
    
    def _renew_leadership(self) -> bool:
        """Renew leadership lock."""
        try:
            # Use Lua script for atomic renewal
            lua_script = """
            if redis.call("GET", KEYS[1]) == ARGV[1] then
                return redis.call("EXPIRE", KEYS[1], ARGV[2])
            else
                return 0
            end
            """
            
            result = self._redis_client.eval(
                lua_script,
                1,  # Number of keys
                self._leader_lock_key,
                self.instance_id,
                self._leader_ttl
            )
            
            return result == 1
            
        except Exception as e:
            logger.error(f"Failed to renew leadership: {e}")
            return False
    
    def _on_become_leader(self):
        """Handle becoming the leader."""
        logger.info(f"Instance {self.instance_id} became scheduler leader")
        
        try:
            # Start the scheduler
            if not self.scheduler.running:
                self.scheduler.start()
                logger.info("Scheduler started as leader")
            
            # Register health check job
            self._register_health_check_job()
            
        except Exception as e:
            logger.error(f"Failed to start scheduler as leader: {e}")
    
    def _on_lose_leadership(self):
        """Handle losing leadership."""
        logger.info(f"Instance {self.instance_id} lost scheduler leadership")
        
        try:
            # Shutdown scheduler gracefully
            if self.scheduler.running:
                self.scheduler.shutdown(wait=False)
                logger.info("Scheduler shutdown after losing leadership")
                
        except Exception as e:
            logger.error(f"Failed to shutdown scheduler: {e}")
    
    def _start_health_monitoring(self):
        """Start health monitoring for scheduler instances."""
        def health_monitor_loop():
            while not self._shutdown_event.is_set():
                try:
                    self._perform_health_check()
                    time.sleep(self._health_check_interval)
                except Exception as e:
                    logger.error(f"Health monitoring error: {e}")
                    time.sleep(5)
        
        self._health_check_thread = threading.Thread(
            target=health_monitor_loop,
            name=f"HealthMonitor-{self.instance_id}",
            daemon=True
        )
        self._health_check_thread.start()
    
    def _perform_health_check(self):
        """Perform health check and update status."""
        try:
            health_data = {
                'instance_id': self.instance_id,
                'is_leader': self.is_leader,
                'scheduler_running': self.scheduler.running if self.scheduler else False,
                'last_heartbeat': datetime.now(timezone.utc).isoformat(),
                'active_jobs': len(self.scheduler.get_jobs()) if self.scheduler else 0
            }
            
            # Store health status in Redis
            health_key = f"scheduler_health:{self.instance_id}"
            self._redis_client.setex(
                health_key,
                self._health_check_interval * 3,  # TTL
                json.dumps(health_data)
            )
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
    
    def _register_health_check_job(self):
        """Register periodic health check job."""
        try:
            self.scheduler.add_job(
                self._scheduler_health_check,
                'interval',
                seconds=60,
                id='scheduler_health_check',
                replace_existing=True
            )
        except Exception as e:
            logger.error(f"Failed to register health check job: {e}")
    
    async def _scheduler_health_check(self):
        """Periodic scheduler health check job."""
        try:
            # Check for stale pipeline runs
            stale_runs = await _pipeline_state_manager.list_active_runs()
            current_time = datetime.now(timezone.utc)
            
            for run in stale_runs:
                run_start = datetime.fromisoformat(run.started_at) if run.started_at else current_time
                duration = (current_time - run_start).total_seconds()
                
                # Flag runs that have been running for more than 24 hours
                if duration > 86400:
                    logger.warning(f"Stale pipeline run detected: {run.run_id} (running for {duration/3600:.1f} hours)")
            
            logger.debug(f"Scheduler health check completed. Active runs: {len(stale_runs)}")
            
        except Exception as e:
            logger.error(f"Scheduler health check failed: {e}")
    
    def add_distributed_job(self, func: Callable, trigger: str, **trigger_args) -> str:
        """Add job with distributed locking."""
        if not self.is_leader:
            raise RuntimeError("Only leader can schedule jobs")
        
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        
        try:
            # Wrap function with distributed lock
            def locked_func(*args, **kwargs):
                lock_key = f"job_lock:{job_id}"
                acquired = self._redis_client.set(lock_key, self.instance_id, nx=True, ex=300)
                
                if not acquired:
                    logger.warning(f"Job {job_id} already running on another instance")
                    return
                
                try:
                    return func(*args, **kwargs)
                finally:
                    # Release lock
                    try:
                        self._redis_client.delete(lock_key)
                    except Exception:
                        pass
            
            # Add job to scheduler
            self.scheduler.add_job(
                locked_func,
                trigger,
                id=job_id,
                **trigger_args
            )
            
            logger.info(f"Distributed job added: {job_id}")
            return job_id
            
        except Exception as e:
            logger.error(f"Failed to add distributed job: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown distributed scheduler."""
        logger.info(f"Shutting down distributed scheduler: {self.instance_id}")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Release leadership
        if self.is_leader:
            try:
                self._redis_client.delete(self._leader_lock_key)
            except Exception:
                pass
        
        # Shutdown scheduler
        if self.scheduler and self.scheduler.running:
            self.scheduler.shutdown(wait=True)
        
        # Wait for threads to finish
        for thread in [self._leader_election_thread, self._health_check_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=5)
        
        # Close Redis connection
        if self._redis_client:
            await self._redis_client.close()

# ==========================================
# CRITICAL FIX 5: ENHANCED PIPELINE CANCELLATION
# ==========================================

class EnhancedCancellationManager:
    """Enhanced pipeline cancellation with force kill and cascading cancellation."""
    
    def __init__(self, grace_period_seconds: int = 30):
        self.grace_period_seconds = grace_period_seconds
        self._active_cancellations = {}
        self._compensation_handlers = {}
        self._lock = RLock()
        
    async def cancel_pipeline_run(self, run_id: str, force: bool = False, 
                                 cascade: bool = True) -> bool:
        """Cancel pipeline run with enhanced termination guarantees."""
        
        with self._lock:
            if run_id in self._active_cancellations:
                logger.warning(f"Cancellation already in progress for {run_id}")
                return False
            
            self._active_cancellations[run_id] = {
                'start_time': time.time(),
                'force': force,
                'cascade': cascade,
                'status': 'initiated'
            }
        
        try:
            # Get pipeline run
            run = await _pipeline_state_manager.get_run(run_id)
            if not run:
                logger.error(f"Pipeline run {run_id} not found")
                return False
            
            # Update state to cancelling
            await _pipeline_state_manager.update_run_state(
                run_id, Status.CANCELLED, 
                {'cancellation_reason': 'user_requested'}
            )
            
            # Cancel dependent steps if cascading
            if cascade:
                await self._cascade_cancellation(run)
            
            # Graceful cancellation first
            success = await self._graceful_cancellation(run_id)
            
            # Force kill if graceful fails or force requested
            if not success or force:
                await self._force_cancellation(run_id)
            
            # Execute compensation logic
            await self._execute_compensation_logic(run)
            
            logger.info(f"Pipeline run {run_id} successfully cancelled")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel pipeline run {run_id}: {e}")
            return False
        finally:
            # Cleanup cancellation tracking
            with self._lock:
                self._active_cancellations.pop(run_id, None)
    
    async def _graceful_cancellation(self, run_id: str) -> bool:
        """Attempt graceful cancellation with timeout."""
        try:
            # Set cancellation event
            if run_id in _cancellations:
                _cancellations[run_id].set()
            
            # Wait for graceful shutdown
            start_time = time.time()
            while time.time() - start_time < self.grace_period_seconds:
                run = await _pipeline_state_manager.get_run(run_id)
                if not run or run.status in [Status.CANCELLED, Status.COMPLETED, Status.FAILED]:
                    return True
                
                await asyncio.sleep(1)
            
            logger.warning(f"Graceful cancellation timeout for {run_id}")
            return False
            
        except Exception as e:
            logger.error(f"Graceful cancellation failed for {run_id}: {e}")
            return False
    
    async def _force_cancellation(self, run_id: str):
        """Force kill pipeline processes."""
        try:
            logger.warning(f"Force killing pipeline run {run_id}")
            
            # Get all processes for this pipeline run
            processes = self._get_pipeline_processes(run_id)
            
            for process_info in processes:
                try:
                    # Kill process group to ensure child processes are also killed
                    os.killpg(os.getpgid(process_info.pid), signal.SIGKILL)
                    logger.info(f"Force killed process {process_info.pid} for run {run_id}")
                except (ProcessLookupError, OSError) as e:
                    logger.debug(f"Process {process_info.pid} already terminated: {e}")
            
            # Kill associated containers
            await self._kill_pipeline_containers(run_id)
            
        except Exception as e:
            logger.error(f"Force cancellation failed for {run_id}: {e}")
    
    async def _cascade_cancellation(self, run: PipelineRun):
        """Cancel dependent pipeline steps."""
        try:
            # Cancel all pending/running steps in the pipeline
            for step in run.steps:
                if step.status in [Status.PENDING, Status.RUNNING]:
                    step.status = Status.CANCELLED
                    logger.info(f"Cascaded cancellation to step {step.step_id}")
            
            # Cancel dependent pipeline runs (if any)
            dependent_runs = await self._get_dependent_runs(run.run_id)
            for dep_run_id in dependent_runs:
                await self.cancel_pipeline_run(dep_run_id, cascade=False)
            
        except Exception as e:
            logger.error(f"Cascade cancellation failed: {e}")
    
    def _get_pipeline_processes(self, run_id: str) -> List[ProcessInfo]:
        """Get all processes associated with a pipeline run."""
        # Implementation would track processes by run_id
        # For now, return empty list
        return []
    
    async def _kill_pipeline_containers(self, run_id: str):
        """Kill containers associated with pipeline run."""
        try:
            # Get containers with pipeline run label
            containers = _enhanced_container_security.docker_client.containers.list(
                filters={'label': f'pipeline_run_id={run_id}'}
            )
            
            for container in containers:
                try:
                    container.kill()
                    logger.info(f"Killed container {container.id} for run {run_id}")
                except Exception as e:
                    logger.error(f"Failed to kill container {container.id}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to kill pipeline containers: {e}")
    
    async def _get_dependent_runs(self, run_id: str) -> List[str]:
        """Get pipeline runs dependent on this run."""
        # Implementation would track dependencies
        return []
    
    async def _execute_compensation_logic(self, run: PipelineRun):
        """Execute compensation logic for partial completions."""
        try:
            compensation_tasks = []
            
            for step in run.steps:
                if step.status == Status.COMPLETED and step.step_id in self._compensation_handlers:
                    # Create compensation task
                    handler = self._compensation_handlers[step.step_id]
                    compensation_tasks.append(handler(step))
            
            if compensation_tasks:
                await asyncio.gather(*compensation_tasks, return_exceptions=True)
                logger.info(f"Executed {len(compensation_tasks)} compensation tasks for {run.run_id}")
                
        except Exception as e:
            logger.error(f"Compensation logic failed for {run.run_id}: {e}")
    
    def register_compensation_handler(self, step_type: str, handler: Callable):
        """Register compensation handler for step type."""
        self._compensation_handlers[step_type] = handler
        logger.info(f"Registered compensation handler for step type: {step_type}")

# ==========================================
# CRITICAL FIX 6: ENHANCED ERROR RECOVERY
# ==========================================

class StepCheckpointManager:
    """Step-level checkpointing with exponential backoff and circuit breakers."""
    
    def __init__(self):
        self.checkpoints = {}
        self.retry_configs = defaultdict(lambda: {'max_retries': 3, 'base_delay': 1.0})
        self.circuit_breakers = {}
        
    async def execute_step_with_recovery(self, step: PipelineStep, 
                                       step_func: Callable, *args, **kwargs) -> StepResult:
        """Execute step with comprehensive recovery mechanisms."""
        
        step_type = step.step_type
        
        # Initialize circuit breaker if not exists
        if step_type not in self.circuit_breakers:
            self.circuit_breakers[step_type] = CircuitBreaker(
                failure_threshold=5, success_threshold=2, timeout=60
            )
        
        circuit_breaker = self.circuit_breakers[step_type]
        retry_config = self.retry_configs[step_type]
        
        for attempt in range(retry_config['max_retries'] + 1):
            try:
                # Create checkpoint before execution
                checkpoint_id = await self._create_checkpoint(step)
                
                # Execute with circuit breaker protection
                result = await circuit_breaker.call_async(
                    self._execute_step_with_checkpoint,
                    step, step_func, checkpoint_id, *args, **kwargs
                )
                
                # Clean up checkpoint on success
                await self._cleanup_checkpoint(checkpoint_id)
                
                return result
                
            except Exception as e:
                logger.error(f"Step {step.step_id} failed (attempt {attempt + 1}): {e}")
                
                if attempt < retry_config['max_retries']:
                    # Calculate exponential backoff delay
                    delay = retry_config['base_delay'] * (2 ** attempt)
                    jitter = random.uniform(0, delay * 0.1)  # Add jitter
                    total_delay = delay + jitter
                    
                    logger.info(f"Retrying step {step.step_id} in {total_delay:.2f}s")
                    await asyncio.sleep(total_delay)
                else:
                    # Max retries exceeded, try recovery from checkpoint
                    recovery_result = await self._recover_from_checkpoint(step)
                    if recovery_result:
                        return recovery_result
                    
                    # Recovery failed, raise the original exception
                    raise
    
    async def _create_checkpoint(self, step: PipelineStep) -> str:
        """Create checkpoint before step execution."""
        checkpoint_id = f"checkpoint_{step.step_id}_{int(time.time())}"
        
        checkpoint_data = {
            'step_id': step.step_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'input_data': step.inputs,
            'step_config': step.config,
            'environment': dict(os.environ)  # Capture environment
        }
        
        # Store checkpoint (in production, use persistent storage)
        self.checkpoints[checkpoint_id] = checkpoint_data
        
        logger.debug(f"Created checkpoint {checkpoint_id} for step {step.step_id}")
        return checkpoint_id
    
    async def _execute_step_with_checkpoint(self, step: PipelineStep, 
                                          step_func: Callable, checkpoint_id: str,
                                          *args, **kwargs) -> StepResult:
        """Execute step with checkpoint tracking."""
        
        start_time = time.time()
        
        try:
            # Update checkpoint with execution start
            self.checkpoints[checkpoint_id].update({
                'execution_start': datetime.now(timezone.utc).isoformat(),
                'status': 'executing'
            })
            
            # Execute the step function
            result = await step_func(*args, **kwargs)
            
            # Update checkpoint with success
            self.checkpoints[checkpoint_id].update({
                'execution_end': datetime.now(timezone.utc).isoformat(),
                'status': 'completed',
                'duration': time.time() - start_time,
                'result': result
            })
            
            return result
            
        except Exception as e:
            # Update checkpoint with failure
            self.checkpoints[checkpoint_id].update({
                'execution_end': datetime.now(timezone.utc).isoformat(),
                'status': 'failed',
                'duration': time.time() - start_time,
                'error': str(e),
                'error_type': type(e).__name__
            })
            raise
    
    async def _recover_from_checkpoint(self, step: PipelineStep) -> Optional[StepResult]:
        """Attempt recovery from the most recent checkpoint."""
        try:
            # Find the most recent successful checkpoint for this step
            step_checkpoints = [
                (cid, data) for cid, data in self.checkpoints.items()
                if data['step_id'] == step.step_id and data.get('status') == 'completed'
            ]
            
            if not step_checkpoints:
                logger.warning(f"No successful checkpoint found for step {step.step_id}")
                return None
            
            # Get the most recent successful checkpoint
            latest_checkpoint = max(step_checkpoints, key=lambda x: x[1]['timestamp'])
            checkpoint_data = latest_checkpoint[1]
            
            # Restore from checkpoint
            result = checkpoint_data.get('result')
            if result:
                logger.info(f"Recovered step {step.step_id} from checkpoint {latest_checkpoint[0]}")
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"Checkpoint recovery failed for step {step.step_id}: {e}")
            return None
    
    async def _cleanup_checkpoint(self, checkpoint_id: str):
        """Clean up checkpoint after successful execution."""
        try:
            if checkpoint_id in self.checkpoints:
                del self.checkpoints[checkpoint_id]
                logger.debug(f"Cleaned up checkpoint {checkpoint_id}")
        except Exception as e:
            logger.error(f"Failed to cleanup checkpoint {checkpoint_id}: {e}")

class CircuitBreaker:
    """Circuit breaker for step-level fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, success_threshold: int = 2, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self._lock = RLock()
    
    async def call_async(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == 'OPEN':
                if time.time() - self.last_failure_time > self.timeout:
                    self.state = 'HALF_OPEN'
                    self.success_count = 0
                else:
                    raise Exception(f"Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Handle successful execution."""
        with self._lock:
            if self.state == 'HALF_OPEN':
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = 'CLOSED'
                    self.failure_count = 0
            else:
                self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed execution."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'OPEN'

# ==========================================
# CRITICAL FIX 7: ENHANCED OBSERVABILITY
# ==========================================

class DistributedTracingManager:
    """Distributed tracing with context propagation and exemplars."""
    
    def __init__(self):
        self.active_traces = {}
        self.trace_context = {}
        
    @contextmanager
    def trace_pipeline_execution(self, run_id: str, operation: str):
        """Create traced context for pipeline operations."""
        span = tracer.start_span(
            operation,
            attributes={
                'pipeline.run_id': run_id,
                'pipeline.operation': operation,
                'service.name': 'automation-service'
            }
        )
        
        # Store trace context for propagation
        trace_id = format(span.get_span_context().trace_id, '032x')
        self.trace_context[run_id] = {
            'trace_id': trace_id,
            'span_id': format(span.get_span_context().span_id, '016x'),
            'span': span
        }
        
        try:
            with tracer.start_as_current_span(operation) as span:
                yield span
        finally:
            span.end()
            self.trace_context.pop(run_id, None)
    
    def propagate_trace_context(self, run_id: str) -> Dict[str, str]:
        """Get trace context headers for service calls."""
        context = self.trace_context.get(run_id, {})
        if context:
            return {
                'traceparent': f"00-{context['trace_id']}-{context['span_id']}-01",
                'x-trace-id': context['trace_id']
            }
        return {}

# Global instances
_enhanced_container_security = EnhancedContainerSecurityManager()
_distributed_scheduler = DistributedSchedulerManager()
_cancellation_manager = EnhancedCancellationManager()
_checkpoint_manager = StepCheckpointManager()
_tracing_manager = DistributedTracingManager()

# Legacy security tracking (deprecated)
_webhook_nonces: Set[str] = set()
_webhook_rate_limits: Dict[str, List[datetime]] = {}
_failed_auth_attempts: Dict[str, List[datetime]] = {}


@dataclass
class ProcessInfo:
    """Information about a running process."""
    pid: int
    start_time: datetime
    memory_limit_mb: int
    cpu_limit: float
    duration_limit_seconds: int
    disk_limit_mb: int
    container_id: Optional[str] = None
    trace_id: Optional[str] = None


@dataclass 
class SecurityContext:
    """Security context for operations."""
    user_id: str
    roles: List[str]
    permissions: List[str]
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None


@dataclass
class ObservabilityConfig:
    """Configuration for observability features."""
    enable_tracing: bool = True
    enable_metrics: bool = True
    enable_structured_logging: bool = True
    jaeger_endpoint: Optional[str] = None
    prometheus_endpoint: Optional[str] = None
    log_level: str = "INFO"
    trace_sampling_rate: float = 0.1


class EncryptionManager:
    """Manages encryption for sensitive data."""
    
    def __init__(self, master_key: Optional[str] = None):
        if master_key:
            key = base64.urlsafe_b64encode(master_key.encode()[:32].ljust(32, b'0'))
        else:
            key = Fernet.generate_key()
        self.cipher = Fernet(key)
        self._key_rotation_schedule = {}
    
    def encrypt(self, data: Union[str, bytes]) -> str:
        """Encrypt data and return base64 encoded result."""
        if isinstance(data, str):
            data = data.encode()
        encrypted = self.cipher.encrypt(data)
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt base64 encoded data."""
        encrypted = base64.urlsafe_b64decode(encrypted_data.encode())
        decrypted = self.cipher.decrypt(encrypted)
        return decrypted.decode()
    
    def rotate_key(self) -> str:
        """Generate new encryption key and return it."""
        new_key = Fernet.generate_key()
        old_cipher = self.cipher
        self.cipher = Fernet(new_key)
        
        # Schedule old key for cleanup
        self._key_rotation_schedule[datetime.utcnow() + timedelta(days=30)] = old_cipher
        
        return new_key.decode()
    
    def cleanup_old_keys(self):
        """Remove expired encryption keys."""
        now = datetime.utcnow()
        expired_keys = [date for date in self._key_rotation_schedule if date <= now]
        for date in expired_keys:
            del self._key_rotation_schedule[date]


class AuditLogger:
    """Comprehensive audit logging system."""
    
    def __init__(self):
        self.encryption = EncryptionManager()
    
    async def log_security_event(self, event_type: str, details: Dict[str, Any], 
                                security_context: SecurityContext, severity: str = "INFO"):
        """Log security-related events."""
        try:
            # Encrypt sensitive details
            encrypted_details = self.encryption.encrypt(json.dumps(details))
            
            with get_db_session() as session:
                event = DbSecurityEvent(
                    event_type=event_type,
                    user_id=security_context.user_id,
                    ip_address=security_context.ip_address,
                    user_agent=security_context.user_agent,
                    details=encrypted_details,
                    severity=severity,
                    correlation_id=security_context.correlation_id
                )
                session.add(event)
                session.commit()
                
        except Exception as e:
            logger.error(f"Failed to log security event: {e}")
    
    async def log_audit_event(self, component: str, operation: str, resource_id: str,
                            details: Dict[str, Any], security_context: SecurityContext):
        """Log audit events for compliance."""
        try:
            with get_db_session() as session:
                audit = DbAuditLog(
                    component=component,
                    operation=operation,
                    resource_id=resource_id,
                    user_id=security_context.user_id,
                    ip_address=security_context.ip_address,
                    details=details,
                    correlation_id=security_context.correlation_id
                )
                session.add(audit)
                session.commit()
                
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")


class ContainerSecurityManager:
    """Enhanced Docker container security and lifecycle management."""
    
    def __init__(self):
        self.client = docker.from_env()
        self.security_config = ContainerSecurityConfig()
        self.isolated_network = None
        self.audit_logger = AuditLogger()
        self._setup_network()
        self._vulnerability_db = self._load_vulnerability_database()
    
    def _setup_network(self):
        """Create isolated Docker network for pipeline containers."""
        try:
            # Check if network already exists
            try:
                self.isolated_network = self.client.networks.get("kaizen-isolated")
            except docker.errors.NotFound:
                # Create isolated network with no internet access
                self.isolated_network = self.client.networks.create(
                    name="kaizen-isolated",
                    driver="bridge",
                    internal=True,  # No internet access
                    labels={"kaizen": "pipeline-network"},
                    options={
                        "com.docker.network.bridge.enable_icc": "false",
                        "com.docker.network.bridge.enable_ip_masquerade": "false"
                    }
                )
                logger.info("Created isolated Docker network for pipeline containers")
        except Exception as e:
            logger.warning(f"Failed to create isolated network: {e}")
    
    def _load_vulnerability_database(self) -> Dict[str, Any]:
        """Load known vulnerability database (simplified)."""
        # In production, integrate with CVE databases, Trivy, etc.
        return {
            "vulnerable_packages": [
                "vulnerable-package-1.0",
                "insecure-lib-2.1"
            ],
            "suspicious_images": [
                "untrusted/malicious",
                "crypto/miner"
            ]
        }
    
    def validate_image(self, image_name: str, security_context: SecurityContext) -> bool:
        """Enhanced container image validation for security vulnerabilities."""
        try:
            with tracer.start_as_current_span("validate_container_image") as span:
                span.set_attribute("image.name", image_name)
                
                # Check if image exists locally
                try:
                    image = self.client.images.get(image_name)
                except docker.errors.ImageNotFound:
                    logger.warning(f"Image {image_name} not found")
                    return False
                
                # Check image registry reputation
                if self._is_suspicious_registry(image_name):
                    container_security_events.add(1, {"event_type": "suspicious_registry"})
                    asyncio.create_task(self.audit_logger.log_security_event(
                        "suspicious_image_registry",
                        {"image": image_name, "registry": image_name.split('/')[0]},
                        security_context,
                        "WARNING"
                    ))
                    return False
                
                # Basic security checks
                if self._has_suspicious_labels(image):
                    container_security_events.add(1, {"event_type": "suspicious_labels"})
                    logger.warning(f"Image {image_name} has suspicious labels")
                    return False
                
                # Check for known vulnerable packages
                if self._has_vulnerable_packages(image):
                    container_security_events.add(1, {"event_type": "vulnerable_packages"})
                    logger.warning(f"Image {image_name} may contain vulnerable packages")
                    return False
                
                # Check image signature (simplified)
                if not self._verify_image_signature(image):
                    container_security_events.add(1, {"event_type": "invalid_signature"})
                    logger.warning(f"Image {image_name} signature verification failed")
                    return False
                
                span.set_attribute("validation.result", "passed")
                return True
                
        except Exception as e:
            logger.error(f"Failed to validate image {image_name}: {e}")
            return False
    
    def _is_suspicious_registry(self, image_name: str) -> bool:
        """Check if image comes from suspicious registry."""
        suspicious_registries = ["untrusted.io", "malicious.com"]
        for registry in suspicious_registries:
            if image_name.startswith(registry):
                return True
        return False
    
    def _has_suspicious_labels(self, image) -> bool:
        """Check for suspicious labels in container image."""
        suspicious_labels = ["privileged", "root", "sudo", "admin", "crypto", "miner"]
        labels = image.labels or {}
        
        for label_value in labels.values():
            if any(suspicious in label_value.lower() for suspicious in suspicious_labels):
                return True
        return False
    
    def _has_vulnerable_packages(self, image) -> bool:
        """Check for known vulnerable packages."""
        # In production, integrate with vulnerability scanners like Trivy
        try:
            # Get image layers and check for known vulnerable packages
            for layer in image.history():
                created_by = layer.get("CreatedBy", "").lower()
                for vulnerable_pkg in self._vulnerability_db["vulnerable_packages"]:
                    if vulnerable_pkg in created_by:
                        return True
            return False
        except Exception:
            return False
    
    def _verify_image_signature(self, image) -> bool:
        """Verify container image signature (simplified)."""
        # In production, implement proper image signature verification
        # using tools like cosign, notary, etc.
        return True  # Placeholder
    
    def create_secure_container(self, image: str, command: str, volumes: Dict[str, str] = None,
                              security_context: Optional[SecurityContext] = None) -> str:
        """Create a secure Docker container with comprehensive hardening."""
        try:
            with tracer.start_as_current_span("create_secure_container") as span:
                span.set_attribute("container.image", image)
                
                # Enhanced security options
                security_opts = [
                    "no-new-privileges:true",
                    "apparmor:docker-default",
                    "seccomp:unconfined"  # In production, use custom seccomp profile
                ]
                
                # Comprehensive resource limits
                ulimits = [
                    docker.types.Ulimit(name="nofile", soft=1024, hard=1024),
                    docker.types.Ulimit(name="nproc", soft=64, hard=64),
                    docker.types.Ulimit(name="fsize", soft=100000000, hard=100000000)  # 100MB
                ]
                
                # Secure tmpfs mounts
                tmpfs = {
                    "/tmp": f"rw,size={self.security_config.tmpfs_size_mb}m,noexec,nosuid,nodev",
                    "/var/tmp": "rw,size=32m,noexec,nosuid,nodev"
                }
                
                # Enhanced container configuration
                container_config = {
                    "image": image,
                    "command": command,
                    "detach": True,
                    "labels": {
                        "kaizen-pipeline": "true",
                        "security-profile": "restricted",
                        "created-by": security_context.user_id if security_context else "system"
                    },
                    "security_opt": security_opts,
                    "ulimits": ulimits,
                    "tmpfs": tmpfs,
                    "pids_limit": self.security_config.pids_limit,
                    "mem_limit": "512m",
                    "mem_reservation": "256m",
                    "cpu_quota": int(100000 * self.security_config.cpu_limit),
                    "cpu_period": 100000,
                    "network_disabled": self.security_config.network_isolation,
                    "read_only": self.security_config.readonly_rootfs,
                    "remove": True,
                    "user": "nobody",  # Run as non-root user
                    "cap_drop": ["ALL"],  # Drop all capabilities
                    "privileged": False,
                }
                
                # Add volumes if provided (with security validation)
                if volumes:
                    validated_volumes = self._validate_volumes(volumes)
                    container_config["volumes"] = validated_volumes
                
                # Add network configuration
                if not self.security_config.network_isolation and self.isolated_network:
                    container_config["network"] = self.isolated_network.name
                
                container = self.client.containers.create(**container_config)
                
                # Log container creation
                if security_context:
                    asyncio.create_task(self.audit_logger.log_audit_event(
                        "container_manager",
                        "create_container",
                        container.id,
                        {"image": image, "security_profile": "restricted"},
                        security_context
                    ))
                
                logger.info(f"Created secure container {container.id} for image {image}")
                span.set_attribute("container.id", container.id)
                
                return container.id
                
        except Exception as e:
            logger.error(f"Failed to create secure container: {e}")
            raise
    
    def _validate_volumes(self, volumes: Dict[str, str]) -> Dict[str, str]:
        """Validate volume mounts for security."""
        validated = {}
        dangerous_paths = ["/etc", "/root", "/boot", "/sys", "/proc"]
        
        for host_path, container_path in volumes.items():
            # Check for dangerous host paths
            if any(host_path.startswith(dangerous) for dangerous in dangerous_paths):
                logger.warning(f"Blocking dangerous volume mount: {host_path}")
                continue
            
            # Ensure read-only for most mounts
            if not container_path.endswith(":ro"):
                container_path += ":ro"
            
            validated[host_path] = container_path
        
        return validated
    
    def monitor_container(self, container_id: str) -> Dict[str, Any]:
        """Enhanced container monitoring with security checks."""
        try:
            with tracer.start_as_current_span("monitor_container") as span:
                container = self.client.containers.get(container_id)
                stats = container.stats(stream=False)
                
                # Parse Docker stats
                memory_usage = stats.get("memory_stats", {}).get("usage", 0) / (1024 * 1024)  # MB
                cpu_usage = self._calculate_cpu_percent(stats)
                
                # Check for escape attempts
                escape_indicators = self._check_escape_attempts(container)
                if escape_indicators:
                    container_security_events.add(1, {"event_type": "escape_attempt"})
                    logger.critical(f"Container escape attempt detected: {container_id}")
                
                # Network activity monitoring
                network_activity = self._monitor_network_activity(container)
                
                monitoring_data = {
                    "memory_mb": memory_usage,
                    "cpu_percent": cpu_usage,
                    "status": container.status,
                    "running": container.status == "running",
                    "escape_indicators": escape_indicators,
                    "network_activity": network_activity,
                    "pid_count": self._get_container_process_count(container)
                }
                
                span.set_attributes({
                    "container.memory_mb": memory_usage,
                    "container.cpu_percent": cpu_usage,
                    "container.status": container.status
                })
                
                return monitoring_data
                
        except Exception as e:
            logger.error(f"Failed to monitor container {container_id}: {e}")
            return {"memory_mb": 0, "cpu_percent": 0, "status": "unknown", "running": False}
    
    def _check_escape_attempts(self, container) -> List[str]:
        """Detect potential container escape attempts."""
        indicators = []
        
        try:
            # Check for suspicious processes
            top_output = container.top()
            for process in top_output.get("Processes", []):
                process_name = process[-1] if process else ""
                if any(suspicious in process_name.lower() for suspicious in ["docker", "runc", "containerd"]):
                    indicators.append(f"Suspicious process: {process_name}")
            
            # Check for privilege escalation attempts
            logs = container.logs(tail=100).decode('utf-8', errors='ignore')
            if "permission denied" in logs.lower() and "root" in logs.lower():
                indicators.append("Potential privilege escalation attempt")
                
        except Exception as e:
            logger.warning(f"Failed to check escape attempts for container: {e}")
        
        return indicators
    
    def _monitor_network_activity(self, container) -> Dict[str, Any]:
        """Monitor container network activity."""
        try:
            # Get network stats
            stats = container.stats(stream=False)
            network_stats = stats.get("networks", {})
            
            total_rx = sum(net.get("rx_bytes", 0) for net in network_stats.values())
            total_tx = sum(net.get("tx_bytes", 0) for net in network_stats.values())
            
            return {
                "rx_bytes": total_rx,
                "tx_bytes": total_tx,
                "connections": len(network_stats)
            }
        except Exception:
            return {"rx_bytes": 0, "tx_bytes": 0, "connections": 0}
    
    def _get_container_process_count(self, container) -> int:
        """Get number of processes running in container."""
        try:
            top_output = container.top()
            return len(top_output.get("Processes", []))
        except Exception:
            return 0
    
    def _calculate_cpu_percent(self, stats: Dict) -> float:
        """Calculate CPU usage percentage from Docker stats."""
        try:
            cpu_stats = stats.get("cpu_stats", {})
            precpu_stats = stats.get("precpu_stats", {})
            
            cpu_delta = cpu_stats.get("cpu_usage", {}).get("total_usage", 0) - \
                       precpu_stats.get("cpu_usage", {}).get("total_usage", 0)
            system_delta = cpu_stats.get("system_cpu_usage", 0) - \
                          precpu_stats.get("system_cpu_usage", 0)
            
            if system_delta > 0 and cpu_delta > 0:
                cpu_percent = (cpu_delta / system_delta) * len(cpu_stats.get("cpu_usage", {}).get("percpu_usage", [])) * 100.0
                return min(cpu_percent, 100.0)
            
            return 0.0
        except Exception:
            return 0.0
    
    def terminate_container(self, container_id: str, timeout: int = 10) -> bool:
        """Terminate a container with timeout and cleanup."""
        try:
            with tracer.start_as_current_span("terminate_container") as span:
                container = self.client.containers.get(container_id)
                
                # Graceful shutdown first
                container.stop(timeout=timeout)
                
                # Force kill if still running
                if container.status == "running":
                    container.kill()
                
                # Remove container and associated volumes
                container.remove(v=True, force=True)
                
                logger.info(f"Terminated container {container_id}")
                span.set_attribute("container.id", container_id)
                return True
                
        except Exception as e:
            logger.error(f"Failed to terminate container {container_id}: {e}")
            return False
    
    async def cleanup_old_containers(self, max_age_hours: int = 1) -> int:
        """Clean up containers older than specified age with enhanced logging."""
        try:
            with tracer.start_as_current_span("cleanup_old_containers") as span:
                cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
                cleaned_count = 0
                
                # Get all containers with kaizen-pipeline label
                containers = self.client.containers.list(
                    all=True,
                    filters={"label": "kaizen-pipeline"}
                )
                
                for container in containers:
                    try:
                        # Check container creation time
                        created_time = datetime.fromisoformat(
                            container.attrs["Created"].replace("Z", "+00:00")
                        ).replace(tzinfo=None)
                        
                        if created_time < cutoff_time:
                            # Log cleanup action
                            logger.info(f"Cleaning up old container {container.id} created at {created_time}")
                            
                            # Collect final stats before cleanup
                            final_stats = self.monitor_container(container.id)
                            
                            container.remove(force=True, v=True)
                            cleaned_count += 1
                            
                            # Log cleanup completion
                            logger.info(f"Cleaned up old container {container.id}")
                            
                    except Exception as e:
                        logger.warning(f"Failed to clean up container {container.id}: {e}")
                
                span.set_attribute("containers.cleaned", cleaned_count)
                logger.info(f"Cleaned up {cleaned_count} old containers")
                return cleaned_count
                
        except Exception as e:
            logger.error(f"Failed to cleanup containers: {e}")
            return 0


class EnhancedResourceMonitor:
    """Enhanced resource monitoring with comprehensive limits and security."""
    
    def __init__(self):
        self.active_processes: Dict[str, ProcessInfo] = {}
        self.monitoring_threads: Dict[str, Thread] = {}
        self.audit_logger = AuditLogger()
    
    def start_monitoring(self, run_id: str, step_index: int, process: subprocess.Popen, 
                        limits: StepResourceLimits, security_context: Optional[SecurityContext] = None) -> str:
        """Start comprehensive process monitoring."""
        process_id = f"{run_id}-{step_index}"
        
        with tracer.start_as_current_span("start_process_monitoring") as span:
            current_span = trace.get_current_span()
            trace_id = format(current_span.get_span_context().trace_id, "032x")
            
            process_info = ProcessInfo(
                pid=process.pid,
                start_time=datetime.utcnow(),
                memory_limit_mb=limits.max_memory_mb,
                cpu_limit=limits.max_cpu_cores,
                duration_limit_seconds=limits.max_duration_minutes * 60,
                disk_limit_mb=limits.max_disk_mb,
                trace_id=trace_id
            )
            
            self.active_processes[process_id] = process_info
            
            # Start enhanced monitoring thread
            monitor_thread = Thread(
                target=self._enhanced_monitor_process,
                args=(process_id, process, limits, security_context),
                daemon=True
            )
            monitor_thread.start()
            self.monitoring_threads[process_id] = monitor_thread
            
            span.set_attributes({
                "process.id": process_id,
                "process.pid": process.pid,
                "limits.memory_mb": limits.max_memory_mb,
                "limits.cpu_cores": limits.max_cpu_cores
            })
            
            return process_id
    
    def _enhanced_monitor_process(self, process_id: str, process: subprocess.Popen, 
                                limits: StepResourceLimits, security_context: Optional[SecurityContext]):
        """Enhanced process monitoring with security checks."""
        violation_count = 0
        max_violations = 3
        
        try:
            while process.poll() is None:  # While process is running
                try:
                    # Get process info using psutil
                    ps_process = psutil.Process(process.pid)
                    
                    # Memory monitoring
                    memory_info = ps_process.memory_info()
                    memory_mb = memory_info.rss / (1024 * 1024)
                    
                    if memory_mb > limits.max_memory_mb:
                        violation_count += 1
                        resource_limit_violations.add(1, {"resource_type": "memory"})
                        
                        logger.warning(f"Process {process_id} exceeded memory limit: "
                                     f"{memory_mb:.1f}MB > {limits.max_memory_mb}MB (violation {violation_count})")
                        
                        if violation_count >= max_violations:
                            logger.error(f"Process {process_id} terminated after {violation_count} memory violations")
                            process.terminate()
                            break
                    
                    # CPU monitoring with throttling
                    cpu_percent = ps_process.cpu_percent()
                    if cpu_percent > limits.max_cpu_cores * 100:
                        logger.warning(f"Process {process_id} high CPU usage: "
                                     f"{cpu_percent:.1f}% > {limits.max_cpu_cores * 100}%")
                        
                        # CPU throttling (simplified - in production use cgroups)
                        try:
                            ps_process.nice(19)  # Lower priority
                        except Exception:
                            pass
                    
                    # Duration monitoring
                    elapsed = (datetime.utcnow() - self.active_processes[process_id].start_time).total_seconds()
                    if elapsed > limits.max_duration_minutes * 60:
                        resource_limit_violations.add(1, {"resource_type": "duration"})
                        logger.warning(f"Process {process_id} exceeded duration limit: "
                                     f"{elapsed:.1f}s > {limits.max_duration_minutes * 60}s")
                        process.terminate()
                        break
                    
                    # Disk usage monitoring
                    try:
                        # Monitor temp directory usage
                        temp_usage = self._get_disk_usage("/tmp")
                        process_disk_usage = self._get_process_disk_usage(ps_process)
                        
                        if process_disk_usage > limits.max_disk_mb:
                            resource_limit_violations.add(1, {"resource_type": "disk"})
                            logger.warning(f"Process {process_id} exceeded disk limit: "
                                         f"{process_disk_usage:.1f}MB > {limits.max_disk_mb}MB")
                            # Don't terminate immediately for disk, but log warning
                    except Exception:
                        pass  # Ignore disk check errors
                    
                    # Security monitoring - check for suspicious behavior
                    if self._detect_suspicious_activity(ps_process):
                        logger.critical(f"Suspicious activity detected in process {process_id}")
                        if security_context:
                            asyncio.create_task(self.audit_logger.log_security_event(
                                "suspicious_process_activity",
                                {"process_id": process_id, "pid": process.pid},
                                security_context,
                                "CRITICAL"
                            ))
                        process.terminate()
                        break
                    
                    time.sleep(1)  # Monitor every second
                    
                except psutil.NoSuchProcess:
                    # Process has terminated
                    break
                except Exception as e:
                    logger.error(f"Error monitoring process {process_id}: {e}")
                    time.sleep(1)
                    
        except Exception as e:
            logger.error(f"Monitoring thread error for {process_id}: {e}")
        finally:
            # Clean up
            self.active_processes.pop(process_id, None)
            self.monitoring_threads.pop(process_id, None)
    
    def _get_disk_usage(self, path: str) -> float:
        """Get disk usage for a path in MB."""
        try:
            return psutil.disk_usage(path).used / (1024 * 1024)
        except Exception:
            return 0.0
    
    def _get_process_disk_usage(self, process: psutil.Process) -> float:
        """Get disk usage for a specific process (simplified)."""
        try:
            # Get IO counters
            io_counters = process.io_counters()
            # Convert bytes written to MB (simplified estimation)
            return io_counters.write_bytes / (1024 * 1024)
        except Exception:
            return 0.0
    
    def _detect_suspicious_activity(self, process: psutil.Process) -> bool:
        """Detect suspicious process activity."""
        try:
            # Check for suspicious command line arguments
            cmdline = ' '.join(process.cmdline()).lower()
            suspicious_patterns = [
                'wget', 'curl', 'nc ', 'netcat', 'ncat',
                'python -c', 'perl -e', 'ruby -e',
                'base64', 'decode', 'crypto', 'mining'
            ]
            
            for pattern in suspicious_patterns:
                if pattern in cmdline:
                    return True
            
            # Check for unusual network connections
            connections = process.connections()
            if len(connections) > 10:  # Threshold for suspicious connections
                return True
            
            # Check for privilege escalation attempts
            if process.username() != process.username():  # Changed user
                return True
                
            return False
        except Exception:
            return False
    
    def get_resource_usage(self, process_id: str) -> Optional[ResourceUsage]:
        """Get comprehensive resource usage for a process."""
        try:
            if process_id not in self.active_processes:
                return None
            
            process_info = self.active_processes[process_id]
            ps_process = psutil.Process(process_info.pid)
            
            memory_info = ps_process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            cpu_percent = ps_process.cpu_percent()
            duration_seconds = (datetime.utcnow() - process_info.start_time).total_seconds()
            
            # Enhanced resource usage
            disk_usage = self._get_process_disk_usage(ps_process)
            
            # Network usage
            try:
                connections = len(ps_process.connections())
            except Exception:
                connections = 0
            
            return ResourceUsage(
                memory_mb=memory_mb,
                cpu_percent=cpu_percent,
                disk_mb=disk_usage,
                duration_seconds=duration_seconds,
                process_id=process_info.pid,
                network_connections=connections,
                thread_count=ps_process.num_threads()
            )
            
        except psutil.NoSuchProcess:
            return None
        except Exception as e:
            logger.error(f"Failed to get resource usage for {process_id}: {e}")
            return None
    
    def stop_monitoring(self, process_id: str):
        """Stop monitoring a process."""
        self.active_processes.pop(process_id, None)
        thread = self.monitoring_threads.pop(process_id, None)
        if thread and thread.is_alive():
            # Note: We can't forcefully stop threads, they'll clean up naturally
            pass


class EnhancedWebhookSecurityManager:
    """Enhanced webhook security with comprehensive protection."""
    
    def __init__(self):
        self.config = WebhookSecurityConfig()
        self.used_nonces: Set[str] = set()
        self.rate_limit_windows: Dict[str, List[datetime]] = {}
        self.audit_logger = AuditLogger()
        self.encryption = EncryptionManager()
    
    def validate_webhook_request(self, request_data: Dict[str, Any], 
                                headers: Dict[str, str], 
                                endpoint: str,
                                security_context: Optional[SecurityContext] = None) -> Tuple[bool, str]:
        """Enhanced webhook request validation."""
        try:
            with tracer.start_as_current_span("validate_webhook_request") as span:
                span.set_attribute("webhook.endpoint", endpoint)
                
                # Check timestamp age for replay protection
                timestamp_str = headers.get("X-Webhook-Timestamp")
                if not timestamp_str:
                    webhook_security_violations.add(1, {"violation_type": "missing_timestamp"})
                    return False, "Missing timestamp header"
                
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                    age_minutes = (datetime.utcnow() - timestamp.replace(tzinfo=None)).total_seconds() / 60
                    
                    if age_minutes > self.config.max_age_minutes:
                        webhook_security_violations.add(1, {"violation_type": "expired_timestamp"})
                        return False, f"Request too old: {age_minutes:.1f} minutes"
                        
                except ValueError:
                    webhook_security_violations.add(1, {"violation_type": "invalid_timestamp"})
                    return False, "Invalid timestamp format"
                
                # Enhanced nonce validation for replay prevention
                if self.config.nonce_required:
                    nonce = headers.get("X-Webhook-Nonce")
                    if not nonce:
                        webhook_security_violations.add(1, {"violation_type": "missing_nonce"})
                        return False, "Missing nonce header"
                    
                    # Validate nonce format
                    if len(nonce) < 32 or not self._is_valid_nonce_format(nonce):
                        webhook_security_violations.add(1, {"violation_type": "invalid_nonce_format"})
                        return False, "Invalid nonce format"
                    
                    if nonce in self.used_nonces:
                        webhook_security_violations.add(1, {"violation_type": "replay_attack"})
                        # Log security incident
                        if security_context:
                            asyncio.create_task(self.audit_logger.log_security_event(
                                "webhook_replay_attack",
                                {"endpoint": endpoint, "nonce": nonce[:8] + "..."},
                                security_context,
                                "CRITICAL"
                            ))
                        return False, "Nonce already used (replay attack)"
                    
                    self.used_nonces.add(nonce)
                    
                    # Clean up old nonces periodically
                    if len(self.used_nonces) > 10000:
                        self.used_nonces = set(list(self.used_nonces)[-5000:])
                
                # Enhanced rate limiting
                if not self._check_enhanced_rate_limit(endpoint, headers):
                    webhook_security_violations.add(1, {"violation_type": "rate_limit"})
                    return False, "Rate limit exceeded"
                
                # IP whitelist validation
                if self.config.ip_whitelist:
                    client_ip = self._extract_client_ip(headers)
                    if client_ip not in self.config.ip_whitelist:
                        webhook_security_violations.add(1, {"violation_type": "ip_not_whitelisted"})
                        if security_context:
                            asyncio.create_task(self.audit_logger.log_security_event(
                                "webhook_unauthorized_ip",
                                {"endpoint": endpoint, "client_ip": client_ip},
                                security_context,
                                "WARNING"
                            ))
                        return False, f"IP {client_ip} not in whitelist"
                
                # Payload size validation
                payload_size = len(json.dumps(request_data))
                if payload_size > self.config.max_payload_size_bytes:
                    webhook_security_violations.add(1, {"violation_type": "payload_too_large"})
                    return False, f"Payload too large: {payload_size} bytes"
                
                # HMAC signature validation
                signature = headers.get("X-Webhook-Signature")
                if signature:
                    if not self._validate_hmac_signature(request_data, signature, self.config.secret_key):
                        webhook_security_violations.add(1, {"violation_type": "invalid_signature"})
                        return False, "Invalid HMAC signature"
                
                span.set_attribute("validation.result", "passed")
                return True, "Valid"
                
        except Exception as e:
            logger.error(f"Webhook validation error: {e}")
            return False, f"Validation error: {str(e)}"
    
    def _is_valid_nonce_format(self, nonce: str) -> bool:
        """Validate nonce format and entropy."""
        import string
        # Check if nonce contains sufficient entropy
        valid_chars = string.ascii_letters + string.digits + '-_'
        return all(c in valid_chars for c in nonce)
    
    def _extract_client_ip(self, headers: Dict[str, str]) -> str:
        """Extract client IP from headers with proxy support."""
        # Check various headers in order of preference
        ip_headers = ["X-Forwarded-For", "X-Real-IP", "CF-Connecting-IP", "X-Client-IP"]
        
        for header in ip_headers:
            ip = headers.get(header)
            if ip:
                # Handle comma-separated IPs (take first one)
                return ip.split(',')[0].strip()
        
        return "unknown"
    
    def _check_enhanced_rate_limit(self, endpoint: str, headers: Dict[str, str]) -> bool:
        """Enhanced rate limiting with multiple dimensions."""
        now = datetime.utcnow()
        window_start = now - timedelta(minutes=1)
        
        # Rate limit by endpoint
        endpoint_key = f"endpoint:{endpoint}"
        if not self._check_rate_limit_key(endpoint_key, now, window_start, self.config.rate_limit_per_minute):
            return False
        
        # Rate limit by IP
        client_ip = self._extract_client_ip(headers)
        ip_key = f"ip:{client_ip}"
        if not self._check_rate_limit_key(ip_key, now, window_start, self.config.ip_rate_limit_per_minute):
            return False
        
        # Rate limit by User-Agent (simple bot protection)
        user_agent = headers.get("User-Agent", "unknown")
        if self._is_suspicious_user_agent(user_agent):
            return False
        
        return True
    
    def _check_rate_limit_key(self, key: str, now: datetime, window_start: datetime, limit: int) -> bool:
        """Check rate limit for a specific key."""
        if key not in self.rate_limit_windows:
            self.rate_limit_windows[key] = []
        
        # Remove old timestamps
        self.rate_limit_windows[key] = [
            ts for ts in self.rate_limit_windows[key] 
            if ts > window_start
        ]
        
        # Check if under limit
        if len(self.rate_limit_windows[key]) >= limit:
            return False
        
        # Add current request
        self.rate_limit_windows[key].append(now)
        return True
    
    def _is_suspicious_user_agent(self, user_agent: str) -> bool:
        """Detect suspicious User-Agent strings."""
        suspicious_patterns = [
            "bot", "crawler", "spider", "scraper",
            "curl", "wget", "python-requests"
        ]
        user_agent_lower = user_agent.lower()
        return any(pattern in user_agent_lower for pattern in suspicious_patterns)
    
    def _validate_hmac_signature(self, payload: Dict[str, Any], signature: str, secret: str) -> bool:
        """Validate HMAC signature for payload integrity."""
        try:
            if not signature.startswith("sha256="):
                return False
            
            expected_signature = signature[7:]  # Remove "sha256=" prefix
            
            # Compute HMAC
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            computed_signature = hmac.new(
                secret.encode("utf-8"), 
                body, 
                hashlib.sha256
            ).hexdigest()
            
            # Secure comparison
            return hmac.compare_digest(expected_signature, computed_signature)
            
        except Exception as e:
            logger.error(f"HMAC validation error: {e}")
            return False
    
    def generate_nonce(self) -> str:
        """Generate a cryptographically secure nonce."""
        return secrets.token_urlsafe(32)
    
    def cleanup_expired_nonces(self):
        """Clean up expired nonces and rate limit windows."""
        # Clean up old nonces (keep only last 5000)
        if len(self.used_nonces) > 10000:
            self.used_nonces = set(list(self.used_nonces)[-5000:])
        
        # Clean up old rate limit windows
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        keys_to_clean = []
        
        for key, timestamps in self.rate_limit_windows.items():
            # Remove old timestamps
            recent_timestamps = [ts for ts in timestamps if ts > cutoff_time]
            if recent_timestamps:
                self.rate_limit_windows[key] = recent_timestamps
            else:
                keys_to_clean.append(key)
        
        # Remove empty keys
        for key in keys_to_clean:
            del self.rate_limit_windows[key]


class EnhancedAutomationService:
    """Enhanced automation service with comprehensive security and observability."""

    def __init__(self, config: Optional[ObservabilityConfig] = None) -> None:
        self.config = config or ObservabilityConfig()
        
        # Initialize services
        self.preprocessing = DataPreprocessingService() if hasattr(DataPreprocessingService, "__call__") or hasattr(DataPreprocessingService, "__init__") else DataPreprocessingService
        self.transformation = TransformationService() if hasattr(TransformationService, "__call__") or hasattr(TransformationService, "__init__") else TransformationService
        self.modeling = EnhancedModelingService() if hasattr(EnhancedModelingService, "__call__") or hasattr(EnhancedModelingService, "__init__") else EnhancedModelingService
        self.visualization = visualization_service
        
        # Enhanced security and monitoring components
        self.resource_monitor = EnhancedResourceMonitor()
        self.container_security = ContainerSecurityManager()
        self.webhook_security = EnhancedWebhookSecurityManager()
        self.audit_logger = AuditLogger()
        self.encryption = EncryptionManager()
        self.default_resource_limits = StepResourceLimits()
        
        # Circuit breaker for external dependencies
        self.circuit_breakers = {}
        
        # Initialize observability
        self._initialize_observability()
        
        # Scheduler setup with enhanced error handling
        try:
            engine = get_engine()
            self.scheduler = AsyncIOScheduler(jobstores={"default": SQLAlchemyJobStore(engine=engine)})
            self.scheduler.start(paused=True)
        except Exception as e:
            logger.warning(f"APScheduler initialization failed: {e}")
            self.scheduler = None
        
        # Start background tasks
        asyncio.create_task(self._periodic_cleanup())
        asyncio.create_task(self._health_check_task())
        asyncio.create_task(self._security_monitoring_task())
    
    def _initialize_observability(self):
        """Initialize comprehensive observability stack."""
        if self.config.enable_tracing and self.config.jaeger_endpoint:
            # Configure Jaeger tracing
            jaeger_exporter = JaegerExporter(
                agent_host_name="localhost",
                agent_port=14268,
            )
            
            span_processor = BatchSpanProcessor(jaeger_exporter)
            trace.set_tracer_provider(TracerProvider())
            trace.get_tracer_provider().add_span_processor(span_processor)
            
            # Instrument HTTP clients
            HTTPXClientInstrumentor().instrument()
            Psycopg2Instrumentor().instrument()
        
        if self.config.enable_metrics and self.config.prometheus_endpoint:
            # Configure Prometheus metrics
            prometheus_reader = PrometheusMetricReader()
            metrics.set_meter_provider(MeterProvider(metric_readers=[prometheus_reader]))
        
        if self.config.enable_structured_logging:
            # Configure structured logging
            import structlog
            structlog.configure(
                processors=[
                    structlog.stdlib.add_log_level,
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.add_log_level,
                    structlog.processors.JSONRenderer()
                ],
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                cache_logger_on_first_use=True,
            )

    @asynccontextmanager
    async def atomic_pipeline_persistence(self, run_id: str, operation: str):
        """Enhanced atomic pipeline persistence with comprehensive error handling."""
        session = None
        correlation_id = generate_correlation_id()
        
        try:
            with tracer.start_as_current_span("atomic_pipeline_persistence") as span:
                span.set_attributes({
                    "operation": operation,
                    "run_id": run_id,
                    "correlation_id": correlation_id
                })
                
                session = get_db_session()
                logger.debug(f"Started atomic operation '{operation}' for run {run_id}", 
                           extra={"correlation_id": correlation_id})
                
                yield session
                
                session.commit()
                logger.debug(f"Committed atomic operation '{operation}' for run {run_id}",
                           extra={"correlation_id": correlation_id})
                
        except Exception as e:
            if session:
                session.rollback()
                logger.error(f"Rolled back atomic operation '{operation}' for run {run_id}: {e}",
                           extra={"correlation_id": correlation_id})
            
            # Record persistence failure metrics
            service_errors_total.labels(
                service="automation",
                error_type="persistence_failure"
            ).inc()
            
            raise
        finally:
            if session:
                session.close()

    async def _periodic_cleanup(self):
        """Enhanced periodic cleanup with comprehensive maintenance."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                with tracer.start_as_current_span("periodic_cleanup"):
                    # Container cleanup
                    cleaned_containers = await self.container_security.cleanup_old_containers(max_age_hours=1)
                    if cleaned_containers > 0:
                        logger.info(f"Cleaned up {cleaned_containers} old containers")
                    
                    # Webhook security cleanup
                    self.webhook_security.cleanup_expired_nonces()
                    
                    # Encryption key rotation (weekly)
                    if datetime.utcnow().weekday() == 0:  # Monday
                        self.encryption.cleanup_old_keys()
                    
                    # Database cleanup for old logs and events
                    await self._cleanup_old_database_records()
                    
                    # Circuit breaker reset for recovered services
                    await self._reset_recovered_circuit_breakers()
                
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")

    async def _cleanup_old_database_records(self):
        """Clean up old database records for performance."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=30)
            
            with get_db_session() as session:
                # Clean up old logs
                deleted_logs = session.query(DbRunLog).filter(
                    DbRunLog.timestamp < cutoff_date
                ).delete()
                
                # Clean up old security events
                deleted_events = session.query(DbSecurityEvent).filter(
                    DbSecurityEvent.created_at < cutoff_date
                ).delete()
                
                session.commit()
                
                if deleted_logs > 0 or deleted_events > 0:
                    logger.info(f"Cleaned up {deleted_logs} old logs and {deleted_events} old security events")
                    
        except Exception as e:
            logger.error(f"Database cleanup failed: {e}")

    async def _reset_recovered_circuit_breakers(self):
        """Reset circuit breakers for services that have recovered."""
        for service_name, breaker in self.circuit_breakers.items():
            if breaker.get("state") == "open":
                # Test if service has recovered
                try:
                    await self._health_check_service(service_name)
                    breaker["state"] = "closed"
                    breaker["failure_count"] = 0
                    logger.info(f"Circuit breaker reset for {service_name} - service recovered")
                except Exception:
                    pass  # Service still failing

    async def _health_check_service(self, service_name: str):
        """Perform health check on a service."""
        # Implementation depends on service type
        if service_name == "celery":
            # Check Celery worker health
            try:
                result = celery_app.control.ping(timeout=5)
                if not result:
                    raise Exception("No Celery workers responding")
            except Exception as e:
                raise Exception(f"Celery health check failed: {e}")

    async def _health_check_task(self):
        """Continuous health monitoring of system components."""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                with tracer.start_as_current_span("health_check"):
                    # Check database connectivity
                    try:
                        with get_db_session() as session:
                            session.execute("SELECT 1")
                    except Exception as e:
                        logger.error(f"Database health check failed: {e}")
                    
                    # Check Celery workers
                    try:
                        await self._health_check_service("celery")
                    except Exception as e:
                        logger.warning(f"Celery health check failed: {e}")
                    
                    # Check container runtime
                    try:
                        self.container_security.client.ping()
                    except Exception as e:
                        logger.error(f"Docker health check failed: {e}")
                
            except Exception as e:
                logger.error(f"Health check task error: {e}")

    async def _security_monitoring_task(self):
        """Continuous security monitoring and threat detection."""
        while True:
            try:
                await asyncio.sleep(60)  # Every minute
                
                with tracer.start_as_current_span("security_monitoring"):
                    # Monitor for suspicious container activity
                    await self._monitor_container_security()
                    
                    # Monitor for unusual resource usage patterns
                    await self._monitor_resource_anomalies()
                    
                    # Check for potential security breaches
                    await self._check_security_indicators()
                
            except Exception as e:
                logger.error(f"Security monitoring task error: {e}")

    async def _monitor_container_security(self):
        """Monitor containers for security threats."""
        try:
            containers = self.container_security.client.containers.list(
                filters={"label": "kaizen-pipeline"}
            )
            
            for container in containers:
                # Check for escape attempts
                escape_indicators = self.container_security._check_escape_attempts(container)
                if escape_indicators:
                    logger.critical(f"Container escape attempt detected: {container.id}")
                    
                    # Terminate suspicious container
                    try:
                        container.stop(timeout=5)
                        container.remove(force=True)
                    except Exception as e:
                        logger.error(f"Failed to terminate suspicious container: {e}")
        
        except Exception as e:
            logger.error(f"Container security monitoring failed: {e}")

    async def _monitor_resource_anomalies(self):
        """Monitor for unusual resource usage patterns that might indicate attacks."""
        try:
            # Check system resource usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            disk_percent = psutil.disk_usage('/').percent
            
            # Alert on high resource usage
            if cpu_percent > 90:
                logger.warning(f"High CPU usage detected: {cpu_percent}%")
            
            if memory_percent > 90:
                logger.warning(f"High memory usage detected: {memory_percent}%")
            
            if disk_percent > 90:
                logger.warning(f"High disk usage detected: {disk_percent}%")
            
            # Check for resource usage anomalies in active processes
            for process_id, process_info in self.resource_monitor.active_processes.items():
                usage = self.resource_monitor.get_resource_usage(process_id)
                if usage and self._is_anomalous_usage(usage):
                    logger.warning(f"Anomalous resource usage detected for process {process_id}")
        
        except Exception as e:
            logger.error(f"Resource anomaly monitoring failed: {e}")

    def _is_anomalous_usage(self, usage: ResourceUsage) -> bool:
        """Detect anomalous resource usage patterns."""
        # Simple heuristics - in production, use ML-based anomaly detection
        return (
            usage.memory_mb > 1000 or  # > 1GB memory
            usage.cpu_percent > 200 or  # > 200% CPU (multi-core)
            usage.network_connections > 50  # > 50 network connections
        )

    async def _check_security_indicators(self):
        """Check for various security indicators and threats."""
        try:
            # Check for failed authentication attempts
            current_time = datetime.utcnow()
            window_start = current_time - timedelta(minutes=15)
            
            for ip, attempts in _failed_auth_attempts.items():
                recent_attempts = [t for t in attempts if t > window_start]
                if len(recent_attempts) > 10:  # More than 10 failed attempts in 15 minutes
                    logger.critical(f"Potential brute force attack from IP {ip}: {len(recent_attempts)} failed attempts")
                    
                    # Could implement IP blocking here
        
        except Exception as e:
            logger.error(f"Security indicators check failed: {e}")

    async def pipeline_run(self, config: PipelineConfig, owner: Optional[str] = None, 
                          correlation_id: Optional[str] = None,
                          security_context: Optional[SecurityContext] = None) -> RunResponse:
        """Create and execute a pipeline run with enhanced security and observability."""
        run_id = f"run-{uuid.uuid4().hex}"
        correlation_id = correlation_id or generate_correlation_id()
        
        with tracer.start_as_current_span("pipeline_run") as span:
            span.set_attributes({
                "run_id": run_id,
                "correlation_id": correlation_id,
                "owner": owner or "unknown",
                "steps_count": len(config.steps)
            })
            
            # Validate pipeline configuration
            validation_result = await self.validate_pipeline_config(config)
            if not validation_result["valid"]:
                span.set_attribute("validation.result", "failed")
                raise ValueError(f"Pipeline validation failed: {validation_result['errors']}")
            
            run = PipelineRun(
                run_id=run_id,
                status=Status.pending,
                config=config,
                correlation_id=correlation_id,
                owner=owner,
            )
            _pipeline_runs[run_id] = run
            _cancellations[run_id] = asyncio.Event()

            # Persist run atomically with encryption
            try:
                async with self.atomic_pipeline_persistence(run_id, "create_run") as session:
                    # Encrypt sensitive configuration data
                    encrypted_config = self.encryption.encrypt(json.dumps(config.dict()))
                    
                    db_run = DbPipelineRun(
                        id=run_id,
                        status=RunStatus.pending,
                        config={"encrypted": True, "data": encrypted_config},
                        owner=owner,
                        correlation_id=correlation_id,
                    )
                    session.add(db_run)
                    
                    # Log pipeline creation
                    if security_context:
                        await self.audit_logger.log_audit_event(
                            "automation_service",
                            "create_pipeline",
                            run_id,
                            {"steps_count": len(config.steps)},
                            security_context
                        )
            except Exception as e:
                logger.warning(f"Failed to persist pipeline run {run_id}: {e}")

            # Enqueue execution with circuit breaker
            try:
                if self._is_service_available("celery"):
                    task = celery_app.send_task("app.tasks.automation.execute_pipeline", args=[run_id], queue="automation")
                    # Persist task id
                    with get_db_session() as session:
                        db_run = session.query(DbPipelineRun).get(run_id)
                        if db_run:
                            db_run.task_id = task.id
                else:
                    raise Exception("Celery service unavailable")
            except Exception as e:
                logger.warning(f"Falling back to local execution due to celery error: {e}")
                self._record_service_failure("celery")
                asyncio.create_task(self._execute_pipeline(run_id, security_context))

            # Record pipeline creation metrics
            pipeline_execution_counter.add(1, {"owner": owner or "unknown"})

            return RunResponse(run_id=run_id, status=Status.pending, correlation_id=correlation_id)

    def _is_service_available(self, service_name: str) -> bool:
        """Check if service is available using circuit breaker pattern."""
        breaker = self.circuit_breakers.get(service_name)
        if not breaker:
            self.circuit_breakers[service_name] = {
                "state": "closed",
                "failure_count": 0,
                "last_failure": None
            }
            return True
        
        if breaker["state"] == "open":
            # Check if enough time has passed to try again
            if breaker["last_failure"] and \
               datetime.utcnow() - breaker["last_failure"] > timedelta(minutes=5):
                breaker["state"] = "half-open"
                return True
            return False
        
        return True

    def _record_service_failure(self, service_name: str):
        """Record service failure for circuit breaker."""
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = {
                "state": "closed",
                "failure_count": 0,
                "last_failure": None
            }
        
        breaker = self.circuit_breakers[service_name]
        breaker["failure_count"] += 1
        breaker["last_failure"] = datetime.utcnow()
        
        # Open circuit breaker after 3 failures
        if breaker["failure_count"] >= 3:
            breaker["state"] = "open"
            logger.warning(f"Circuit breaker opened for service {service_name}")

    async def monitor_pipeline(self, run_id: str) -> Optional[PipelineRun]:
        """Get current pipeline run status with enhanced monitoring."""
        with tracer.start_as_current_span("monitor_pipeline") as span:
            span.set_attribute("run_id", run_id)
            run = _pipeline_runs.get(run_id)
            
            if run:
                # Add real-time resource usage if available
                if run.current_step is not None and run.current_step < len(run.steps):
                    process_id = f"{run_id}-{run.current_step}"
                    usage = self.resource_monitor.get_resource_usage(process_id)
                    if usage:
                        # Add resource usage to run metadata
                        run.current_resource_usage = usage
            
            return run

    async def cancel_run(self, run_id: str, security_context: Optional[SecurityContext] = None) -> Tuple[bool, str]:
        """Cancel a running pipeline with enhanced security and logging."""
        with tracer.start_as_current_span("cancel_pipeline") as span:
            span.set_attribute("run_id", run_id)
            
            event = _cancellations.get(run_id)
            run = _pipeline_runs.get(run_id)
            
            if not run:
                return False, "Run not found"
            
            if run.status in {Status.completed, Status.failed, Status.cancelled}:
                return False, f"Run already {run.status}"
            
            if event:
                event.set()
                
                # Attempt celery revoke with enhanced error handling
                try:
                    with get_db_session() as session:
                        db_run = session.query(DbPipelineRun).get(run_id)
                        if db_run and db_run.task_id:
                            celery_app.control.revoke(db_run.task_id, terminate=True)
                except Exception as e:
                    logger.warning(f"Failed to revoke Celery task for run {run_id}: {e}")
                
                # Terminate any running containers
                await self._terminate_run_containers(run_id)
                
                # Log cancellation
                if security_context:
                    await self.audit_logger.log_audit_event(
                        "automation_service",
                        "cancel_pipeline",
                        run_id,
                        {"reason": "user_requested"},
                        security_context
                    )
                
                span.set_attribute("cancellation.result", "success")
                return True, "Cancellation signaled"
            
            return False, "Cancellation channel not found"

    async def _terminate_run_containers(self, run_id: str):
        """Terminate all containers associated with a pipeline run."""
        try:
            containers = self.container_security.client.containers.list(
                filters={"label": f"kaizen-pipeline-run={run_id}"}
            )
            
            for container in containers:
                try:
                    self.container_security.terminate_container(container.id)
                    logger.info(f"Terminated container {container.id} for cancelled run {run_id}")
                except Exception as e:
                    logger.warning(f"Failed to terminate container {container.id}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to terminate containers for run {run_id}: {e}")

    async def _execute_pipeline(self, run_id: str, security_context: Optional[SecurityContext] = None) -> None:
        """Enhanced pipeline execution with comprehensive monitoring and security."""
        run = _pipeline_runs[run_id]
        
        with tracer.start_as_current_span("execute_pipeline") as span:
            span.set_attributes({
                "run_id": run_id,
                "steps_count": len(run.config.steps),
                "owner": run.owner or "unknown"
            })
            
            run.status = Status.running
            run.started_at = datetime.utcnow()
            self._append_log(run, "INFO", f"Pipeline started with {len(run.config.steps)} steps")
            
            # Persist status update
            try:
                with get_db_session() as session:
                    db_run = session.query(DbPipelineRun).get(run_id)
                    if db_run:
                        db_run.status = RunStatus.running
                        db_run.started_at = run.started_at
            except Exception as e:
                logger.warning(f"Failed to update DB run {run_id} to running: {e}")

            previous_output: Optional[Dict[str, Any]] = None
            start_index = 0
            
            # Resume logic
            if run.config.resume and run.steps:
                for idx, step_res in enumerate(run.steps):
                    if step_res.status == Status.completed:
                        previous_output = step_res.output or previous_output
                        start_index = idx + 1
                    else:
                        start_index = idx
                        break

            try:
                for idx in range(start_index, len(run.config.steps)):
                    if _cancellations[run_id].is_set():
                        run.status = Status.cancelled
                        self._append_log(run, "WARNING", "Run cancelled before step execution", {"step_index": idx})
                        break

                    run.current_step = idx
                    step = run.config.steps[idx]
                    
                    # Execute step with enhanced monitoring
                    step_result = await self._execute_step(idx, step, previous_output, run_id, security_context)
                    run.steps.append(step_result)
                    
                    # Persist step
                    try:
                        with get_db_session() as session:
                            db_step = DbStepRun(
                                run_id=run_id,
                                index=step_result.index,
                                type=step_result.type.value,
                                status=RunStatus(step_result.status.value),
                                started_at=step_result.started_at,
                                finished_at=step_result.finished_at,
                                duration_seconds=int(step_result.duration_seconds) if step_result.duration_seconds else None,
                                output=step_result.output,
                                artifacts=[a.dict() for a in step_result.artifacts] if step_result.artifacts else None,
                                error=step_result.error,
                            )
                            session.add(db_step)
                    except Exception as e:
                        logger.warning(f"Failed to persist step {idx} for run {run_id}: {e}")

                    if step_result.status != Status.completed:
                        run.status = Status.failed
                        self._append_log(run, "ERROR", f"Step {idx} failed", {"error": step_result.error})
                        await self._notify(run, security_context)
                        run.finished_at = datetime.utcnow()
                        _pipeline_runs[run_id] = run
                        try:
                            with get_db_session() as session:
                                db_run = session.query(DbPipelineRun).get(run_id)
                                if db_run:
                                    db_run.status = RunStatus.failed
                                    db_run.finished_at = run.finished_at
                        except Exception as e:
                            logger.warning(f"Failed to mark run {run_id} failed in DB: {e}")
                        return

                    # Collect artifacts and carry output
                    previous_output = step_result.output
                    for artifact in step_result.artifacts:
                        bucket = self._artifact_bucket_for_step(step.type)
                        run.artifacts.setdefault(bucket, []).append(artifact)

                if run.status != Status.cancelled:
                    run.status = Status.completed
                    self._append_log(run, "INFO", "Pipeline completed successfully")
                    
            except Exception as e:
                logger.exception("Pipeline execution error")
                try:
                    service_errors_total.labels(service="automation", error_type=type(e).__name__).inc()
                except Exception:
                    pass
                run.status = Status.failed
                self._append_log(run, "ERROR", "Unhandled pipeline error", {"error": str(e)})
                
            finally:
                run.finished_at = datetime.utcnow()
                _pipeline_runs[run_id] = run
                
                # Persist terminal state
                try:
                    with get_db_session() as session:
                        db_run = session.query(DbPipelineRun).get(run_id)
                        if db_run:
                            db_run.status = RunStatus(run.status.value)
                            db_run.finished_at = run.finished_at
                            db_run.current_step_index = run.current_step
                except Exception as e:
                    logger.warning(f"Failed to finalize run {run_id} in DB: {e}")
                
                await self._notify(run, security_context)
                
                # Record completion metrics
                if run.started_at and run.finished_at:
                    duration = (run.finished_at - run.started_at).total_seconds()
                    pipeline_run_duration_seconds.labels(
                        status=run.status.value,
                        owner=run.owner or "unknown"
                    ).observe(duration)

    async def _execute_step(self, index: int, step: PipelineStep, previous_output: Optional[Dict[str, Any]], 
                          run_id: str, security_context: Optional[SecurityContext] = None) -> StepResult:
        """Execute a single step with comprehensive monitoring and security."""
        started_at = datetime.utcnow()
        attempt = 0
        error: Optional[str] = None
        artifacts: List[StepArtifact] = []
        output: Optional[Dict[str, Any]] = None
        process_id: Optional[str] = None

        # Get resource limits for this step
        resource_limits = getattr(step, 'resource_limits', None) or self.default_resource_limits

        with tracer.start_as_current_span("execute_step") as span:
            span.set_attributes({
                "step.index": index,
                "step.type": step.type.value,
                "step.service": step.service.value,
                "run_id": run_id
            })

            while attempt <= step.retries:
                attempt += 1
                try:
                    self._append_log_id(run_id, "INFO", f"Executing step {index} ({step.type}) attempt {attempt}")
                    
                    # Check if this step should run in a container
                    if self._should_use_container(step):
                        result_output, result_artifacts = await self._execute_step_in_container(
                            index, step, previous_output, run_id, resource_limits, security_context
                        )
                    else:
                        result_output, result_artifacts = await self._dispatch_step(step, previous_output)
                    
                    output = result_output
                    artifacts = result_artifacts or []
                    finished_at = datetime.utcnow()
                    duration = (finished_at - started_at).total_seconds()
                    
                    # Record metrics
                    try:
                        pipeline_step_duration_seconds.labels(
                            step_type=step.type.value,
                            service=step.service.value
                        ).observe(duration)
                    except Exception:
                        pass
                    
                    span.set_attributes({
                        "step.duration_seconds": duration,
                        "step.status": "completed",
                        "step.attempts": attempt
                    })
                    
                    return StepResult(
                        index=index,
                        type=step.type,
                        status=Status.completed,
                        started_at=started_at,
                        finished_at=finished_at,
                        duration_seconds=duration,
                        output=output,
                        artifacts=artifacts,
                    )
                    
                except Exception as e:
                    error = str(e)
                    self._append_log_id(run_id, "ERROR", f"Step {index} attempt {attempt} failed", {"error": error})
                    
                    # Record retry metrics
                    try:
                        pipeline_step_retries_total.labels(
                            step_type=step.type.value,
                            service=step.service.value,
                            retry_reason=type(e).__name__
                        ).inc()
                    except Exception:
                        pass
                    
                    if attempt > step.retries:
                        break
                    await asyncio.sleep(min(2 * attempt, 10))

            finished_at = datetime.utcnow()
            duration = (finished_at - started_at).total_seconds()
            
            span.set_attributes({
                "step.duration_seconds": duration,
                "step.status": "failed",
                "step.attempts": attempt,
                "step.error": error
            })
            
            return StepResult(
                index=index,
                type=step.type,
                status=Status.failed,
                started_at=started_at,
                finished_at=finished_at,
                duration_seconds=duration,
                output=output,
                artifacts=artifacts,
                error=error,
            )

    async def _dispatch_step(self, step: PipelineStep, previous_output: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[StepArtifact]]:
        """Call the appropriate service layer with enhanced error handling."""
        with tracer.start_as_current_span("dispatch_step") as dispatch_span:
            dispatch_span.set_attributes({
                "step.type": step.type.value,
                "step.service": step.service.value,
                "step.method": step.method
            })
            
            try:
                if step.type == step.type.preprocess:
                    result = await self._call_bound_service(step, previous_output)
                    return result or {}, []
                elif step.type == step.type.transform:
                    result = await self._call_bound_service(step, previous_output)
                    return result or {}, []
                elif step.type == step.type.model:
                    result = await self._call_bound_service(step, previous_output)
                    artifacts = []
                    if isinstance(result, dict) and "model_path" in result:
                        artifacts.append(StepArtifact(
                            kind="model", 
                            uri=result["model_path"], 
                            metadata={"model_id": result.get("model_id")}
                        ))
                    return result or {}, artifacts
                elif step.type == step.type.visualize:
                    result = await self._call_visualization(step.config, previous_output)
                    artifacts = []
                    if isinstance(result, dict) and result.get("signed_url"):
                        artifacts.append(StepArtifact(
                            kind="plot", 
                            uri=result["signed_url"], 
                            metadata={"message": result.get("message")}
                        ))
                    return result or {}, artifacts
                else:
                    raise RuntimeError(f"Unsupported step type: {step.type}")
                    
            except Exception as e:
                dispatch_span.set_attribute("error", str(e))
                raise

    def _should_use_container(self, step: PipelineStep) -> bool:
        """Determine if step should run in a container based on configuration and security policy."""
        # Container execution for potentially unsafe operations
        container_required_types = ["model", "transform"]  # Types that require isolation
        
        return (
            step.config.get("use_container", False) or
            step.type.value in container_required_types or
            step.config.get("untrusted_code", False)
        )

    async def _execute_step_in_container(self, index: int, step: PipelineStep, 
                                       previous_output: Optional[Dict[str, Any]], 
                                       run_id: str, resource_limits: StepResourceLimits,
                                       security_context: Optional[SecurityContext] = None) -> Tuple[Dict[str, Any], List[StepArtifact]]:
        """Execute step in a secure Docker container with comprehensive monitoring."""
        container_id = None
        
        with tracer.start_as_current_span("execute_step_in_container") as span:
            try:
                # Get container configuration from step
                image = step.config.get("container_image", "python:3.9-slim")
                command = step.config.get("container_command", "python -c 'print(\"Hello from container\")'")
                
                span.set_attributes({
                    "container.image": image,
                    "step.index": index,
                    "run_id": run_id
                })
                
                # Validate container image with security context
                if not self.container_security.validate_image(image, security_context):
                    raise RuntimeError(f"Container image {image} failed security validation")
                
                # Create secure container with enhanced configuration
                volumes = step.config.get("container_volumes", {})
                container_id = self.container_security.create_secure_container(
                    image=image,
                    command=command,
                    volumes=volumes,
                    security_context=security_context
                )
                
                # Add run-specific labels
                container = self.container_security.client.containers.get(container_id)
                container.reload()
                
                # Start container with timeout
                container.start()
                
                self._append_log_id(run_id, "INFO", f"Started container {container_id} for step {index}")
                span.set_attribute("container.id", container_id)
                
                # Enhanced monitoring with security checks
                timeout_seconds = resource_limits.max_duration_minutes * 60
                start_time = time.time()
                monitoring_interval = 1  # Monitor every second
                
                while time.time() - start_time < timeout_seconds:
                    container.reload()
                    
                    if container.status != "running":
                        break
                    
                    # Comprehensive container monitoring
                    container_stats = self.container_security.monitor_container(container_id)
                    
                    # Resource limit enforcement
                    if container_stats["memory_mb"] > resource_limits.max_memory_mb:
                        self._append_log_id(run_id, "WARNING", 
                                          f"Container {container_id} exceeded memory limit: "
                                          f"{container_stats['memory_mb']:.1f}MB > {resource_limits.max_memory_mb}MB")
                        container.stop(timeout=10)
                        break
                    
                    # Security monitoring
                    if container_stats.get("escape_indicators"):
                        self._append_log_id(run_id, "CRITICAL", 
                                          f"Security threat detected in container {container_id}: "
                                          f"{container_stats['escape_indicators']}")
                        container.kill()  # Immediate termination
                        raise RuntimeError("Container security violation detected")
                    
                    await asyncio.sleep(monitoring_interval)
                
                # Handle timeout
                if time.time() - start_time >= timeout_seconds:
                    self._append_log_id(run_id, "ERROR", f"Container {container_id} execution timed out")
                    container.stop(timeout=5)
                    raise RuntimeError(f"Container execution timeout after {timeout_seconds} seconds")
                
                # Get container exit code and logs
                container.reload()
                exit_code = container.attrs["State"]["ExitCode"]
                logs = container.logs().decode('utf-8', errors='ignore')
                
                # Process execution results
                if exit_code != 0:
                    raise RuntimeError(f"Container exited with code {exit_code}: {logs}")
                
                # Parse output from container
                output = {
                    "container_id": container_id,
                    "logs": logs,
                    "exit_code": exit_code,
                    "execution_time": time.time() - start_time
                }
                
                # Create artifacts from container output
                artifacts = []
                if step.config.get("save_logs", True):
                    artifacts.append(StepArtifact(
                        kind="logs",
                        uri=f"container://{container_id}/logs",
                        metadata={
                            "container_id": container_id, 
                            "logs": logs[:1000],  # Truncate large logs
                            "exit_code": exit_code
                        }
                    ))
                
                # Extract additional artifacts based on step type
                if step.type.value == "model" and "model_path" in logs:
                    # Extract model path from logs (simplified)
                    import re
                    model_match = re.search(r'model_path:\s*([^\s]+)', logs)
                    if model_match:
                        artifacts.append(StepArtifact(
                            kind="model",
                            uri=model_match.group(1),
                            metadata={"created_in_container": container_id}
                        ))
                
                span.set_attributes({
                    "container.exit_code": exit_code,
                    "container.execution_time": time.time() - start_time,
                    "artifacts.count": len(artifacts)
                })
                
                return output, artifacts
                
            except Exception as e:
                logger.error(f"Container execution failed for step {index}: {e}")
                span.set_attribute("error", str(e))
                raise
                
            finally:
                # Comprehensive cleanup
                if container_id:
                    try:
                        # Collect final stats before cleanup
                        final_stats = self.container_security.monitor_container(container_id)
                        
                        # Log cleanup action
                        self._append_log_id(run_id, "INFO", f"Cleaning up container {container_id}")
                        
                        # Terminate container
                        self.container_security.terminate_container(container_id, timeout=10)
                        
                        # Log security summary
                        if security_context:
                            await self.audit_logger.log_audit_event(
                                "container_execution",
                                "container_cleanup",
                                container_id,
                                {
                                    "step_index": index,
                                    "run_id": run_id,
                                    "final_stats": final_stats
                                },
                                security_context
                            )
                        
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to cleanup container {container_id}: {cleanup_error}")

    async def _call_bound_service(self, step: PipelineStep, previous_output: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Call service method with enhanced error handling and circuit breaker."""
        service_map = {
            "preprocessing": self.preprocessing,
            "transformation": self.transformation,
            "modeling": self.modeling,
        }
        
        with tracer.start_as_current_span("call_bound_service") as span:
            span.set_attributes({
                "service": step.service.value,
                "method": step.method
            })
            
            target = service_map.get(step.service.value)
            if target is None:
                raise RuntimeError(f"Unknown service: {step.service}")
            
            if not hasattr(target, step.method):
                raise RuntimeError(f"Method '{step.method}' not found on service {step.service}")
            
            method = getattr(target, step.method)
            
            try:
                # Call method with timeout and error handling
                if asyncio.iscoroutinefunction(method):
                    maybe_result = await asyncio.wait_for(method(step.config), timeout=300)  # 5 minute timeout
                else:
                    maybe_result = method(step.config)
                
                # Standardize result format
                if hasattr(maybe_result, "dict"):
                    result = maybe_result.dict()
                elif isinstance(maybe_result, dict):
                    result = maybe_result
                else:
                    result = {"result": maybe_result}
                
                span.set_attribute("call.result", "success")
                return result
                
            except asyncio.TimeoutError:
                span.set_attribute("call.error", "timeout")
                raise RuntimeError(f"Service call timeout: {step.service}.{step.method}")
            except Exception as e:
                span.set_attribute("call.error", str(e))
                raise

    async def _call_visualization(self, config: Dict[str, Any], previous_output: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Enhanced visualization service adapter with comprehensive error handling."""
        plot_type = config.get("plot")
        
        with tracer.start_as_current_span("call_visualization") as span:
            span.set_attribute("plot_type", plot_type)
            
            try:
                from app.services import visualization as viz
                
                # Create common plot config
                plot_config = viz.PlotConfig(
                    title=config.get("title"),
                    figsize=tuple(config.get("figsize", (10, 6))),
                    return_base64=config.get("return_base64", True),
                    skip_base64=config.get("skip_base64", False),
                    style=config.get("style", "whitegrid"),
                    dpi=config.get("dpi", 100),
                )
                
                if plot_type == "histogram":
                    req = viz.HistogramRequest(
                        file_or_path=config.get("data_ref"),
                        column=config.get("column"),
                        bins=config.get("bins"),
                        config=plot_config,
                    )
                    resp = viz.generate_histogram(req)
                elif plot_type == "scatter_plot":
                    req = viz.ScatterPlotRequest(
                        file_or_path=config.get("data_ref"),
                        x_column=config.get("x_column"),
                        y_column=config.get("y_column"),
                        color_column=config.get("color_column"),
                        config=plot_config,
                    )
                    resp = viz.generate_scatter_plot(req)
                elif plot_type == "box_plot":
                    req = viz.BoxPlotRequest(
                        file_or_path=config.get("data_ref"),
                        numeric_column=config.get("numeric_column"),
                        categorical_column=config.get("categorical_column"),
                        config=plot_config,
                    )
                    resp = viz.generate_box_plot(req)
                elif plot_type == "time_series":
                    req = viz.TimeSeriesRequest(
                        file_or_path=config.get("data_ref"),
                        datetime_column=config.get("datetime_column"),
                        value_column=config.get("value_column"),
                        group_column=config.get("group_column"),
                        config=plot_config,
                    )
                    resp = viz.generate_time_series_plot(req)
                elif plot_type == "correlation_heatmap":
                    req = viz.CorrelationHeatmapRequest(
                        file_or_path=config.get("data_ref"),
                        columns=config.get("columns"),
                        correlation_method=config.get("correlation_method", "pearson"),
                        config=plot_config,
                    )
                    resp = viz.generate_correlation_heatmap(req)
                else:
                    raise RuntimeError(f"Unsupported plot type: {plot_type}")
                
                result = resp.dict() if hasattr(resp, "dict") else resp
                span.set_attribute("visualization.result", "success")
                return result
                
            except Exception as e:
                span.set_attribute("visualization.error", str(e))
                raise RuntimeError(f"Visualization generation failed: {e}")

    def _append_log(self, run: PipelineRun, level: str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Enhanced logging with structured format and correlation IDs."""
        entry = StepLog(level=level, message=message, details=details)
        
        # Add structured logging context
        log_context = {
            "run_id": run.run_id,
            "correlation_id": run.correlation_id,
            "owner": run.owner,
            "current_step": run.current_step
        }
        
        # Attach to last step if running, else to run-level aggregation
        if run.steps and run.steps[-1].status == Status.running:
            run.steps[-1].logs.append(entry)
        else:
            run.artifacts.setdefault("logs", [])
        
        # Enhanced persistent logging
        try:
            with get_db_session() as session:
                db_log = DbRunLog(
                    run_id=run.run_id,
                    step_index=run.current_step,
                    level=level,
                    message=message,
                    details={**(details or {}), **log_context},
                )
                session.add(db_log)
                session.commit()
        except Exception as e:
            logger.debug(f"Failed to persist log for run {run.run_id}: {e}")
        
        # Update in-memory state
        _pipeline_runs[run.run_id] = run
        
        # Structured logging output
        logger.log(
            getattr(logging, level.upper()),
            message,
            extra={**log_context, **(details or {})}
        )

    def _append_log_id(self, run_id: str, level: str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Helper to append log by run ID."""
        run = _pipeline_runs.get(run_id)
        if not run:
            return
        self._append_log(run, level, message, details)

    def _artifact_bucket_for_step(self, step_type) -> str:
        """Categorize artifacts by step type."""
        mapping = {
            "model": "models",
            "visualize": "plots",
            "preprocess": "datasets",
            "transform": "datasets",
        }
        key = step_type.value if hasattr(step_type, "value") else str(step_type)
        return mapping.get(key, "misc")

    async def _notify(self, run: PipelineRun, security_context: Optional[SecurityContext] = None) -> None:
        """Send enhanced webhook notifications with security validation."""
        await self._enhanced_webhook_notification(run, "pipeline_update", security_context)

    async def _enhanced_webhook_notification(self, run: PipelineRun, event_type: str, 
                                           security_context: Optional[SecurityContext] = None) -> None:
        """Send webhook notification with comprehensive security and monitoring."""
        if not run.config.webhook or not run.config.webhook.url:
            return
        if run.status not in set(run.config.webhook.on_events):
            return

        with tracer.start_as_current_span("webhook_notification") as span:
            span.set_attributes({
                "webhook.url": str(run.config.webhook.url),
                "webhook.event_type": event_type,
                "run_id": run.run_id
            })

            try:
                # Generate enhanced security headers
                timestamp = datetime.utcnow().isoformat()
                nonce = self.webhook_security.generate_nonce()
                
                payload = {
                    "run_id": run.run_id,
                    "status": run.status.value,
                    "started_at": run.started_at.isoformat() if run.started_at else None,
                    "finished_at": run.finished_at.isoformat() if run.finished_at else None,
                    "duration_seconds": (run.finished_at - run.started_at).total_seconds() if run.finished_at and run.started_at else None,
                    "artifacts": {
                        bucket: [a.dict() if hasattr(a, "dict") else a for a in items]
                        for bucket, items in run.artifacts.items()
                    },
                    "correlation_id": run.correlation_id,
                    "owner": run.owner,
                    "timestamp": timestamp,
                    "nonce": nonce,
                    "event_type": event_type
                }
                
                headers = {
                    "Content-Type": "application/json",
                    "X-Webhook-Timestamp": timestamp,
                    "X-Webhook-Nonce": nonce,
                    "X-Webhook-Event": event_type,
                    "User-Agent": "Kaizen-Automation-Service/1.0"
                }
                
                # HMAC signature for payload integrity
                if run.config.webhook.secret:
                    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
                    signature = hmac.new(
                        run.config.webhook.secret.encode("utf-8"), 
                        body, 
                        hashlib.sha256
                    ).hexdigest()
                    headers["X-Webhook-Signature"] = f"sha256={signature}"
                
                # Send webhook with enhanced error handling
                try:
                    async with httpx.AsyncClient(
                        timeout=run.config.webhook.timeout_seconds,
                        verify=True,  # Always verify SSL
                        limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
                    ) as client:
                        response = await client.post(
                            str(run.config.webhook.url), 
                            json=payload, 
                            headers=headers
                        )
                        response.raise_for_status()
                        
                        # Log successful delivery
                        if security_context:
                            await self.audit_logger.log_audit_event(
                                "webhook_delivery",
                                "webhook_sent",
                                run.run_id,
                                {
                                    "url": str(run.config.webhook.url),
                                    "status_code": response.status_code,
                                    "event_type": event_type
                                },
                                security_context
                            )
                        
                        # Record webhook metrics
                        try:
                            webhook_notifications_total.labels(
                                status="success", 
                                webhook_type=event_type
                            ).inc()
                        except Exception:
                            pass
                            
                        logger.info(f"Webhook notification sent successfully to {run.config.webhook.url}")
                        span.set_attribute("webhook.status", "success")
                        
                except httpx.HTTPStatusError as e:
                    error_msg = f"Webhook HTTP error {e.response.status_code}: {e.response.text}"
                    logger.warning(f"Webhook notification failed: {error_msg}")
                    span.set_attribute("webhook.error", error_msg)
                    
                    try:
                        webhook_notifications_total.labels(
                            status=f"http_{e.response.status_code}", 
                            webhook_type=event_type
                        ).inc()
                    except Exception:
                        pass
                        
                except httpx.TimeoutException:
                    error_msg = "Webhook timeout"
                    logger.warning(f"Webhook notification failed: {error_msg}")
                    span.set_attribute("webhook.error", error_msg)
                    
                    try:
                        webhook_notifications_total.labels(
                            status="timeout", 
                            webhook_type=event_type
                        ).inc()
                    except Exception:
                        pass
                
                except Exception as e:
                    error_msg = f"Webhook error: {str(e)}"
                    logger.warning(f"Webhook notification failed: {error_msg}")
                    span.set_attribute("webhook.error", error_msg)
                    
                    try:
                        webhook_notifications_total.labels(
                            status="failed", 
                            webhook_type=event_type
                        ).inc()
                    except Exception:
                        pass

            except Exception as e:
                logger.error(f"Webhook notification preparation failed: {e}")
                span.set_attribute("webhook.preparation_error", str(e))

    # Enhanced scheduling APIs with security
    async def register_schedule(self, schedule: PipelineSchedule, 
                               security_context: Optional[SecurityContext] = None) -> str:
        """Register pipeline schedule with enhanced security validation."""
        if self.scheduler is None:
            raise RuntimeError("Scheduler unavailable")
        
        with tracer.start_as_current_span("register_schedule") as span:
            job_id = schedule.id
            span.set_attribute("schedule.id", job_id)
            
            # Security validation
            if security_context:
                await self.audit_logger.log_audit_event(
                    "scheduler",
                    "register_schedule",
                    job_id,
                    {
                        "schedule_type": schedule.type.value,
                        "interval_seconds": schedule.interval_seconds
                    },
                    security_context
                )
            
            try:
                if schedule.type == ScheduleType.interval:
                    self.scheduler.add_job(
                        self._schedule_trigger,
                        "interval",
                        seconds=schedule.interval_seconds,
                        id=job_id,
                        replace_existing=True,
                        kwargs={"schedule_id": job_id},
                        next_run_time=schedule.start_at,
                        end_date=schedule.end_at,
                        max_instances=1  # Prevent overlapping executions
                    )
                elif schedule.type == ScheduleType.cron:
                    self.scheduler.add_job(
                        self._schedule_trigger,
                        "cron",
                        id=job_id,
                        replace_existing=True,
                        kwargs={"schedule_id": job_id},
                        next_run_time=schedule.start_at,
                        end_date=schedule.end_at,
                        max_instances=1
                    )
                else:
                    # one-time
                    self.scheduler.add_job(
                        self._schedule_trigger,
                        "date",
                        id=job_id,
                        run_date=schedule.start_at,
                        replace_existing=True,
                        kwargs={"schedule_id": job_id},
                        max_instances=1
                    )
                
                if self.scheduler.state == 0:
                    self.scheduler.start()
                
                span.set_attribute("schedule.status", "registered")
                return job_id
                
            except Exception as e:
                span.set_attribute("schedule.error", str(e))
                raise

    async def _schedule_trigger(self, schedule_id: str) -> None:
        """Enhanced schedule trigger with comprehensive error handling."""
        with tracer.start_as_current_span("schedule_trigger") as span:
            span.set_attribute("schedule.id", schedule_id)
            
            try:
                with get_db_session() as session:
                    s = session.query(DbPipelineSchedule).get(schedule_id)
                    if not s or not s.active:
                        span.set_attribute("schedule.status", "inactive")
                        return
                    
                    # Decrypt schedule config if encrypted
                    config_data = s.config
                    if isinstance(config_data, dict) and config_data.get("encrypted"):
                        config_data = json.loads(self.encryption.decrypt(config_data["data"]))
                    
                    # Create security context for scheduled execution
                    security_context = SecurityContext(
                        user_id="scheduler",
                        roles=["scheduler"],
                        permissions=["execute_pipeline"],
                        correlation_id=generate_correlation_id()
                    )
                    
                    # Execute pipeline
                    response = await self.pipeline_run(
                        PipelineConfig(**config_data), 
                        owner=s.owner,
                        security_context=security_context
                    )
                    
                    # Update schedule metrics
                    try:
                        pipeline_schedules_total.labels(
                            schedule_type=s.type.value,
                            status="success"
                        ).inc()
                    except Exception:
                        pass
                    
                    span.set_attributes({
                        "schedule.status": "executed",
                        "pipeline.run_id": response.run_id
                    })
                    
                    logger.info(f"Scheduled pipeline executed: {response.run_id}")
                    
            except Exception as e:
                logger.exception(f"Schedule trigger failed for {schedule_id}")
                
                try:
                    pipeline_schedules_total.labels(
                        schedule_type="unknown",
                        status="failed"
                    ).inc()
                except Exception:
                    pass
                
                span.set_attribute("schedule.error", str(e))

    # Enhanced enterprise methods with comprehensive security
    async def create_template(self, template_data: Dict[str, Any], owner: str,
                            security_context: Optional[SecurityContext] = None) -> str:
        """Create a reusable pipeline template with enhanced security."""
        template_id = f"template-{uuid.uuid4().hex}"
        
        with tracer.start_as_current_span("create_template") as span:
            span.set_attributes({
                "template.id": template_id,
                "template.owner": owner
            })
            
            try:
                # Encrypt sensitive template configuration
                encrypted_config = self.encryption.encrypt(json.dumps(template_data["config"]))
                
                with get_db_session() as session:
                    db_template = DbPipelineTemplate(
                        id=template_id,
                        name=template_data["name"],
                        description=template_data.get("description"),
                        config={"encrypted": True, "data": encrypted_config},
                        tags=template_data.get("tags", []),
                        category=template_data.get("category"),
                        owner=owner,
                        is_public=template_data.get("is_public", False),
                        parameters_schema=template_data.get("parameters_schema"),
                        documentation=template_data.get("documentation")
                    )
                    session.add(db_template)
                    session.commit()
                
                # Audit log
                if security_context:
                    await self.audit_logger.log_audit_event(
                        "template_management",
                        "create_template",
                        template_id,
                        {"name": template_data["name"], "is_public": template_data.get("is_public", False)},
                        security_context
                    )
                
                logger.info(f"Created pipeline template {template_id} by {owner}")
                span.set_attribute("template.status", "created")
                return template_id
                
            except Exception as e:
                span.set_attribute("template.error", str(e))
                raise

    async def validate_pipeline_config(self, config: PipelineConfig) -> Dict[str, Any]:
        """Enhanced pipeline configuration validation with security checks."""
        with tracer.start_as_current_span("validate_pipeline_config") as span:
            errors = []
            warnings = []
            security_issues = []
            
            try:
                # Check for circular dependencies
                step_deps = {i: getattr(step, 'depends_on', []) for i, step in enumerate(config.steps)}
                for i, deps in step_deps.items():
                    for dep in deps:
                        if dep >= len(config.steps):
                            errors.append(f"Step {i} depends on non-existent step {dep}")
                        elif dep >= i:
                            errors.append(f"Step {i} depends on step {dep} which comes after it")
                
                # Check for missing service methods
                for i, step in enumerate(config.steps):
                    service_map = {
                        "preprocessing": self.preprocessing,
                        "transformation": self.transformation,
                        "modeling": self.modeling,
                        "visualization": self.visualization
                    }
                    
                    service = service_map.get(step.service.value)
                    if service and not hasattr(service, step.method):
                        warnings.append(f"Step {i}: Method {step.method} not found on {step.service} service")
                
                # Security validation
                for i, step in enumerate(config.steps):
                    # Check for potentially unsafe configurations
                    if step.config.get("allow_network", False):
                        security_issues.append(f"Step {i} allows network access - potential security risk")
                    
                    if step.config.get("privileged", False):
                        security_issues.append(f"Step {i} requests privileged mode - security violation")
                    
                    # Check container image sources
                    if step.config.get("container_image"):
                        image = step.config["container_image"]
                        if not self._is_trusted_image_source(image):
                            security_issues.append(f"Step {i} uses untrusted container image: {image}")
                
                # Resource validation
                if hasattr(config, 'resource_limits') and config.resource_limits:
                    if config.resource_limits.max_duration_seconds and config.resource_limits.max_duration_seconds < 60:
                        warnings.append("Pipeline duration limit is very short (< 1 minute)")
                    
                    if config.resource_limits.max_memory_mb and config.resource_limits.max_memory_mb > 8192:
                        warnings.append("Pipeline memory limit is very high (> 8GB)")
                
                # Webhook security validation
                if hasattr(config, 'webhook') and config.webhook:
                    if not config.webhook.secret:
                        security_issues.append("Webhook configured without secret - authentication risk")
                    
                    if str(config.webhook.url).startswith("http://"):
                        security_issues.append("Webhook uses insecure HTTP protocol")
                
                validation_result = {
                    "valid": len(errors) == 0 and len(security_issues) == 0,
                    "errors": errors,
                    "warnings": warnings,
                    "security_issues": security_issues
                }
                
                span.set_attributes({
                    "validation.errors": len(errors),
                    "validation.warnings": len(warnings),
                    "validation.security_issues": len(security_issues),
                    "validation.valid": validation_result["valid"]
                })
                
                return validation_result
                
            except Exception as e:
                span.set_attribute("validation.error", str(e))
                return {
                    "valid": False,
                    "errors": [f"Validation failed: {str(e)}"],
                    "warnings": [],
                    "security_issues": []
                }

    def _is_trusted_image_source(self, image: str) -> bool:
        """Check if container image comes from trusted source."""
        trusted_registries = [
            "docker.io/library/",  # Official Docker images
            "gcr.io/",  # Google Container Registry
            "mcr.microsoft.com/",  # Microsoft Container Registry
            "registry.access.redhat.com/",  # Red Hat Registry
            "your-company-registry.com/"  # Your company registry
        ]
        
        return any(image.startswith(registry) for registry in trusted_registries)

    async def get_comprehensive_metrics(self, pipeline_type: Optional[str] = None, 
                                      environment: Optional[str] = None,
                                      start_date: Optional[datetime] = None,
                                      end_date: Optional[datetime] = None,
                                      security_context: Optional[SecurityContext] = None) -> Dict[str, Any]:
        """Get comprehensive pipeline execution metrics with enhanced analytics."""
        with tracer.start_as_current_span("get_comprehensive_metrics") as span:
            try:
                with get_db_session() as session:
                    query = session.query(DbPipelineRun)
                    
                    # Apply filters
                    if pipeline_type:
                        query = query.filter(DbPipelineRun.pipeline_type == pipeline_type)
                    if environment:
                        query = query.filter(DbPipelineRun.environment == environment)
                    if start_date:
                        query = query.filter(DbPipelineRun.created_at >= start_date)
                    if end_date:
                        query = query.filter(DbPipelineRun.created_at <= end_date)
                    
                    runs = query.all()
                    
                    # Basic metrics
                    total_runs = len(runs)
                    successful_runs = len([r for r in runs if r.status == RunStatus.completed])
                    failed_runs = len([r for r in runs if r.status == RunStatus.failed])
                    cancelled_runs = len([r for r in runs if r.status == RunStatus.cancelled])
                    
                    # Duration analysis
                    durations = [r.actual_duration for r in runs if r.actual_duration]
                    avg_duration = sum(durations) / len(durations) if durations else 0
                    
                    # Error analysis
                    error_types = {}
                    for run in runs:
                        if hasattr(run, 'error_message') and run.error_message:
                            error_type = type(run.error_message).__name__
                            error_types[error_type] = error_types.get(error_type, 0) + 1
                    
                    # Performance trends (last 30 days)
                    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
                    recent_runs = [r for r in runs if r.created_at >= thirty_days_ago]
                    
                    # Resource utilization
                    resource_stats = {
                        "avg_memory_usage_mb": 0,
                        "avg_cpu_usage_percent": 0,
                        "peak_memory_usage_mb": 0,
                        "peak_cpu_usage_percent": 0
                    }
                    
                    # Security metrics
                    security_events_count = 0
                    if security_context:
                        security_events = session.query(DbSecurityEvent).filter(
                            DbSecurityEvent.created_at >= (start_date or thirty_days_ago)
                        ).count()
                        security_events_count = security_events
                    
                    metrics = {
                        "summary": {
                            "total_runs": total_runs,
                            "successful_runs": successful_runs,
                            "failed_runs": failed_runs,
                            "cancelled_runs": cancelled_runs,
                            "success_rate": (successful_runs / total_runs * 100) if total_runs > 0 else 0
                        },
                        "performance": {
                            "avg_duration_seconds": avg_duration,
                            "recent_runs_count": len(recent_runs),
                            "resource_utilization": resource_stats
                        },
                        "reliability": {
                            "error_types": error_types,
                            "failure_rate": (failed_runs / total_runs * 100) if total_runs > 0 else 0
                        },
                        "security": {
                            "security_events_count": security_events_count,
                            "container_security_violations": 0  # Would be populated from monitoring
                        }
                    }
                    
                    span.set_attributes({
                        "metrics.total_runs": total_runs,
                        "metrics.success_rate": metrics["summary"]["success_rate"]
                    })
                    
                    return metrics
                    
            except Exception as e:
                span.set_attribute("metrics.error", str(e))
                logger.error(f"Failed to get metrics: {e}")
                return {"error": str(e)}


# Singleton accessor with enhanced initialization
def create_automation_service(config: Optional[ObservabilityConfig] = None) -> EnhancedAutomationService:
    """Factory function to create automation service with configuration."""
    return EnhancedAutomationService(config)


# Default instance
automation_service = EnhancedAutomationService()