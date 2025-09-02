# app/core/model_registry.py
import asyncio
import logging
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
import joblib

from app.config import settings


class ModelStatus(Enum):
    """Model status enumeration."""
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"
    UPDATING = "updating"
    DEPRECATED = "deprecated"


class ModelType(Enum):
    """Model type enumeration."""
    SKLEARN = "sklearn"
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    CUSTOM = "custom"
    PREPROCESSING = "preprocessing"
    TRANSFORMATION = "transformation"


@dataclass
class ModelMetadata:
    """Model metadata container."""
    name: str
    version: str
    model_type: ModelType
    created_at: datetime
    updated_at: datetime
    status: ModelStatus
    file_path: Optional[str] = None
    description: str = ""
    tags: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    input_schema: Optional[Dict] = None
    output_schema: Optional[Dict] = None
    size_bytes: int = 0
    checksum: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            'name': self.name,
            'version': self.version,
            'model_type': self.model_type.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'status': self.status.value,
            'file_path': self.file_path,
            'description': self.description,
            'tags': self.tags,
            'metrics': self.metrics,
            'parameters': self.parameters,
            'dependencies': self.dependencies,
            'input_schema': self.input_schema,
            'output_schema': self.output_schema,
            'size_bytes': self.size_bytes,
            'checksum': self.checksum
        }


@dataclass
class CachedModel:
    """Cached model container."""
    model: Any
    metadata: ModelMetadata
    last_accessed: datetime
    access_count: int = 0
    
    def update_access(self):
        """Update access tracking."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1


class ModelRegistry:
    """
    Centralized model registry for managing ML models in Kaizen.
    
    Features:
    - Model loading and caching
    - Metadata management
    - Health monitoring
    - Version control
    - Automatic cleanup
    """
    
    def __init__(self, 
                 model_dir: Optional[str] = None,
                 cache_size_mb: int = 1024,
                 cache_ttl_hours: int = 24):
        """
        Initialize the model registry.
        
        Args:
            model_dir: Directory for storing models
            cache_size_mb: Maximum cache size in MB
            cache_ttl_hours: Cache TTL in hours
        """
        self.model_dir = Path(model_dir or settings.MODEL_DIR or "./models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_size_mb = cache_size_mb
        self.cache_ttl_hours = cache_ttl_hours
        
        # In-memory model cache
        self._cache: Dict[str, CachedModel] = {}
        self._cache_lock = threading.RLock()
        
        # Model metadata storage
        self._metadata: Dict[str, ModelMetadata] = {}
        self._metadata_file = self.model_dir / "registry.json"
        
        # Registry state
        self._initialized = False
        self._healthy = False
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None
        
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> None:
        """Initialize the model registry."""
        try:
            self.logger.info("Initializing model registry...")
            
            # Load existing metadata
            await self._load_metadata()
            
            # Start background tasks
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
            self._health_check_task = asyncio.create_task(self._periodic_health_check())
            
            self._initialized = True
            self._healthy = True
            
            self.logger.info(f"Model registry initialized with {len(self._metadata)} models")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize model registry: {e}")
            self._healthy = False
            raise
    
    async def cleanup(self) -> None:
        """Clean up registry resources."""
        try:
            self.logger.info("Cleaning up model registry...")
            
            # Cancel background tasks
            if self._cleanup_task:
                self._cleanup_task.cancel()
            if self._health_check_task:
                self._health_check_task.cancel()
            
            # Save metadata
            await self._save_metadata()
            
            # Clear cache
            with self._cache_lock:
                self._cache.clear()
            
            self.logger.info("Model registry cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during registry cleanup: {e}")
    
    def is_healthy(self) -> bool:
        """Check if registry is healthy."""
        return self._initialized and self._healthy
    
    async def register_model(self, 
                           name: str,
                           model: Any,
                           version: str = "1.0.0",
                           model_type: ModelType = ModelType.SKLEARN,
                           description: str = "",
                           tags: Optional[List[str]] = None,
                           metrics: Optional[Dict[str, float]] = None,
                           parameters: Optional[Dict[str, Any]] = None,
                           save_to_disk: bool = True) -> ModelMetadata:
        """
        Register a new model.
        
        Args:
            name: Model name
            model: Model object
            version: Model version
            model_type: Type of model
            description: Model description
            tags: Model tags
            metrics: Model performance metrics
            parameters: Model parameters
            save_to_disk: Whether to save model to disk
            
        Returns:
            Model metadata
        """
        try:
            model_id = f"{name}:{version}"
            
            # Create metadata
            metadata = ModelMetadata(
                name=name,
                version=version,
                model_type=model_type,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                status=ModelStatus.LOADING,
                description=description,
                tags=tags or [],
                metrics=metrics or {},
                parameters=parameters or {}
            )
            
            # Save to disk if requested
            if save_to_disk:
                file_path = await self._save_model_to_disk(model_id, model, metadata)
                metadata.file_path = str(file_path)
                metadata.checksum = await self._calculate_checksum(file_path)
                metadata.size_bytes = file_path.stat().st_size
            
            # Add to cache
            with self._cache_lock:
                self._cache[model_id] = CachedModel(
                    model=model,
                    metadata=metadata,
                    last_accessed=datetime.utcnow()
                )
            
            # Update metadata registry
            metadata.status = ModelStatus.READY
            self._metadata[model_id] = metadata
            
            # Save metadata
            await self._save_metadata()
            
            self.logger.info(f"Registered model: {model_id}")
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to register model {name}:{version}: {e}")
            raise
    
    async def get_model(self, name: str, version: str = "latest") -> Any:
        """
        Get a model by name and version.
        
        Args:
            name: Model name
            version: Model version or "latest"
            
        Returns:
            Model object
        """
        try:
            # Resolve version
            if version == "latest":
                version = self._get_latest_version(name)
            
            model_id = f"{name}:{version}"
            
            # Check cache first
            with self._cache_lock:
                if model_id in self._cache:
                    cached_model = self._cache[model_id]
                    cached_model.update_access()
                    return cached_model.model
            
            # Load from disk
            if model_id in self._metadata:
                metadata = self._metadata[model_id]
                if metadata.file_path and Path(metadata.file_path).exists():
                    model = await self._load_model_from_disk(Path(metadata.file_path), metadata)
                    
                    # Add to cache
                    with self._cache_lock:
                        self._cache[model_id] = CachedModel(
                            model=model,
                            metadata=metadata,
                            last_accessed=datetime.utcnow()
                        )
                    
                    return model
            
            raise FileNotFoundError(f"Model not found: {model_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to get model {name}:{version}: {e}")
            raise
    
    async def get_metadata(self, name: str, version: str = "latest") -> ModelMetadata:
        """Get model metadata."""
        if version == "latest":
            version = self._get_latest_version(name)
        
        model_id = f"{name}:{version}"
        
        if model_id not in self._metadata:
            raise FileNotFoundError(f"Model metadata not found: {model_id}")
        
        return self._metadata[model_id]
    
    def list_models(self, 
                   name_filter: Optional[str] = None,
                   tag_filter: Optional[str] = None,
                   status_filter: Optional[ModelStatus] = None) -> List[ModelMetadata]:
        """
        List models with optional filtering.
        
        Args:
            name_filter: Filter by model name
            tag_filter: Filter by tag
            status_filter: Filter by status
            
        Returns:
            List of model metadata
        """
        models = list(self._metadata.values())
        
        if name_filter:
            models = [m for m in models if name_filter.lower() in m.name.lower()]
        
        if tag_filter:
            models = [m for m in models if tag_filter in m.tags]
        
        if status_filter:
            models = [m for m in models if m.status == status_filter]
        
        # Sort by updated_at descending
        models.sort(key=lambda x: x.updated_at, reverse=True)
        
        return models
    
    async def delete_model(self, name: str, version: str) -> None:
        """Delete a model."""
        model_id = f"{name}:{version}"
        
        # Remove from cache
        with self._cache_lock:
            if model_id in self._cache:
                del self._cache[model_id]
        
        # Remove from metadata
        if model_id in self._metadata:
            metadata = self._metadata[model_id]
            
            # Delete file
            if metadata.file_path and Path(metadata.file_path).exists():
                Path(metadata.file_path).unlink()
            
            del self._metadata[model_id]
            await self._save_metadata()
        
        self.logger.info(f"Deleted model: {model_id}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._cache_lock:
            total_size = sum(
                len(pickle.dumps(cached.model)) 
                for cached in self._cache.values()
            )
            
            return {
                "cached_models": len(self._cache),
                "cache_size_bytes": total_size,
                "cache_size_mb": total_size / (1024 * 1024),
                "cache_utilization": min(100, (total_size / (self.cache_size_mb * 1024 * 1024)) * 100)
            }
    
    async def _load_metadata(self) -> None:
        """Load metadata from disk."""
        if self._metadata_file.exists():
            try:
                with open(self._metadata_file, 'r') as f:
                    data = json.load(f)
                
                for model_id, metadata_dict in data.items():
                    metadata = ModelMetadata(
                        name=metadata_dict['name'],
                        version=metadata_dict['version'],
                        model_type=ModelType(metadata_dict['model_type']),
                        created_at=datetime.fromisoformat(metadata_dict['created_at']),
                        updated_at=datetime.fromisoformat(metadata_dict['updated_at']),
                        status=ModelStatus(metadata_dict['status']),
                        file_path=metadata_dict.get('file_path'),
                        description=metadata_dict.get('description', ''),
                        tags=metadata_dict.get('tags', []),
                        metrics=metadata_dict.get('metrics', {}),
                        parameters=metadata_dict.get('parameters', {}),
                        dependencies=metadata_dict.get('dependencies', []),
                        input_schema=metadata_dict.get('input_schema'),
                        output_schema=metadata_dict.get('output_schema'),
                        size_bytes=metadata_dict.get('size_bytes', 0),
                        checksum=metadata_dict.get('checksum')
                    )
                    self._metadata[model_id] = metadata
                
                self.logger.info(f"Loaded metadata for {len(self._metadata)} models")
                
            except Exception as e:
                self.logger.error(f"Failed to load metadata: {e}")
    
    async def _save_metadata(self) -> None:
        """Save metadata to disk."""
        try:
            data = {
                model_id: metadata.to_dict() 
                for model_id, metadata in self._metadata.items()
            }
            
            with open(self._metadata_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Failed to save metadata: {e}")
    
    async def _save_model_to_disk(self, 
                                 model_id: str, 
                                 model: Any, 
                                 metadata: ModelMetadata) -> Path:
        """Save model to disk."""
        file_path = self.model_dir / f"{model_id.replace(':', '_')}.joblib"
        
        try:
            if isinstance(model, BaseEstimator):
                joblib.dump(model, file_path)
            else:
                # Use pickle for other types
                with open(file_path, 'wb') as f:
                    pickle.dump(model, f)
            
            return file_path
            
        except Exception as e:
            self.logger.error(f"Failed to save model {model_id}: {e}")
            raise
    
    async def _load_model_from_disk(self, 
                                   file_path: Path, 
                                   metadata: ModelMetadata) -> Any:
        """Load model from disk."""
        try:
            if metadata.model_type == ModelType.SKLEARN:
                return joblib.load(file_path)
            else:
                # Use pickle for other types
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
                    
        except Exception as e:
            self.logger.error(f"Failed to load model from {file_path}: {e}")
            raise
    
    async def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate file checksum."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _get_latest_version(self, name: str) -> str:
        """Get the latest version of a model."""
        versions = [
            metadata.version 
            for metadata in self._metadata.values() 
            if metadata.name == name
        ]
        
        if not versions:
            raise FileNotFoundError(f"No versions found for model: {name}")
        
        # Simple version sorting (assumes semantic versioning)
        versions.sort(key=lambda v: tuple(map(int, v.split('.'))), reverse=True)
        return versions[0]
    
    async def _periodic_cleanup(self) -> None:
        """Periodic cache cleanup task."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self._cleanup_cache()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in periodic cleanup: {e}")
    
    async def _cleanup_cache(self) -> None:
        """Clean up expired cache entries."""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.cache_ttl_hours)
        
        with self._cache_lock:
            expired_keys = [
                key for key, cached in self._cache.items()
                if cached.last_accessed < cutoff_time
            ]
            
            for key in expired_keys:
                del self._cache[key]
            
            if expired_keys:
                self.logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    async def _periodic_health_check(self) -> None:
        """Periodic health check task."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                self._check_health()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health check: {e}")
                self._healthy = False
    
    def _check_health(self) -> None:
        """Perform health checks."""
        try:
            # Check if model directory is accessible
            if not self.model_dir.exists():
                self.model_dir.mkdir(parents=True, exist_ok=True)
            
            # Check cache size
            stats = self.get_cache_stats()
            if stats['cache_utilization'] > 90:
                self.logger.warning("Cache utilization is high: {:.1f}%".format(stats['cache_utilization']))
            
            self._healthy = True
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            self._healthy = False