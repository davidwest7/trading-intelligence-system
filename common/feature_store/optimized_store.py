"""
Optimized Feature Store for Trading System

High-performance feature store with Redis backend for <5ms reads,
feature versioning, drift detection, and intelligent caching.
"""

import asyncio
import json
import logging
import hashlib
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dataclasses import asdict
import pickle
import gzip

import redis.asyncio as redis
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class FeatureMetadata(BaseModel):
    """Metadata for a feature"""
    feature_name: str
    feature_version: str
    data_type: str
    created_at: datetime
    last_updated: datetime
    ttl_seconds: int
    size_bytes: int
    compression_ratio: float
    access_count: int
    last_accessed: datetime


class FeatureDrift(BaseModel):
    """Feature drift information"""
    feature_name: str
    drift_score: float
    drift_type: str  # 'distribution', 'statistical', 'temporal'
    detected_at: datetime
    severity: str  # 'low', 'medium', 'high', 'critical'
    details: Dict[str, Any]


class OptimizedFeatureStore:
    """
    High-performance feature store with Redis backend
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Redis configuration
        self.redis_config = {
            'host': config.get('redis_host', 'localhost'),
            'port': config.get('redis_port', 6379),
            'db': config.get('redis_db', 1),  # Use separate DB for features
            'decode_responses': False,  # Keep binary for compression
            'max_connections': config.get('redis_max_connections', 20),
            'socket_keepalive': True,
            'socket_keepalive_options': {},
        }
        
        # Performance configuration
        self.max_read_latency_ms = config.get('max_read_latency_ms', 5)
        self.default_ttl_seconds = config.get('default_ttl_seconds', 3600)  # 1 hour
        self.compression_threshold_bytes = config.get('compression_threshold_bytes', 1024)
        self.batch_size = config.get('batch_size', 100)
        
        # Drift detection configuration
        self.drift_detection_enabled = config.get('drift_detection_enabled', True)
        self.drift_threshold = config.get('drift_threshold', 0.1)
        self.drift_window_size = config.get('drift_window_size', 1000)
        
        # Initialize components
        self.redis_client: Optional[redis.Redis] = None
        self.feature_metadata: Dict[str, FeatureMetadata] = {}
        self.drift_history: List[FeatureDrift] = []
        
        # Performance metrics
        self.metrics = {
            'reads': 0,
            'writes': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_read_latency_ms': 0.0,
            'avg_write_latency_ms': 0.0,
            'compression_savings_pct': 0.0,
            'drift_detections': 0,
        }
        
        # Health status
        self.is_healthy = False
        self.last_health_check = datetime.utcnow()
        
    async def start(self):
        """Start the feature store"""
        logger.info("🚀 Starting Optimized Feature Store...")
        
        try:
            # Initialize Redis connection
            await self._init_redis()
            
            # Load existing metadata
            await self._load_metadata()
            
            # Start background tasks
            asyncio.create_task(self._health_monitor())
            asyncio.create_task(self._cleanup_expired_features())
            asyncio.create_task(self._drift_detection_worker())
            
            self.is_healthy = True
            logger.info("✅ Optimized Feature Store started successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to start feature store: {e}")
            raise
    
    async def stop(self):
        """Stop the feature store"""
        logger.info("🛑 Stopping Optimized Feature Store...")
        
        self.is_healthy = False
        
        # Save metadata
        await self._save_metadata()
        
        # Close Redis connection
        if self.redis_client:
            try:
                await self.redis_client.close()
                logger.info("✅ Redis connection closed")
            except Exception as e:
                logger.error(f"❌ Error closing Redis: {e}")
        
        logger.info("✅ Optimized Feature Store stopped")
    
    async def _init_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis(**self.redis_config)
            await self.redis_client.ping()
            logger.info("✅ Redis connection established")
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            raise
    
    async def _load_metadata(self):
        """Load feature metadata from Redis"""
        try:
            metadata_key = "feature_store:metadata"
            metadata_data = await self.redis_client.get(metadata_key)
            
            if metadata_data:
                metadata_dict = json.loads(metadata_data.decode('utf-8'))
                for feature_name, meta_dict in metadata_dict.items():
                    self.feature_metadata[feature_name] = FeatureMetadata(**meta_dict)
                
                logger.info(f"✅ Loaded metadata for {len(self.feature_metadata)} features")
            else:
                logger.info("ℹ️ No existing metadata found")
                
        except Exception as e:
            logger.error(f"❌ Failed to load metadata: {e}")
    
    async def _save_metadata(self):
        """Save feature metadata to Redis"""
        try:
            metadata_dict = {
                name: meta.dict() for name, meta in self.feature_metadata.items()
            }
            
            metadata_key = "feature_store:metadata"
            await self.redis_client.set(
                metadata_key,
                json.dumps(metadata_dict),
                ex=86400  # 24 hours
            )
            
            logger.info(f"✅ Saved metadata for {len(self.feature_metadata)} features")
            
        except Exception as e:
            logger.error(f"❌ Failed to save metadata: {e}")
    
    async def set_feature(self, 
                         feature_name: str, 
                         feature_data: Any, 
                         ttl_seconds: Optional[int] = None,
                         compress: bool = True) -> bool:
        """Set a feature in the store"""
        if not self.is_healthy:
            logger.error("Feature store not healthy")
            return False
        
        try:
            start_time = time.time()
            
            # Generate feature version
            feature_version = self._generate_feature_version(feature_data)
            
            # Serialize and compress data
            serialized_data, compression_ratio = await self._serialize_data(
                feature_data, compress
            )
            
            # Create feature key
            feature_key = f"feature:{feature_name}:{feature_version}"
            
            # Set TTL
            ttl = ttl_seconds or self.default_ttl_seconds
            
            # Store in Redis
            await self.redis_client.set(feature_key, serialized_data, ex=ttl)
            
            # Update metadata
            metadata = FeatureMetadata(
                feature_name=feature_name,
                feature_version=feature_version,
                data_type=type(feature_data).__name__,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
                ttl_seconds=ttl,
                size_bytes=len(serialized_data),
                compression_ratio=compression_ratio,
                access_count=0,
                last_accessed=datetime.utcnow()
            )
            
            self.feature_metadata[feature_name] = metadata
            
            # Update metrics
            self.metrics['writes'] += 1
            latency = (time.time() - start_time) * 1000
            self.metrics['avg_write_latency_ms'] = (
                (self.metrics['avg_write_latency_ms'] * (self.metrics['writes'] - 1) + latency) /
                self.metrics['writes']
            )
            
            logger.debug(f"✅ Stored feature {feature_name} (v{feature_version}) in {latency:.2f}ms")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store feature {feature_name}: {e}")
            return False
    
    async def get_feature(self, feature_name: str, version: Optional[str] = None) -> Optional[Any]:
        """Get a feature from the store"""
        if not self.is_healthy:
            logger.error("Feature store not healthy")
            return None
        
        try:
            start_time = time.time()
            
            # Get metadata
            if feature_name not in self.feature_metadata:
                self.metrics['cache_misses'] += 1
                logger.debug(f"❌ Feature {feature_name} not found")
                return None
            
            metadata = self.feature_metadata[feature_name]
            
            # Use specified version or latest
            feature_version = version or metadata.feature_version
            feature_key = f"feature:{feature_name}:{feature_version}"
            
            # Get from Redis
            serialized_data = await self.redis_client.get(feature_key)
            
            if not serialized_data:
                self.metrics['cache_misses'] += 1
                logger.debug(f"❌ Feature {feature_name} (v{feature_version}) not found in Redis")
                return None
            
            # Deserialize data
            feature_data = await self._deserialize_data(serialized_data)
            
            # Update metadata
            metadata.access_count += 1
            metadata.last_accessed = datetime.utcnow()
            
            # Update metrics
            self.metrics['reads'] += 1
            self.metrics['cache_hits'] += 1
            latency = (time.time() - start_time) * 1000
            self.metrics['avg_read_latency_ms'] = (
                (self.metrics['avg_read_latency_ms'] * (self.metrics['reads'] - 1) + latency) /
                self.metrics['reads']
            )
            
            # Check latency SLA
            if latency > self.max_read_latency_ms:
                logger.warning(f"⚠️ Read latency {latency:.2f}ms exceeds SLA of {self.max_read_latency_ms}ms")
            
            logger.debug(f"✅ Retrieved feature {feature_name} (v{feature_version}) in {latency:.2f}ms")
            return feature_data
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve feature {feature_name}: {e}")
            self.metrics['cache_misses'] += 1
            return None
    
    async def get_features_batch(self, feature_names: List[str]) -> Dict[str, Any]:
        """Get multiple features in batch"""
        if not self.is_healthy:
            logger.error("Feature store not healthy")
            return {}
        
        try:
            start_time = time.time()
            
            # Prepare keys
            keys = []
            for feature_name in feature_names:
                if feature_name in self.feature_metadata:
                    metadata = self.feature_metadata[feature_name]
                    keys.append(f"feature:{feature_name}:{metadata.feature_version}")
                else:
                    keys.append(None)
            
            # Batch get from Redis
            values = await self.redis_client.mget([k for k in keys if k is not None])
            
            # Deserialize results
            results = {}
            for i, feature_name in enumerate(feature_names):
                if keys[i] is not None and values[i] is not None:
                    feature_data = await self._deserialize_data(values[i])
                    results[feature_name] = feature_data
                    
                    # Update metadata
                    metadata = self.feature_metadata[feature_name]
                    metadata.access_count += 1
                    metadata.last_accessed = datetime.utcnow()
            
            # Update metrics
            self.metrics['reads'] += len(feature_names)
            self.metrics['cache_hits'] += len(results)
            self.metrics['cache_misses'] += len(feature_names) - len(results)
            
            latency = (time.time() - start_time) * 1000
            logger.debug(f"✅ Retrieved {len(results)}/{len(feature_names)} features in {latency:.2f}ms")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve features batch: {e}")
            return {}
    
    async def delete_feature(self, feature_name: str) -> bool:
        """Delete a feature from the store"""
        if not self.is_healthy:
            logger.error("Feature store not healthy")
            return False
        
        try:
            if feature_name not in self.feature_metadata:
                logger.warning(f"⚠️ Feature {feature_name} not found for deletion")
                return False
            
            metadata = self.feature_metadata[feature_name]
            feature_key = f"feature:{feature_name}:{metadata.feature_version}"
            
            # Delete from Redis
            await self.redis_client.delete(feature_key)
            
            # Remove metadata
            del self.feature_metadata[feature_name]
            
            logger.info(f"✅ Deleted feature {feature_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete feature {feature_name}: {e}")
            return False
    
    async def list_features(self) -> List[str]:
        """List all available features"""
        return list(self.feature_metadata.keys())
    
    async def get_feature_metadata(self, feature_name: str) -> Optional[FeatureMetadata]:
        """Get metadata for a feature"""
        return self.feature_metadata.get(feature_name)
    
    async def _serialize_data(self, data: Any, compress: bool = True) -> Tuple[bytes, float]:
        """Serialize and optionally compress data"""
        try:
            # Serialize to pickle
            serialized = pickle.dumps(data)
            original_size = len(serialized)
            
            # Compress if enabled and above threshold
            if compress and original_size > self.compression_threshold_bytes:
                compressed = gzip.compress(serialized)
                compressed_size = len(compressed)
                
                # Use compression if it saves space
                if compressed_size < original_size:
                    compression_ratio = compressed_size / original_size
                    return compressed, compression_ratio
            
            return serialized, 1.0
            
        except Exception as e:
            logger.error(f"❌ Serialization failed: {e}")
            raise
    
    async def _deserialize_data(self, serialized_data: bytes) -> Any:
        """Deserialize data"""
        try:
            # Try to decompress first
            try:
                decompressed = gzip.decompress(serialized_data)
                return pickle.loads(decompressed)
            except (OSError, zlib.error):
                # Not compressed, load directly
                return pickle.loads(serialized_data)
                
        except Exception as e:
            logger.error(f"❌ Deserialization failed: {e}")
            raise
    
    def _generate_feature_version(self, data: Any) -> str:
        """Generate version hash for feature data"""
        try:
            # Create hash from data
            if isinstance(data, (dict, list)):
                data_str = json.dumps(data, sort_keys=True, default=str)
            elif isinstance(data, pd.DataFrame):
                data_str = data.to_json()
            elif isinstance(data, np.ndarray):
                data_str = data.tobytes().hex()
            else:
                data_str = str(data)
            
            # Generate hash
            hash_obj = hashlib.sha256(data_str.encode('utf-8'))
            return hash_obj.hexdigest()[:16]  # Use first 16 characters
            
        except Exception as e:
            logger.error(f"❌ Version generation failed: {e}")
            return str(int(time.time()))  # Fallback to timestamp
    
    async def _health_monitor(self):
        """Monitor feature store health"""
        while self.is_healthy:
            try:
                # Check Redis connection
                await self.redis_client.ping()
                
                # Check performance metrics
                if self.metrics['avg_read_latency_ms'] > self.max_read_latency_ms * 2:
                    logger.warning(f"⚠️ High read latency: {self.metrics['avg_read_latency_ms']:.2f}ms")
                
                # Update health check timestamp
                self.last_health_check = datetime.utcnow()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                self.is_healthy = False
                break
    
    async def _cleanup_expired_features(self):
        """Clean up expired features"""
        while self.is_healthy:
            try:
                current_time = datetime.utcnow()
                expired_features = []
                
                for feature_name, metadata in self.feature_metadata.items():
                    # Check if feature is expired
                    if (current_time - metadata.last_updated).total_seconds() > metadata.ttl_seconds:
                        expired_features.append(feature_name)
                
                # Delete expired features
                for feature_name in expired_features:
                    await self.delete_feature(feature_name)
                
                if expired_features:
                    logger.info(f"🧹 Cleaned up {len(expired_features)} expired features")
                
                await asyncio.sleep(300)  # Clean up every 5 minutes
                
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(300)
    
    async def _drift_detection_worker(self):
        """Background worker for drift detection"""
        if not self.drift_detection_enabled:
            return
        
        while self.is_healthy:
            try:
                # Check for drift in recently accessed features
                current_time = datetime.utcnow()
                recent_features = [
                    name for name, meta in self.feature_metadata.items()
                    if (current_time - meta.last_accessed).total_seconds() < 3600  # Last hour
                ]
                
                for feature_name in recent_features:
                    await self._detect_feature_drift(feature_name)
                
                await asyncio.sleep(600)  # Check drift every 10 minutes
                
            except Exception as e:
                logger.error(f"Drift detection error: {e}")
                await asyncio.sleep(600)
    
    async def _detect_feature_drift(self, feature_name: str):
        """Detect drift in a specific feature"""
        try:
            # Get current feature data
            current_data = await self.get_feature(feature_name)
            if current_data is None:
                return
            
            # Simple drift detection (can be enhanced)
            if isinstance(current_data, (list, np.ndarray)):
                # Statistical drift detection
                if len(current_data) > 10:
                    mean_val = np.mean(current_data)
                    std_val = np.std(current_data)
                    
                    # Check for significant changes
                    if std_val > self.drift_threshold:
                        drift = FeatureDrift(
                            feature_name=feature_name,
                            drift_score=std_val,
                            drift_type='statistical',
                            detected_at=datetime.utcnow(),
                            severity='medium' if std_val < 0.5 else 'high',
                            details={'mean': mean_val, 'std': std_val}
                        )
                        
                        self.drift_history.append(drift)
                        self.metrics['drift_detections'] += 1
                        
                        logger.warning(f"⚠️ Drift detected in {feature_name}: {drift.drift_score:.3f}")
            
        except Exception as e:
            logger.error(f"Drift detection failed for {feature_name}: {e}")
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the feature store"""
        return {
            'healthy': self.is_healthy,
            'last_health_check': self.last_health_check.isoformat(),
            'metrics': self.metrics,
            'feature_count': len(self.feature_metadata),
            'drift_detections': len(self.drift_history),
            'avg_read_latency_ms': self.metrics['avg_read_latency_ms'],
            'cache_hit_rate': (
                self.metrics['cache_hits'] / max(self.metrics['reads'], 1)
            ),
        }
    
    async def get_drift_report(self) -> List[FeatureDrift]:
        """Get drift detection report"""
        return self.drift_history.copy()
