#!/usr/bin/env python3
"""
Simple Feature Store for Testing

A simplified feature store that works without external dependencies
for testing the complete architecture.
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
from collections import defaultdict

logger = logging.getLogger(__name__)


class SimpleFeatureStore:
    """
    Simple feature store for testing without external dependencies
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Performance configuration
        self.max_read_latency_ms = config.get('max_read_latency_ms', 5)
        self.default_ttl_seconds = config.get('default_ttl_seconds', 3600)  # 1 hour
        self.compression_threshold_bytes = config.get('compression_threshold_bytes', 1024)
        self.batch_size = config.get('batch_size', 100)
        
        # Feature storage (in-memory for testing)
        self.features: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.feature_metadata: Dict[str, Dict[str, Any]] = {}
        self.feature_ttl: Dict[str, datetime] = {}
        
        # Performance metrics
        self.metrics = {
            'reads': 0,
            'writes': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_read_latency_ms': 0.0,
            'avg_write_latency_ms': 0.0,
            'compression_savings_pct': 0.0,
        }
        
        # Health status
        self.is_healthy = False
        
    async def start(self):
        """Start the feature store"""
        logger.info("🚀 Starting Simple Feature Store...")
        
        try:
            # Start cleanup task
            asyncio.create_task(self._cleanup_expired_features())
            self.is_healthy = True
            logger.info("✅ Simple Feature Store started successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to start feature store: {e}")
            raise
    
    async def stop(self):
        """Stop the feature store"""
        logger.info("🛑 Stopping Simple Feature Store...")
        self.is_healthy = False
        logger.info("✅ Simple Feature Store stopped")
    
    async def write_features(self, feature_group: str, data: pd.DataFrame, 
                           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Write features to the store"""
        start_time = time.time()
        
        try:
            # Generate feature key
            feature_key = f"{feature_group}:{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Serialize data
            serialized_data = self._serialize_data(data)
            
            # Store features
            self.features[feature_group][feature_key] = {
                'data': serialized_data,
                'metadata': metadata or {},
                'created_at': datetime.utcnow(),
                'size_bytes': len(serialized_data)
            }
            
            # Set TTL
            ttl_seconds = metadata.get('ttl_seconds', self.default_ttl_seconds) if metadata else self.default_ttl_seconds
            self.feature_ttl[feature_key] = datetime.utcnow() + timedelta(seconds=ttl_seconds)
            
            # Update metrics
            write_time = (time.time() - start_time) * 1000
            self.metrics['writes'] += 1
            self.metrics['avg_write_latency_ms'] = (
                (self.metrics['avg_write_latency_ms'] * (self.metrics['writes'] - 1) + write_time) / self.metrics['writes']
            )
            
            logger.debug(f"Wrote features: {feature_key} ({len(serialized_data)} bytes)")
            return True
            
        except Exception as e:
            logger.error(f"Error writing features: {e}")
            return False
    
    async def get_features(self, symbols: List[str], feature_groups: List[str], 
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """Get features from the store"""
        start_time_query = time.time()
        
        try:
            all_data = []
            
            for feature_group in feature_groups:
                if feature_group in self.features:
                    for feature_key, feature_data in self.features[feature_group].items():
                        # Check if feature is expired
                        if feature_key in self.feature_ttl:
                            if datetime.utcnow() > self.feature_ttl[feature_key]:
                                continue
                        
                        # Deserialize data
                        data = self._deserialize_data(feature_data['data'])
                        
                        # Filter by symbols if provided
                        if symbols and 'symbol' in data.columns:
                            data = data[data['symbol'].isin(symbols)]
                        
                        # Filter by time if provided
                        if start_time and 'timestamp' in data.columns:
                            data = data[data['timestamp'] >= start_time]
                        if end_time and 'timestamp' in data.columns:
                            data = data[data['timestamp'] <= end_time]
                        
                        all_data.append(data)
            
            if all_data:
                result = pd.concat(all_data, ignore_index=True)
                
                # Update metrics
                read_time = (time.time() - start_time_query) * 1000
                self.metrics['reads'] += 1
                self.metrics['cache_hits'] += 1
                self.metrics['avg_read_latency_ms'] = (
                    (self.metrics['avg_read_latency_ms'] * (self.metrics['reads'] - 1) + read_time) / self.metrics['reads']
                )
                
                logger.debug(f"Retrieved features: {len(result)} rows")
                return result
            else:
                self.metrics['cache_misses'] += 1
                logger.debug("No features found")
                return None
                
        except Exception as e:
            logger.error(f"Error getting features: {e}")
            return None
    
    async def get_feature_metadata(self, feature_group: str) -> Dict[str, Any]:
        """Get metadata for a feature group"""
        if feature_group in self.features:
            metadata = {
                'feature_count': len(self.features[feature_group]),
                'total_size_bytes': sum(f['size_bytes'] for f in self.features[feature_group].values()),
                'last_updated': max(f['created_at'] for f in self.features[feature_group].values()),
                'features': list(self.features[feature_group].keys())
            }
            return metadata
        else:
            return {}
    
    def _serialize_data(self, data: pd.DataFrame) -> bytes:
        """Serialize data with optional compression"""
        try:
            # Convert to bytes
            serialized = pickle.dumps(data)
            
            # Compress if above threshold
            if len(serialized) > self.compression_threshold_bytes:
                compressed = gzip.compress(serialized)
                if len(compressed) < len(serialized):
                    return compressed
            
            return serialized
            
        except Exception as e:
            logger.error(f"Error serializing data: {e}")
            return pickle.dumps(data)
    
    def _deserialize_data(self, data: bytes) -> pd.DataFrame:
        """Deserialize data with decompression if needed"""
        try:
            # Try to decompress first
            try:
                decompressed = gzip.decompress(data)
                return pickle.loads(decompressed)
            except:
                # If decompression fails, try direct deserialization
                return pickle.loads(data)
                
        except Exception as e:
            logger.error(f"Error deserializing data: {e}")
            return pd.DataFrame()
    
    async def _cleanup_expired_features(self):
        """Clean up expired features"""
        while self.is_healthy:
            try:
                current_time = datetime.utcnow()
                expired_keys = []
                
                for feature_key, expiry_time in self.feature_ttl.items():
                    if current_time > expiry_time:
                        expired_keys.append(feature_key)
                
                # Remove expired features
                for feature_key in expired_keys:
                    for feature_group in self.features:
                        if feature_key in self.features[feature_group]:
                            del self.features[feature_group][feature_key]
                    del self.feature_ttl[feature_key]
                
                if expired_keys:
                    logger.debug(f"Cleaned up {len(expired_keys)} expired features")
                
                # Wait before next cleanup
                await asyncio.sleep(60)  # Clean up every minute
                
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(60)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.metrics.copy()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status"""
        return {
            'is_healthy': self.is_healthy,
            'feature_groups': list(self.features.keys()),
            'total_features': sum(len(features) for features in self.features.values()),
            'expired_features': len([k for k, v in self.feature_ttl.items() if datetime.utcnow() > v])
        }
