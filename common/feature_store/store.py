"""
Feature Store implementation using DuckDB/Parquet
"""

import duckdb
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
import json


class FeatureStore:
    """
    Feature Store for trading intelligence system
    
    Uses DuckDB for fast analytical queries over Parquet files
    Stores features, labels, and metadata for ML models
    
    TODO Items:
    1. Implement feature versioning
    2. Add feature lineage tracking  
    3. Implement feature quality monitoring
    4. Add feature drift detection
    5. Implement point-in-time correctness
    6. Add feature serving for real-time inference
    7. Implement feature transformation pipelines
    8. Add data validation and schema enforcement
    9. Implement feature importance tracking
    10. Add automated feature engineering
    """
    
    def __init__(self, db_path: str = "data/features.db", 
                 parquet_path: str = "data/parquet/"):
        self.db_path = db_path
        self.parquet_path = Path(parquet_path)
        self.parquet_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize DuckDB connection
        self.conn = duckdb.connect(db_path)
        self._initialize_schema()
    
    def _initialize_schema(self):
        """Initialize feature store schema"""
        # TODO: Create proper schema for feature metadata
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS feature_metadata (
                feature_name VARCHAR,
                feature_group VARCHAR,
                data_type VARCHAR,
                description TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                version INTEGER,
                tags JSON
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS feature_lineage (
                feature_name VARCHAR,
                source_tables JSON,
                transformation_logic TEXT,
                dependencies JSON,
                created_at TIMESTAMP
            )
        """)
    
    async def write_features(self, feature_group: str, 
                           features_df: pd.DataFrame,
                           metadata: Dict[str, Any] = None) -> bool:
        """
        Write features to the store
        
        Args:
            feature_group: Group name for features (e.g., 'technical_indicators')
            features_df: DataFrame with features and timestamps
            metadata: Additional metadata about features
            
        Returns:
            Success status
        """
        try:
            # Ensure timestamp column exists
            if 'timestamp' not in features_df.columns:
                features_df['timestamp'] = datetime.now()
            
            # Write to Parquet with partitioning
            partition_path = self.parquet_path / feature_group
            partition_path.mkdir(exist_ok=True)
            
            # Partition by date for efficient queries
            features_df['date'] = pd.to_datetime(features_df['timestamp']).dt.date
            
            parquet_file = partition_path / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
            features_df.to_parquet(parquet_file, engine='pyarrow', index=False)
            
            # Update metadata
            await self._update_feature_metadata(feature_group, features_df.columns.tolist(), metadata)
            
            return True
            
        except Exception as e:
            print(f"Error writing features: {e}")
            return False
    
    async def read_features(self, feature_group: str,
                          features: List[str] = None,
                          start_time: datetime = None,
                          end_time: datetime = None,
                          symbols: List[str] = None) -> pd.DataFrame:
        """
        Read features from the store
        
        Args:
            feature_group: Feature group to read from
            features: Specific features to read (None for all)
            start_time: Start time filter
            end_time: End time filter  
            symbols: Symbol filter
            
        Returns:
            DataFrame with requested features
        """
        try:
            # Build query
            partition_path = self.parquet_path / feature_group
            if not partition_path.exists():
                return pd.DataFrame()
            
            # Use DuckDB to query Parquet files efficiently
            query = f"SELECT * FROM '{partition_path}/*.parquet'"
            
            # Add filters
            conditions = []
            if start_time:
                conditions.append(f"timestamp >= '{start_time}'")
            if end_time:
                conditions.append(f"timestamp <= '{end_time}'")
            if symbols:
                symbol_list = "', '".join(symbols)
                conditions.append(f"symbol IN ('{symbol_list}')")
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            # Select specific features
            if features:
                feature_list = ", ".join(['timestamp', 'symbol'] + features)
                query = query.replace("SELECT *", f"SELECT {feature_list}")
            
            result = self.conn.execute(query).fetchdf()
            return result
            
        except Exception as e:
            print(f"Error reading features: {e}")
            return pd.DataFrame()
    
    async def create_feature_set(self, name: str, 
                               feature_groups: List[str],
                               join_keys: List[str] = None) -> str:
        """
        Create a feature set by joining multiple feature groups
        
        Args:
            name: Name for the feature set
            feature_groups: List of feature groups to join
            join_keys: Keys to join on (default: ['timestamp', 'symbol'])
            
        Returns:
            Feature set ID
        """
        if join_keys is None:
            join_keys = ['timestamp', 'symbol']
        
        # TODO: Implement feature set creation with proper joins
        # This would create a view or materialized table combining features
        feature_set_id = f"fs_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return feature_set_id
    
    async def get_point_in_time_features(self, timestamp: datetime,
                                       symbols: List[str],
                                       feature_groups: List[str]) -> pd.DataFrame:
        """
        Get point-in-time correct features for a specific timestamp
        
        This ensures no look-ahead bias by only using data available at the timestamp
        """
        # TODO: Implement point-in-time correctness
        # 1. For each feature group, find the latest available data <= timestamp
        # 2. Join features while maintaining temporal consistency
        # 3. Handle missing data appropriately
        
        features_list = []
        for group in feature_groups:
            group_features = await self.read_features(
                feature_group=group,
                end_time=timestamp,
                symbols=symbols
            )
            
            if not group_features.empty:
                # Get latest features for each symbol
                latest_features = group_features.groupby('symbol').last().reset_index()
                features_list.append(latest_features)
        
        # Simple join for now (TODO: improve)
        if features_list:
            result = features_list[0]
            for df in features_list[1:]:
                result = result.merge(df, on='symbol', how='outer', suffixes=('', '_dup'))
            return result
        
        return pd.DataFrame()
    
    async def _update_feature_metadata(self, feature_group: str, 
                                     feature_names: List[str],
                                     metadata: Dict[str, Any] = None):
        """Update feature metadata in the catalog"""
        metadata = metadata or {}
        
        for feature_name in feature_names:
            if feature_name in ['timestamp', 'symbol', 'date']:
                continue  # Skip system columns
                
            self.conn.execute("""
                INSERT OR REPLACE INTO feature_metadata 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                feature_name,
                feature_group, 
                'DOUBLE',  # Default type
                metadata.get('description', ''),
                datetime.now(),
                datetime.now(),
                1,  # Version
                json.dumps(metadata.get('tags', {}))
            ])
    
    def get_feature_catalog(self) -> pd.DataFrame:
        """Get catalog of all available features"""
        return self.conn.execute("""
            SELECT feature_name, feature_group, description, created_at, tags
            FROM feature_metadata 
            ORDER BY feature_group, feature_name
        """).fetchdf()
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_feature_store():
        store = FeatureStore()
        
        # Create sample features
        sample_data = pd.DataFrame({
            'timestamp': [datetime.now() - timedelta(hours=i) for i in range(10)],
            'symbol': ['AAPL'] * 10,
            'rsi': np.random.uniform(30, 70, 10),
            'sma_20': np.random.uniform(150, 160, 10),
            'volume': np.random.uniform(1000000, 5000000, 10)
        })
        
        # Write features
        await store.write_features('technical_indicators', sample_data)
        
        # Read features back
        features = await store.read_features('technical_indicators')
        print(f"Retrieved {len(features)} feature records")
        
        # Get catalog
        catalog = store.get_feature_catalog()
        print("Feature catalog:")
        print(catalog)
        
        store.close()
    
    asyncio.run(test_feature_store())
