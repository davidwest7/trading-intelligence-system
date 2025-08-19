"""
Base data adapter interface
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime


class BaseDataAdapter(ABC):
    """Base class for all data adapters"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.is_connected = False
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to data source"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Close connection to data source"""
        pass
    
    @abstractmethod
    async def get_ohlcv(self, symbol: str, timeframe: str, 
                       since: datetime, limit: int = 1000) -> pd.DataFrame:
        """Get OHLCV data"""
        pass
    
    @abstractmethod
    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get current quote"""
        pass
    
    def health_check(self) -> Dict[str, Any]:
        """Check adapter health"""
        return {
            "name": self.name,
            "connected": self.is_connected,
            "config": {k: v for k, v in self.config.items() if "key" not in k.lower()}
        }
