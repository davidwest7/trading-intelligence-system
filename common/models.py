"""
Common models and base classes for all agents
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime


class BaseAgent(ABC):
    """Base class for all trading agents"""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.created_at = datetime.now()
        self.is_active = True
    
    @abstractmethod
    async def process(self, *args, **kwargs) -> Dict[str, Any]:
        """Main processing method for the agent"""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "name": self.name,
            "active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "config": self.config
        }


class BaseDataAdapter(ABC):
    """Base class for all data adapters"""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.created_at = datetime.now()
        self.is_connected = False
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the data source"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from the data source"""
        pass
    
    @abstractmethod
    async def get_ohlcv(self, symbol: str, timeframe: str,
                       since: datetime, limit: int = 1000) -> Any:
        """Get OHLCV data"""
        pass
    
    @abstractmethod
    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get current quote"""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get adapter status"""
        return {
            "name": self.name,
            "connected": self.is_connected,
            "created_at": self.created_at.isoformat(),
            "config": self.config
        }
