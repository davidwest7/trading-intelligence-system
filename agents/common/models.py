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
