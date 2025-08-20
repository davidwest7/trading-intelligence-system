"""
Common models and base classes for all agents
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid

from schemas.contracts import Signal, SignalType, RegimeType, HorizonType, DirectionType


class BaseAgent(ABC):
    """
    Base class for all trading agents with uncertainty quantification
    
    All agents must emit signals with (μ, σ, horizon) uncertainty quantification
    """
    
    def __init__(self, name: str, agent_type: SignalType, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.agent_type = agent_type
        self.config = config or {}
        self.created_at = datetime.now()
        self.is_active = True
        
        # Performance tracking
        self.signals_generated = 0
        self.avg_confidence = 0.0
        self.last_signal_time = None
        
    @abstractmethod
    async def generate_signals(self, symbols: List[str], **kwargs) -> List[Signal]:
        """
        Generate uncertainty-quantified signals for given symbols
        
        Args:
            symbols: List of symbols to analyze
            **kwargs: Additional parameters
            
        Returns:
            List of Signal objects with (μ, σ, horizon) uncertainty quantification
        """
        pass
    
    async def process(self, symbols: List[str], **kwargs) -> List[Signal]:
        """
        Main processing method that generates signals
        
        This is the standard interface all agents must implement
        """
        signals = await self.generate_signals(symbols, **kwargs)
        
        # Update performance metrics
        self.signals_generated += len(signals)
        if signals:
            self.avg_confidence = sum(s.confidence for s in signals) / len(signals)
            self.last_signal_time = datetime.now()
        
        return signals
    
    def detect_regime(self, market_data: Dict[str, Any]) -> RegimeType:
        """
        Detect current market regime based on market data
        
        Default implementation - can be overridden by specific agents
        """
        # Simple regime detection based on volatility
        volatility = market_data.get('volatility', 0.15)
        
        if volatility > 0.25:
            return RegimeType.HIGH_VOL
        elif volatility < 0.10:
            return RegimeType.LOW_VOL
        else:
            return RegimeType.RISK_ON  # Default
    
    def calculate_uncertainty(self, base_signal: float, confidence: float, 
                            market_conditions: Dict[str, Any]) -> float:
        """
        Calculate uncertainty (sigma) based on signal and market conditions
        
        Args:
            base_signal: The base signal strength
            confidence: Agent confidence in the signal
            market_conditions: Current market conditions
            
        Returns:
            Uncertainty (standard deviation) of the signal
        """
        # Base uncertainty inversely related to confidence
        base_uncertainty = (1.0 - confidence) * 0.05
        
        # Adjust for market conditions
        volatility = market_conditions.get('volatility', 0.15)
        liquidity = market_conditions.get('liquidity', 1.0)
        
        # Higher volatility and lower liquidity increase uncertainty
        volatility_adjustment = volatility * 0.1
        liquidity_adjustment = max(0, (1.0 - liquidity) * 0.02)
        
        total_uncertainty = base_uncertainty + volatility_adjustment + liquidity_adjustment
        
        # Ensure uncertainty is reasonable (between 0.005 and 0.10)
        return max(0.005, min(0.10, total_uncertainty))
    
    def determine_horizon(self, signal_strength: float, agent_type: SignalType) -> HorizonType:
        """
        Determine appropriate time horizon based on signal characteristics
        
        Args:
            signal_strength: Strength of the signal
            agent_type: Type of agent generating the signal
            
        Returns:
            Appropriate time horizon for the signal
        """
        # Agent-specific horizon preferences
        if agent_type in [SignalType.TECHNICAL, SignalType.FLOW]:
            if abs(signal_strength) > 0.05:
                return HorizonType.INTRADAY
            else:
                return HorizonType.SHORT_TERM
        elif agent_type in [SignalType.SENTIMENT, SignalType.MONEY_FLOWS]:
            return HorizonType.SHORT_TERM
        elif agent_type in [SignalType.MACRO, SignalType.UNDERVALUED, SignalType.VALUE]:
            return HorizonType.MEDIUM_TERM
        else:
            return HorizonType.SHORT_TERM  # Default
    
    def create_signal(self, symbol: str, mu: float, confidence: float,
                     market_conditions: Dict[str, Any], trace_id: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> Signal:
        """
        Create a standardized signal with uncertainty quantification
        
        Args:
            symbol: Trading symbol
            mu: Expected return (mean)
            confidence: Agent confidence [0,1]
            market_conditions: Current market conditions
            trace_id: Optional trace ID for tracking
            metadata: Optional additional metadata
            
        Returns:
            Standardized Signal object
        """
        # Calculate uncertainty
        sigma = self.calculate_uncertainty(mu, confidence, market_conditions)
        
        # Determine horizon
        horizon = self.determine_horizon(mu, self.agent_type)
        
        # Detect regime
        regime = self.detect_regime(market_conditions)
        
        # Determine direction
        if mu > 0.01:
            direction = DirectionType.LONG
        elif mu < -0.01:
            direction = DirectionType.SHORT
        else:
            direction = DirectionType.NEUTRAL
        
        # Create signal
        signal = Signal(
            trace_id=trace_id or str(uuid.uuid4()),
            agent_id=self.name,
            agent_type=self.agent_type,
            symbol=symbol,
            mu=mu,
            sigma=sigma,
            confidence=confidence,
            horizon=horizon,
            regime=regime,
            direction=direction,
            model_version=self.config.get('model_version', '1.0.0'),
            feature_version=self.config.get('feature_version', '1.0.0'),
            metadata=metadata or {}
        )
        
        return signal
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status with performance metrics"""
        return {
            "name": self.name,
            "agent_type": self.agent_type.value,
            "active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "signals_generated": self.signals_generated,
            "avg_confidence": self.avg_confidence,
            "last_signal_time": self.last_signal_time.isoformat() if self.last_signal_time else None,
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
