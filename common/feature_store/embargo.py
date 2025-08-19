"""
Multi-Event Embargo System
Prevents data leakage through comprehensive embargo management and universe drift tracking
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque
import json
import hashlib


class EmbargoType(str, Enum):
    """Types of embargo events"""
    EARNINGS = "earnings"
    SPLIT = "split"
    DIVIDEND = "dividend"
    MERGER = "merger"
    IPO = "ipo"
    DELISTING = "delisting"
    CORPORATE_ACTION = "corporate_action"
    NEWS_EVENT = "news_event"
    REGULATORY = "regulatory"
    MARKET_HALT = "market_halt"


@dataclass
class EmbargoEvent:
    """Individual embargo event"""
    event_id: str
    event_type: EmbargoType
    symbol: str
    event_date: datetime
    embargo_start: datetime
    embargo_end: datetime
    embargo_horizon: int  # Days before event
    embargo_duration: int  # Days after event
    confidence: float
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_active(self, current_time: datetime) -> bool:
        """Check if embargo is currently active"""
        return self.embargo_start <= current_time <= self.embargo_end
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "symbol": self.symbol,
            "event_date": self.event_date.isoformat(),
            "embargo_start": self.embargo_start.isoformat(),
            "embargo_end": self.embargo_end.isoformat(),
            "embargo_horizon": self.embargo_horizon,
            "embargo_duration": self.embargo_duration,
            "confidence": self.confidence,
            "source": self.source,
            "metadata": self.metadata
        }


@dataclass
class UniverseDrift:
    """Universe composition drift tracking"""
    timestamp: datetime
    universe_id: str
    symbols_added: Set[str] = field(default_factory=set)
    symbols_removed: Set[str] = field(default_factory=set)
    symbols_modified: Set[str] = field(default_factory=set)
    drift_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "universe_id": self.universe_id,
            "symbols_added": list(self.symbols_added),
            "symbols_removed": list(self.symbols_removed),
            "symbols_modified": list(self.symbols_modified),
            "drift_score": self.drift_score,
            "metadata": self.metadata
        }


class MultiEventEmbargoManager:
    """
    Comprehensive embargo management system with universe drift tracking
    
    Features:
    - Multi-horizon embargo management
    - Corporate action tracking
    - Universe drift detection
    - Purged K-fold cross-validation support
    - Event-specific embargo rules
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Embargo configuration
        self.default_embargo_horizons = {
            EmbargoType.EARNINGS: 7,      # 7 days before earnings
            EmbargoType.SPLIT: 3,         # 3 days before split
            EmbargoType.DIVIDEND: 2,      # 2 days before ex-dividend
            EmbargoType.MERGER: 14,       # 14 days before merger
            EmbargoType.IPO: 30,          # 30 days before IPO
            EmbargoType.DELISTING: 7,     # 7 days before delisting
            EmbargoType.CORPORATE_ACTION: 5,
            EmbargoType.NEWS_EVENT: 1,    # 1 day before news
            EmbargoType.REGULATORY: 3,    # 3 days before regulatory event
            EmbargoType.MARKET_HALT: 0    # Immediate embargo
        }
        
        self.default_embargo_durations = {
            EmbargoType.EARNINGS: 3,      # 3 days after earnings
            EmbargoType.SPLIT: 1,         # 1 day after split
            EmbargoType.DIVIDEND: 1,      # 1 day after ex-dividend
            EmbargoType.MERGER: 7,        # 7 days after merger
            EmbargoType.IPO: 14,          # 14 days after IPO
            EmbargoType.DELISTING: 30,    # 30 days after delisting
            EmbargoType.CORPORATE_ACTION: 3,
            EmbargoType.NEWS_EVENT: 2,    # 2 days after news
            EmbargoType.REGULATORY: 5,    # 5 days after regulatory event
            EmbargoType.MARKET_HALT: 1    # 1 day after halt
        }
        
        # Storage
        self.active_embargos: Dict[str, EmbargoEvent] = {}
        self.embargo_history: List[EmbargoEvent] = []
        self.universe_drift_history: List[UniverseDrift] = []
        self.universe_snapshots: Dict[str, Set[str]] = {}
        
        # Performance tracking
        self.embargo_violations = 0
        self.total_checks = 0
        self.drift_alerts = 0
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def add_embargo_event(self, event: EmbargoEvent) -> bool:
        """Add a new embargo event"""
        try:
            # Validate embargo dates
            if event.embargo_start >= event.embargo_end:
                self.logger.error(f"Invalid embargo dates for event {event.event_id}")
                return False
            
            # Check for overlapping embargos
            overlapping = await self._check_overlapping_embargos(event)
            if overlapping:
                self.logger.warning(f"Overlapping embargo detected for {event.symbol}")
                # Merge or extend existing embargo
                await self._merge_embargos(event, overlapping)
            else:
                self.active_embargos[event.event_id] = event
                self.embargo_history.append(event)
            
            self.logger.info(f"Added embargo event: {event.event_id} for {event.symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding embargo event: {e}")
            return False
    
    async def check_embargo_status(self, symbol: str, timestamp: datetime, 
                                 feature_horizon: int = 0) -> Tuple[bool, List[str]]:
        """
        Check if symbol is under embargo at given timestamp
        
        Args:
            symbol: Stock symbol
            timestamp: Current timestamp
            feature_horizon: How many days ahead the feature looks
            
        Returns:
            (is_embargoed, list_of_active_embargo_reasons)
        """
        self.total_checks += 1
        
        try:
            active_embargos = []
            
            # Check all active embargos for this symbol
            for event_id, event in self.active_embargos.items():
                if event.symbol == symbol and event.is_active(timestamp):
                    # Adjust for feature horizon
                    adjusted_end = event.embargo_end + timedelta(days=feature_horizon)
                    if timestamp <= adjusted_end:
                        active_embargos.append(f"{event.event_type.value}: {event.event_id}")
            
            # Check universe drift
            drift_issues = await self._check_universe_drift(symbol, timestamp)
            if drift_issues:
                active_embargos.extend(drift_issues)
            
            is_embargoed = len(active_embargos) > 0
            
            if is_embargoed:
                self.embargo_violations += 1
                self.logger.warning(f"Embargo violation for {symbol}: {active_embargos}")
            
            return is_embargoed, active_embargos
            
        except Exception as e:
            self.logger.error(f"Error checking embargo status: {e}")
            return True, ["error_checking_embargo"]  # Fail safe
    
    async def create_purged_kfold_splits(self, data: pd.DataFrame, n_splits: int = 5,
                                       embargo_horizon: int = 30) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create purged K-fold cross-validation splits with embargo protection
        
        Args:
            data: DataFrame with timestamp index
            n_splits: Number of CV splits
            embargo_horizon: Days to purge around events
            
        Returns:
            List of (train, test) splits
        """
        try:
            splits = []
            data_sorted = data.sort_index()
            
            # Calculate split boundaries
            split_size = len(data_sorted) // n_splits
            
            for i in range(n_splits):
                # Define test period
                test_start_idx = i * split_size
                test_end_idx = (i + 1) * split_size if i < n_splits - 1 else len(data_sorted)
                
                test_start = data_sorted.index[test_start_idx]
                test_end = data_sorted.index[test_end_idx - 1]
                
                # Purge embargo periods
                purge_start = test_start - timedelta(days=embargo_horizon)
                purge_end = test_end + timedelta(days=embargo_horizon)
                
                # Create train/test split with purging
                train_mask = (data_sorted.index < purge_start) | (data_sorted.index > purge_end)
                test_mask = (data_sorted.index >= test_start) & (data_sorted.index <= test_end)
                
                train_data = data_sorted[train_mask]
                test_data = data_sorted[test_mask]
                
                splits.append((train_data, test_data))
                
                self.logger.info(f"Created purged split {i+1}: train={len(train_data)}, test={len(test_data)}")
            
            return splits
            
        except Exception as e:
            self.logger.error(f"Error creating purged K-fold splits: {e}")
            return []
    
    async def track_universe_drift(self, universe_id: str, current_symbols: Set[str],
                                 timestamp: datetime) -> UniverseDrift:
        """Track universe composition drift"""
        try:
            previous_symbols = self.universe_snapshots.get(universe_id, set())
            
            # Calculate drift
            symbols_added = current_symbols - previous_symbols
            symbols_removed = previous_symbols - current_symbols
            symbols_modified = symbols_added | symbols_removed
            
            # Calculate drift score
            total_symbols = len(previous_symbols | current_symbols)
            drift_score = len(symbols_modified) / total_symbols if total_symbols > 0 else 0.0
            
            # Create drift record
            drift = UniverseDrift(
                timestamp=timestamp,
                universe_id=universe_id,
                symbols_added=symbols_added,
                symbols_removed=symbols_removed,
                symbols_modified=symbols_modified,
                drift_score=drift_score,
                metadata={
                    "total_symbols": total_symbols,
                    "drift_percentage": drift_score * 100
                }
            )
            
            # Store drift history
            self.universe_drift_history.append(drift)
            self.universe_snapshots[universe_id] = current_symbols.copy()
            
            # Alert on significant drift
            if drift_score > 0.1:  # 10% drift threshold
                self.drift_alerts += 1
                self.logger.warning(f"Significant universe drift detected: {drift_score:.2%}")
            
            return drift
            
        except Exception as e:
            self.logger.error(f"Error tracking universe drift: {e}")
            return UniverseDrift(timestamp=timestamp, universe_id=universe_id)
    
    async def get_embargo_summary(self) -> Dict[str, Any]:
        """Get summary of embargo system status"""
        try:
            active_count = len(self.active_embargos)
            total_events = len(self.embargo_history)
            
            # Count by type
            type_counts = defaultdict(int)
            for event in self.embargo_history:
                type_counts[event.event_type.value] += 1
            
            # Calculate violation rate
            violation_rate = self.embargo_violations / self.total_checks if self.total_checks > 0 else 0.0
            
            return {
                "active_embargos": active_count,
                "total_events": total_events,
                "events_by_type": dict(type_counts),
                "violation_rate": violation_rate,
                "total_checks": self.total_checks,
                "drift_alerts": self.drift_alerts,
                "universe_drifts": len(self.universe_drift_history)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting embargo summary: {e}")
            return {}
    
    async def cleanup_expired_embargos(self, current_time: datetime) -> int:
        """Remove expired embargo events"""
        try:
            expired_count = 0
            expired_ids = []
            
            for event_id, event in self.active_embargos.items():
                if event.embargo_end < current_time:
                    expired_ids.append(event_id)
                    expired_count += 1
            
            for event_id in expired_ids:
                del self.active_embargos[event_id]
            
            if expired_count > 0:
                self.logger.info(f"Cleaned up {expired_count} expired embargos")
            
            return expired_count
            
        except Exception as e:
            self.logger.error(f"Error cleaning up expired embargos: {e}")
            return 0
    
    async def _check_overlapping_embargos(self, new_event: EmbargoEvent) -> List[EmbargoEvent]:
        """Check for overlapping embargo events"""
        overlapping = []
        
        for event in self.active_embargos.values():
            if (event.symbol == new_event.symbol and 
                event.embargo_start <= new_event.embargo_end and 
                new_event.embargo_start <= event.embargo_end):
                overlapping.append(event)
        
        return overlapping
    
    async def _merge_embargos(self, new_event: EmbargoEvent, overlapping: List[EmbargoEvent]):
        """Merge overlapping embargo events"""
        if not overlapping:
            return
        
        # Find the most restrictive embargo
        earliest_start = min([e.embargo_start for e in overlapping] + [new_event.embargo_start])
        latest_end = max([e.embargo_end for e in overlapping] + [new_event.embargo_end])
        
        # Create merged embargo
        merged_event = EmbargoEvent(
            event_id=f"merged_{hashlib.md5(str(earliest_start).encode()).hexdigest()[:8]}",
            event_type=new_event.event_type,
            symbol=new_event.symbol,
            event_date=new_event.event_date,
            embargo_start=earliest_start,
            embargo_end=latest_end,
            embargo_horizon=(new_event.event_date - earliest_start).days,
            embargo_duration=(latest_end - new_event.event_date).days,
            confidence=max([e.confidence for e in overlapping] + [new_event.confidence]),
            source="merged",
            metadata={"merged_from": [e.event_id for e in overlapping] + [new_event.event_id]}
        )
        
        # Remove old embargos and add merged one
        for event in overlapping:
            if event.event_id in self.active_embargos:
                del self.active_embargos[event.event_id]
        
        self.active_embargos[merged_event.event_id] = merged_event
        self.embargo_history.append(merged_event)
    
    async def _check_universe_drift(self, symbol: str, timestamp: datetime) -> List[str]:
        """Check for universe drift issues"""
        drift_issues = []
        
        # Check recent drift history
        recent_drifts = [d for d in self.universe_drift_history[-10:] 
                        if timestamp - d.timestamp < timedelta(days=30)]
        
        for drift in recent_drifts:
            if symbol in drift.symbols_removed:
                drift_issues.append(f"symbol_removed_from_universe_{drift.universe_id}")
            elif symbol in drift.symbols_added:
                drift_issues.append(f"symbol_recently_added_to_universe_{drift.universe_id}")
        
        return drift_issues


# Factory function for easy integration
async def create_embargo_manager(config: Optional[Dict[str, Any]] = None) -> MultiEventEmbargoManager:
    """Create and initialize embargo manager"""
    manager = MultiEventEmbargoManager(config)
    
    # Add default corporate action embargoes
    await manager.add_embargo_event(EmbargoEvent(
        event_id="default_earnings",
        event_type=EmbargoType.EARNINGS,
        symbol="*",  # Wildcard for all symbols
        event_date=datetime.now(),
        embargo_start=datetime.now() - timedelta(days=7),
        embargo_end=datetime.now() + timedelta(days=3),
        embargo_horizon=7,
        embargo_duration=3,
        confidence=0.9,
        source="default"
    ))
    
    return manager
