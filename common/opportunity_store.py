"""
Centralized Opportunity Store - Collects and manages opportunities from all agents
"""

import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import threading
import os

@dataclass
class Opportunity:
    """Standardized opportunity data structure"""
    id: str
    ticker: str
    agent_type: str  # technical, value, flow, insider, etc.
    opportunity_type: str  # Technical, Value, Flow, Insider, etc.
    entry_reason: str
    upside_potential: float
    confidence: float
    time_horizon: str
    discovered_at: datetime
    job_id: str
    raw_data: Dict[str, Any]  # Original agent data
    priority_score: float = 0.0
    status: str = "active"  # active, expired, executed
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)
        data['discovered_at'] = self.discovered_at.isoformat()
        return data

class OpportunityStore:
    """Centralized storage for all trading opportunities"""
    
    def __init__(self, db_path: str = "opportunities.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS opportunities (
                    id TEXT PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    agent_type TEXT NOT NULL,
                    opportunity_type TEXT NOT NULL,
                    entry_reason TEXT NOT NULL,
                    upside_potential REAL NOT NULL,
                    confidence REAL NOT NULL,
                    time_horizon TEXT NOT NULL,
                    discovered_at TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    raw_data TEXT NOT NULL,
                    priority_score REAL DEFAULT 0.0,
                    status TEXT DEFAULT 'active',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    
    def add_opportunity(self, opportunity: Opportunity) -> bool:
        """Add a new opportunity to the store"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO opportunities 
                        (id, ticker, agent_type, opportunity_type, entry_reason, 
                         upside_potential, confidence, time_horizon, discovered_at, 
                         job_id, raw_data, priority_score, status)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        opportunity.id,
                        opportunity.ticker,
                        opportunity.agent_type,
                        opportunity.opportunity_type,
                        opportunity.entry_reason,
                        opportunity.upside_potential,
                        opportunity.confidence,
                        opportunity.time_horizon,
                        opportunity.discovered_at.isoformat(),
                        opportunity.job_id,
                        json.dumps(opportunity.raw_data),
                        opportunity.priority_score,
                        opportunity.status
                    ))
                    conn.commit()
            return True
        except Exception as e:
            print(f"Error adding opportunity: {e}")
            return False
    
    def add_opportunities_from_agent(self, agent_type: str, job_id: str, 
                                   opportunities: List[Dict[str, Any]]) -> int:
        """Add multiple opportunities from an agent"""
        added_count = 0
        for opp_data in opportunities:
            try:
                opportunity = Opportunity(
                    id=f"{agent_type}_{job_id}_{opp_data.get('ticker', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    ticker=opp_data.get('ticker', 'Unknown'),
                    agent_type=agent_type,
                    opportunity_type=opp_data.get('type', 'Unknown'),
                    entry_reason=opp_data.get('entry_reason', 'Not specified'),
                    upside_potential=opp_data.get('upside_potential', 0.0),
                    confidence=opp_data.get('confidence', 0.0),
                    time_horizon=opp_data.get('time_horizon', 'Unknown'),
                    discovered_at=datetime.now(),
                    job_id=job_id,
                    raw_data=opp_data
                )
                if self.add_opportunity(opportunity):
                    added_count += 1
            except Exception as e:
                print(f"Error processing opportunity: {e}")
        return added_count
    
    def get_all_opportunities(self, status: str = "active") -> List[Opportunity]:
        """Get all opportunities with optional status filter"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM opportunities 
                    WHERE status = ? 
                    ORDER BY priority_score DESC, discovered_at DESC
                """, (status,))
                
                opportunities = []
                for row in cursor.fetchall():
                    opp = Opportunity(
                        id=row[0],
                        ticker=row[1],
                        agent_type=row[2],
                        opportunity_type=row[3],
                        entry_reason=row[4],
                        upside_potential=row[5],
                        confidence=row[6],
                        time_horizon=row[7],
                        discovered_at=datetime.fromisoformat(row[8]),
                        job_id=row[9],
                        raw_data=json.loads(row[10]),
                        priority_score=row[11],
                        status=row[12]
                    )
                    opportunities.append(opp)
                return opportunities
        except Exception as e:
            print(f"Error getting opportunities: {e}")
            return []
    
    def get_top_opportunities(self, limit: int = 10, status: str = "active") -> List[Opportunity]:
        """Get top opportunities by priority score"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM opportunities 
                    WHERE status = ? 
                    ORDER BY priority_score DESC, discovered_at DESC
                    LIMIT ?
                """, (status, limit))
                
                opportunities = []
                for row in cursor.fetchall():
                    opp = Opportunity(
                        id=row[0],
                        ticker=row[1],
                        agent_type=row[2],
                        opportunity_type=row[3],
                        entry_reason=row[4],
                        upside_potential=row[5],
                        confidence=row[6],
                        time_horizon=row[7],
                        discovered_at=datetime.fromisoformat(row[8]),
                        job_id=row[9],
                        raw_data=json.loads(row[10]),
                        priority_score=row[11],
                        status=row[12]
                    )
                    opportunities.append(opp)
                return opportunities
        except Exception as e:
            print(f"Error getting top opportunities: {e}")
            return []
    
    def update_priority_scores(self, scoring_function):
        """Update priority scores for all opportunities"""
        try:
            opportunities = self.get_all_opportunities()
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    for opp in opportunities:
                        new_score = scoring_function(opp)
                        conn.execute("""
                            UPDATE opportunities 
                            SET priority_score = ? 
                            WHERE id = ?
                        """, (new_score, opp.id))
                    conn.commit()
        except Exception as e:
            print(f"Error updating priority scores: {e}")
    
    def get_opportunities_by_agent(self, agent_type: str) -> List[Opportunity]:
        """Get opportunities from a specific agent"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM opportunities 
                    WHERE agent_type = ? AND status = 'active'
                    ORDER BY priority_score DESC, discovered_at DESC
                """, (agent_type,))
                
                opportunities = []
                for row in cursor.fetchall():
                    opp = Opportunity(
                        id=row[0],
                        ticker=row[1],
                        agent_type=row[2],
                        opportunity_type=row[3],
                        entry_reason=row[4],
                        upside_potential=row[5],
                        confidence=row[6],
                        time_horizon=row[7],
                        discovered_at=datetime.fromisoformat(row[8]),
                        job_id=row[9],
                        raw_data=json.loads(row[10]),
                        priority_score=row[11],
                        status=row[12]
                    )
                    opportunities.append(opp)
                return opportunities
        except Exception as e:
            print(f"Error getting opportunities by agent: {e}")
            return []
    
    def get_signals(self) -> List[Dict[str, Any]]:
        """Get all stored signals for processing"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM opportunities 
                    WHERE status = 'active'
                    ORDER BY priority_score DESC, discovered_at DESC
                """)
                
                signals = []
                for row in cursor.fetchall():
                    signal = {
                        'signal_id': row[0],
                        'agent_type': row[2],
                        'symbol': row[1],
                        'confidence': row[6],
                        'direction': 'long' if row[5] > 0 else 'short',
                        'mu': row[5],  # upside_potential as mu
                        'sigma': 0.1,  # default uncertainty
                        'timestamp': row[8],
                        'metadata': json.loads(row[10])
                    }
                    signals.append(signal)
                return signals
        except Exception as e:
            print(f"Error getting signals: {e}")
            return []
    
    async def add_signal(self, signal) -> bool:
        """Add a signal to the opportunity store"""
        try:
            # Convert signal to opportunity
            opportunity = Opportunity(
                id=signal.signal_id,
                ticker=signal.symbol,
                agent_type=signal.agent_type.value,
                opportunity_type=signal.agent_type.value.title(),
                entry_reason=f"Signal from {signal.agent_type.value} agent",
                upside_potential=signal.mu,
                confidence=signal.confidence,
                time_horizon=signal.horizon.value,
                discovered_at=signal.timestamp,
                job_id="signal_processing",
                raw_data=signal.metadata
            )
            return self.add_opportunity(opportunity)
        except Exception as e:
            print(f"Error adding signal: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get opportunity statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Total opportunities
                total = conn.execute("SELECT COUNT(*) FROM opportunities WHERE status = 'active'").fetchone()[0]
                
                # By agent type
                agent_stats = {}
                cursor = conn.execute("""
                    SELECT agent_type, COUNT(*) 
                    FROM opportunities 
                    WHERE status = 'active' 
                    GROUP BY agent_type
                """)
                for row in cursor.fetchall():
                    agent_stats[row[0]] = row[1]
                
                # Average scores
                avg_score = conn.execute("""
                    SELECT AVG(priority_score) 
                    FROM opportunities 
                    WHERE status = 'active'
                """).fetchone()[0] or 0.0
                
                return {
                    'total_opportunities': total,
                    'by_agent_type': agent_stats,
                    'average_priority_score': avg_score
                }
        except Exception as e:
            print(f"Error getting statistics: {e}")
            return {'total_opportunities': 0, 'by_agent_type': {}, 'average_priority_score': 0.0}

# Global opportunity store instance
opportunity_store = OpportunityStore()
