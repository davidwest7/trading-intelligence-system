"""
Backtesting Module
=================

Comprehensive backtesting system with:
- Polygon Pro data integration
- S3 data lake architecture
- Realistic execution simulation
- Walk-forward analysis
- Performance attribution
"""

from .engine import BacktestEngine
from .execution import ExecutionEngine
from .data_ingestion import PolygonDataIngestion
from .metrics import BacktestMetrics

__all__ = [
    'BacktestEngine',
    'ExecutionEngine', 
    'PolygonDataIngestion',
    'BacktestMetrics'
]
