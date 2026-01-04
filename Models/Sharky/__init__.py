"""
Sharky Strategy Package

Implements Sharky6999's near-certainty scalping strategy
Combined with intelligent capital allocation
"""

from .sharky_scanner import SharkyScanner, SharkyOpportunity, CertaintyAnalyzer
from .allocation_engine import CapitalAllocationEngine, AllocationDecision, StrategyPerformance

__all__ = [
    'SharkyScanner',
    'SharkyOpportunity', 
    'CertaintyAnalyzer',
    'CapitalAllocationEngine',
    'AllocationDecision',
    'StrategyPerformance'
]

__version__ = '1.0.0'
__author__ = 'ERC Trading System'