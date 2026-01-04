"""
A/B Testing Framework for ERC Trading System

Provides:
- Strategy variant testing
- Statistical significance testing
- Performance comparison
- Automatic winner selection

CRITICAL: Don't change strategies without data!
A/B testing provides statistical proof that changes help.
"""

import numpy as np
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from scipy import stats
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """Test status"""
    RUNNING = "running"
    CONCLUDED = "concluded"
    STOPPED = "stopped"


class Winner(Enum):
    """Test winner"""
    CONTROL = "control"
    VARIANT = "variant"
    NO_DIFFERENCE = "no_difference"
    INCONCLUSIVE = "inconclusive"


@dataclass
class StrategyVariant:
    """A strategy variant for testing"""
    name: str
    config: Dict
    description: str = ""


@dataclass
class TestResult:
    """Result from a single trade/decision"""
    variant: str
    timestamp: datetime
    pnl: float
    success: bool
    metadata: Dict = field(default_factory=dict)


@dataclass
class ABTestMetrics:
    """Metrics for an A/B test variant"""
    variant: str
    sample_size: int
    total_pnl: float
    avg_pnl: float
    std_pnl: float
    win_rate: float
    sharpe_ratio: float
    profit_factor: float


@dataclass
class ABTestAnalysis:
    """Statistical analysis of A/B test"""
    test_id: str
    control_metrics: ABTestMetrics
    variant_metrics: ABTestMetrics
    pvalue: float
    confidence: float
    effect_size: float  # Cohen's d
    winner: Winner
    is_significant: bool
    sample_size_needed: int
    recommendation: str


@dataclass
class ABTest:
    """A/B test definition"""
    test_id: str
    name: str
    control: StrategyVariant
    variant: StrategyVariant
    traffic_split: float = 0.5  # Fraction going to variant
    min_samples: int = 100
    significance_level: float = 0.05
    status: TestStatus = TestStatus.RUNNING
    created_at: datetime = field(default_factory=datetime.now)
    results: List[TestResult] = field(default_factory=list)
    concluded_at: Optional[datetime] = None
    winner: Optional[Winner] = None


class ABTestingFramework:
    """
    A/B testing framework for comparing strategy variants.

    Usage:
        framework = ABTestingFramework()

        # Create test
        test = framework.create_test(
            name="Kelly sizing test",
            control=StrategyVariant("full_kelly", {"kelly_fraction": 1.0}),
            variant=StrategyVariant("half_kelly", {"kelly_fraction": 0.5})
        )

        # During trading, get variant to use
        variant = framework.get_variant(test.test_id)

        # Record results
        framework.record_result(test.test_id, variant, pnl=10.5)

        # Check if we have a winner
        analysis = framework.analyze(test.test_id)
    """

    def __init__(self, data_dir: str = "data/ab_tests"):
        self.tests: Dict[str, ABTest] = {}
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def create_test(self, name: str,
                    control: StrategyVariant,
                    variant: StrategyVariant,
                    traffic_split: float = 0.5,
                    min_samples: int = 100,
                    significance_level: float = 0.05) -> ABTest:
        """
        Create a new A/B test.

        Args:
            name: Test name
            control: Control (baseline) strategy
            variant: Variant strategy to test
            traffic_split: Fraction of traffic to variant (0-1)
            min_samples: Minimum samples before concluding
            significance_level: P-value threshold for significance

        Returns:
            Created ABTest
        """
        test_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        test = ABTest(
            test_id=test_id,
            name=name,
            control=control,
            variant=variant,
            traffic_split=traffic_split,
            min_samples=min_samples,
            significance_level=significance_level
        )

        self.tests[test_id] = test

        logger.info(
            f"Created A/B test '{name}': "
            f"{control.name} vs {variant.name} "
            f"({traffic_split:.0%} to variant)"
        )

        return test

    def get_variant(self, test_id: str) -> str:
        """
        Get which variant to use for this request.

        Uses random assignment based on traffic split.

        Args:
            test_id: Test identifier

        Returns:
            Variant name ('control' or 'variant')
        """
        test = self.tests.get(test_id)

        if test is None or test.status != TestStatus.RUNNING:
            return 'control'

        if np.random.random() < test.traffic_split:
            return test.variant.name
        else:
            return test.control.name

    def get_config(self, test_id: str, variant_name: str) -> Dict:
        """Get configuration for a variant"""
        test = self.tests.get(test_id)

        if test is None:
            return {}

        if variant_name == test.variant.name:
            return test.variant.config
        else:
            return test.control.config

    def record_result(self, test_id: str, variant_name: str,
                      pnl: float, success: bool = None,
                      metadata: Optional[Dict] = None):
        """
        Record a test result.

        Args:
            test_id: Test identifier
            variant_name: Which variant was used
            pnl: Profit/loss for this trade
            success: Whether trade was successful (auto-determined if None)
            metadata: Additional data about the trade
        """
        test = self.tests.get(test_id)

        if test is None:
            logger.warning(f"Test {test_id} not found")
            return

        if success is None:
            success = pnl > 0

        result = TestResult(
            variant=variant_name,
            timestamp=datetime.now(),
            pnl=pnl,
            success=success,
            metadata=metadata or {}
        )

        test.results.append(result)

        # Check if we should auto-conclude
        self._check_auto_conclude(test)

    def analyze(self, test_id: str) -> Optional[ABTestAnalysis]:
        """
        Analyze A/B test results.

        Performs statistical testing to determine if there's a
        significant difference between variants.

        Args:
            test_id: Test identifier

        Returns:
            ABTestAnalysis with statistical results
        """
        test = self.tests.get(test_id)

        if test is None:
            return None

        # Separate results by variant
        control_results = [r for r in test.results if r.variant == test.control.name]
        variant_results = [r for r in test.results if r.variant == test.variant.name]

        if len(control_results) < 10 or len(variant_results) < 10:
            return ABTestAnalysis(
                test_id=test_id,
                control_metrics=self._calculate_metrics(test.control.name, control_results),
                variant_metrics=self._calculate_metrics(test.variant.name, variant_results),
                pvalue=1.0,
                confidence=0.0,
                effect_size=0.0,
                winner=Winner.INCONCLUSIVE,
                is_significant=False,
                sample_size_needed=test.min_samples,
                recommendation="Need more data"
            )

        # Calculate metrics
        control_metrics = self._calculate_metrics(test.control.name, control_results)
        variant_metrics = self._calculate_metrics(test.variant.name, variant_results)

        # Statistical test (Welch's t-test for unequal variances)
        control_pnls = [r.pnl for r in control_results]
        variant_pnls = [r.pnl for r in variant_results]

        t_stat, pvalue = stats.ttest_ind(variant_pnls, control_pnls, equal_var=False)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            (control_metrics.std_pnl**2 + variant_metrics.std_pnl**2) / 2
        )
        effect_size = (variant_metrics.avg_pnl - control_metrics.avg_pnl) / pooled_std if pooled_std > 0 else 0

        # Determine winner
        is_significant = pvalue < test.significance_level

        if not is_significant:
            winner = Winner.INCONCLUSIVE
            recommendation = f"No significant difference (p={pvalue:.3f}). Need more samples."
        elif effect_size > 0.2:
            winner = Winner.VARIANT
            recommendation = f"Variant is significantly better! Effect size: {effect_size:.2f}"
        elif effect_size < -0.2:
            winner = Winner.CONTROL
            recommendation = f"Control is significantly better. Effect size: {effect_size:.2f}"
        else:
            winner = Winner.NO_DIFFERENCE
            recommendation = "Statistically significant but small practical difference."

        # Sample size needed for 80% power
        sample_size_needed = self._calculate_sample_size(
            effect_size=0.2,  # Minimum detectable effect
            alpha=test.significance_level,
            power=0.8
        )

        return ABTestAnalysis(
            test_id=test_id,
            control_metrics=control_metrics,
            variant_metrics=variant_metrics,
            pvalue=pvalue,
            confidence=1 - pvalue,
            effect_size=effect_size,
            winner=winner,
            is_significant=is_significant,
            sample_size_needed=sample_size_needed,
            recommendation=recommendation
        )

    def _calculate_metrics(self, variant: str,
                           results: List[TestResult]) -> ABTestMetrics:
        """Calculate metrics for a variant"""
        if not results:
            return ABTestMetrics(
                variant=variant,
                sample_size=0,
                total_pnl=0,
                avg_pnl=0,
                std_pnl=0,
                win_rate=0,
                sharpe_ratio=0,
                profit_factor=0
            )

        pnls = [r.pnl for r in results]
        wins = [r for r in results if r.success]
        losses = [r for r in results if not r.success]

        win_pnls = [r.pnl for r in wins if r.pnl > 0]
        loss_pnls = [abs(r.pnl) for r in losses if r.pnl < 0]

        avg_pnl = np.mean(pnls)
        std_pnl = np.std(pnls) if len(pnls) > 1 else 0

        return ABTestMetrics(
            variant=variant,
            sample_size=len(results),
            total_pnl=sum(pnls),
            avg_pnl=avg_pnl,
            std_pnl=std_pnl,
            win_rate=len(wins) / len(results),
            sharpe_ratio=avg_pnl / std_pnl * np.sqrt(252) if std_pnl > 0 else 0,
            profit_factor=sum(win_pnls) / sum(loss_pnls) if loss_pnls else float('inf')
        )

    def _calculate_sample_size(self, effect_size: float,
                               alpha: float = 0.05,
                               power: float = 0.8) -> int:
        """Calculate required sample size per group"""
        # Using approximation formula
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)

        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2

        return int(np.ceil(n))

    def _check_auto_conclude(self, test: ABTest):
        """Check if test should auto-conclude"""
        if test.status != TestStatus.RUNNING:
            return

        control_count = sum(1 for r in test.results if r.variant == test.control.name)
        variant_count = sum(1 for r in test.results if r.variant == test.variant.name)

        # Need minimum samples in both groups
        if control_count < test.min_samples or variant_count < test.min_samples:
            return

        # Analyze
        analysis = self.analyze(test.test_id)

        if analysis and analysis.is_significant:
            # Early stopping for clear winner
            if abs(analysis.effect_size) > 0.5:  # Large effect
                self.conclude_test(test.test_id, analysis.winner)

    def conclude_test(self, test_id: str, winner: Winner):
        """Conclude a test with a winner"""
        test = self.tests.get(test_id)

        if test is None:
            return

        test.status = TestStatus.CONCLUDED
        test.winner = winner
        test.concluded_at = datetime.now()

        logger.info(
            f"Test '{test.name}' concluded: Winner = {winner.value}"
        )

        # Save results
        self._save_test(test)

    def stop_test(self, test_id: str):
        """Stop a test without concluding"""
        test = self.tests.get(test_id)

        if test:
            test.status = TestStatus.STOPPED
            self._save_test(test)

    def _save_test(self, test: ABTest):
        """Save test results to file"""
        filepath = self.data_dir / f"{test.test_id}.json"

        data = {
            'test_id': test.test_id,
            'name': test.name,
            'control': {
                'name': test.control.name,
                'config': test.control.config
            },
            'variant': {
                'name': test.variant.name,
                'config': test.variant.config
            },
            'traffic_split': test.traffic_split,
            'status': test.status.value,
            'winner': test.winner.value if test.winner else None,
            'created_at': test.created_at.isoformat(),
            'concluded_at': test.concluded_at.isoformat() if test.concluded_at else None,
            'results': [
                {
                    'variant': r.variant,
                    'timestamp': r.timestamp.isoformat(),
                    'pnl': r.pnl,
                    'success': r.success
                }
                for r in test.results
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def get_running_tests(self) -> List[ABTest]:
        """Get all running tests"""
        return [t for t in self.tests.values() if t.status == TestStatus.RUNNING]

    def get_test_summary(self, test_id: str) -> Optional[Dict]:
        """Get summary of test progress"""
        test = self.tests.get(test_id)

        if test is None:
            return None

        analysis = self.analyze(test_id)

        control_count = sum(1 for r in test.results if r.variant == test.control.name)
        variant_count = sum(1 for r in test.results if r.variant == test.variant.name)

        return {
            'test_id': test_id,
            'name': test.name,
            'status': test.status.value,
            'control_samples': control_count,
            'variant_samples': variant_count,
            'min_samples': test.min_samples,
            'progress': min(control_count, variant_count) / test.min_samples,
            'current_leader': analysis.winner.value if analysis else 'inconclusive',
            'pvalue': analysis.pvalue if analysis else 1.0
        }


def test_ab_framework():
    """Test the A/B testing framework"""
    print("=== A/B Testing Framework Test ===\n")

    framework = ABTestingFramework()

    # Create test
    test = framework.create_test(
        name="Kelly Fraction Test",
        control=StrategyVariant("full_kelly", {"kelly_fraction": 1.0}),
        variant=StrategyVariant("half_kelly", {"kelly_fraction": 0.5}),
        min_samples=50,
        traffic_split=0.5
    )

    print(f"Created test: {test.test_id}")

    # Simulate trading with both variants
    np.random.seed(42)

    print("\nSimulating trades...")

    # Control has higher variance (full kelly)
    control_mean, control_std = 2.0, 8.0
    # Variant has lower variance (half kelly)
    variant_mean, variant_std = 1.8, 4.0

    for i in range(100):
        # Get variant for this trade
        variant = framework.get_variant(test.test_id)

        # Simulate PnL
        if variant == "half_kelly":
            pnl = np.random.normal(variant_mean, variant_std)
        else:
            pnl = np.random.normal(control_mean, control_std)

        # Record result
        framework.record_result(test.test_id, variant, pnl)

    # Analyze results
    analysis = framework.analyze(test.test_id)

    if analysis:
        print(f"\n--- Test Results ---")
        print(f"Control ({test.control.name}):")
        print(f"  Samples: {analysis.control_metrics.sample_size}")
        print(f"  Avg PnL: ${analysis.control_metrics.avg_pnl:.2f}")
        print(f"  Std PnL: ${analysis.control_metrics.std_pnl:.2f}")
        print(f"  Sharpe: {analysis.control_metrics.sharpe_ratio:.2f}")

        print(f"\nVariant ({test.variant.name}):")
        print(f"  Samples: {analysis.variant_metrics.sample_size}")
        print(f"  Avg PnL: ${analysis.variant_metrics.avg_pnl:.2f}")
        print(f"  Std PnL: ${analysis.variant_metrics.std_pnl:.2f}")
        print(f"  Sharpe: {analysis.variant_metrics.sharpe_ratio:.2f}")

        print(f"\n--- Statistical Analysis ---")
        print(f"P-value: {analysis.pvalue:.4f}")
        print(f"Effect size: {analysis.effect_size:.3f}")
        print(f"Significant: {analysis.is_significant}")
        print(f"Winner: {analysis.winner.value}")
        print(f"\nRecommendation: {analysis.recommendation}")


if __name__ == "__main__":
    test_ab_framework()
