"""
Probability Estimation Framework for Arbitrage Detection

Provides Bayesian probability estimation with confidence intervals
for prediction market arbitrage.

Key Concepts:
1. Market prices are NOISY signals of true probability
2. Higher volume = more informative price
3. Confidence intervals help identify true arbitrage vs noise

CRITICAL: A 2% arbitrage at 95% confidence is VERY different
from a 5% arbitrage at 60% confidence.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from scipy import stats
from scipy.special import betainc, beta as beta_fn
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProbabilityEstimate:
    """Probability estimate with uncertainty"""
    mean: float           # Point estimate
    std: float            # Standard deviation
    lower_95: float       # 95% CI lower bound
    upper_95: float       # 95% CI upper bound
    alpha: float          # Beta distribution alpha
    beta_param: float     # Beta distribution beta
    sample_size: float    # Effective sample size (from volume)
    confidence: float     # Overall confidence in estimate


class BetaProbabilityModel:
    """
    Models probability estimates using Beta distributions.

    Why Beta?
    1. Conjugate prior for binomial outcomes
    2. Naturally bounded to [0, 1]
    3. Volume translates to sample size → tighter distribution
    4. Easy Bayesian updates

    Key insight: Price alone doesn't tell us confidence.
    Price 0.50 with $1M volume vs $100 volume are VERY different.
    """

    def __init__(self, prior_strength: float = 2.0,
                 volume_scale: float = 1000.0):
        """
        Args:
            prior_strength: Strength of uniform prior (higher = more conservative)
            volume_scale: Volume normalization factor
        """
        self.prior_strength = prior_strength
        self.volume_scale = volume_scale

    def estimate_from_price(self, price: float, volume: float) -> ProbabilityEstimate:
        """
        Estimate probability distribution from price and volume.

        Higher volume = tighter distribution around price.
        Low volume = wide distribution (uncertain).

        The beta distribution parameters are:
        alpha = 1 + price * effective_n
        beta = 1 + (1 - price) * effective_n

        Where effective_n = sqrt(volume / volume_scale) to dampen extreme volumes.

        Args:
            price: Market price (0-1)
            volume: Trading volume

        Returns:
            ProbabilityEstimate with distribution parameters
        """
        # Clamp price to valid range
        price = np.clip(price, 0.001, 0.999)

        # Effective sample size from volume (square root to dampen)
        effective_n = np.sqrt(max(volume, 1) / self.volume_scale)

        # Beta distribution parameters
        # Prior: Beta(prior_strength/2, prior_strength/2) - weakly informative
        alpha = self.prior_strength / 2 + price * effective_n
        beta_param = self.prior_strength / 2 + (1 - price) * effective_n

        # Calculate statistics
        mean = alpha / (alpha + beta_param)
        variance = (alpha * beta_param) / ((alpha + beta_param) ** 2 * (alpha + beta_param + 1))
        std = np.sqrt(variance)

        # 95% credible interval
        lower_95 = stats.beta.ppf(0.025, alpha, beta_param)
        upper_95 = stats.beta.ppf(0.975, alpha, beta_param)

        # Confidence based on interval width
        ci_width = upper_95 - lower_95
        confidence = 1 - ci_width  # Narrower = more confident

        return ProbabilityEstimate(
            mean=mean,
            std=std,
            lower_95=lower_95,
            upper_95=upper_95,
            alpha=alpha,
            beta_param=beta_param,
            sample_size=effective_n,
            confidence=confidence
        )

    def estimate_from_order_book(self, best_bid: float, best_ask: float,
                                 bid_volume: float, ask_volume: float) -> ProbabilityEstimate:
        """
        Estimate probability from order book data.

        Uses volume-weighted midpoint and total liquidity.

        Args:
            best_bid: Best bid price
            best_ask: Best ask price
            bid_volume: Volume at best bid
            ask_volume: Volume at best ask

        Returns:
            ProbabilityEstimate
        """
        total_volume = bid_volume + ask_volume

        if total_volume > 0:
            # Volume-weighted price
            weighted_price = (best_bid * bid_volume + best_ask * ask_volume) / total_volume
        else:
            weighted_price = (best_bid + best_ask) / 2

        return self.estimate_from_price(weighted_price, total_volume)

    def bayesian_update(self, prior: ProbabilityEstimate,
                        new_price: float, new_volume: float) -> ProbabilityEstimate:
        """
        Update probability estimate with new information.

        Bayesian update: combine prior distribution with new evidence.

        Args:
            prior: Previous probability estimate
            new_price: New observed price
            new_volume: New trading volume

        Returns:
            Updated ProbabilityEstimate
        """
        # New evidence contribution
        effective_n = np.sqrt(max(new_volume, 1) / self.volume_scale)

        # Update parameters
        new_alpha = prior.alpha + new_price * effective_n
        new_beta = prior.beta_param + (1 - new_price) * effective_n

        # Calculate updated statistics
        mean = new_alpha / (new_alpha + new_beta)
        variance = (new_alpha * new_beta) / ((new_alpha + new_beta) ** 2 * (new_alpha + new_beta + 1))
        std = np.sqrt(variance)

        lower_95 = stats.beta.ppf(0.025, new_alpha, new_beta)
        upper_95 = stats.beta.ppf(0.975, new_alpha, new_beta)

        ci_width = upper_95 - lower_95
        confidence = 1 - ci_width

        return ProbabilityEstimate(
            mean=mean,
            std=std,
            lower_95=lower_95,
            upper_95=upper_95,
            alpha=new_alpha,
            beta_param=new_beta,
            sample_size=prior.sample_size + effective_n,
            confidence=confidence
        )


@dataclass
class ArbitrageConfidence:
    """Confidence analysis for an arbitrage opportunity"""
    is_significant: bool        # Statistically significant at given level
    probability: float          # P(true arbitrage exists)
    expected_profit: float      # Expected profit accounting for uncertainty
    var_95: float              # 95% Value at Risk
    confidence_level: float    # Our confidence in this being real
    reason: str


class ArbitrageProbabilityAnalyzer:
    """
    Analyzes arbitrage opportunities using Bayesian probability framework.

    Key insight: Arbitrage exists when P(YES) + P(NO) < 1.
    But we don't know P(YES) exactly - we have distributions.

    We need P(P_yes + P_no < 1) - probability that true arbitrage exists.
    """

    def __init__(self, model: Optional[BetaProbabilityModel] = None,
                 min_confidence: float = 0.90):
        """
        Args:
            model: Probability model to use
            min_confidence: Minimum confidence to consider opportunity valid
        """
        self.model = model or BetaProbabilityModel()
        self.min_confidence = min_confidence

    def calculate_arbitrage_probability(self, yes_estimate: ProbabilityEstimate,
                                        no_estimate: ProbabilityEstimate,
                                        n_simulations: int = 10000) -> float:
        """
        Calculate probability that true arbitrage exists.

        Uses Monte Carlo simulation to estimate P(P_yes + P_no < 1).

        Args:
            yes_estimate: Probability estimate for YES outcome
            no_estimate: Probability estimate for NO outcome
            n_simulations: Number of MC simulations

        Returns:
            Probability that arbitrage exists (0-1)
        """
        # Sample from both distributions
        yes_samples = stats.beta.rvs(
            yes_estimate.alpha, yes_estimate.beta_param, size=n_simulations
        )
        no_samples = stats.beta.rvs(
            no_estimate.alpha, no_estimate.beta_param, size=n_simulations
        )

        # Count how often sum < 1
        sum_samples = yes_samples + no_samples
        arbitrage_count = np.sum(sum_samples < 1.0)

        return arbitrage_count / n_simulations

    def calculate_profit_distribution(self, yes_estimate: ProbabilityEstimate,
                                      no_estimate: ProbabilityEstimate,
                                      position_size: float = 1.0,
                                      n_simulations: int = 10000) -> Dict:
        """
        Calculate expected profit distribution.

        For arbitrage: profit = 1 - (P_yes + P_no)

        Args:
            yes_estimate: YES probability estimate
            no_estimate: NO probability estimate
            position_size: Number of sets
            n_simulations: MC simulations

        Returns:
            Dict with profit statistics
        """
        yes_samples = stats.beta.rvs(
            yes_estimate.alpha, yes_estimate.beta_param, size=n_simulations
        )
        no_samples = stats.beta.rvs(
            no_estimate.alpha, no_estimate.beta_param, size=n_simulations
        )

        profit_per_set = 1.0 - (yes_samples + no_samples)
        total_profit = profit_per_set * position_size

        return {
            'mean': np.mean(total_profit),
            'std': np.std(total_profit),
            'median': np.median(total_profit),
            'percentile_5': np.percentile(total_profit, 5),
            'percentile_95': np.percentile(total_profit, 95),
            'prob_positive': np.mean(total_profit > 0),
            'prob_negative': np.mean(total_profit < 0),
            'var_95': -np.percentile(total_profit, 5),  # 95% VaR
            'expected_shortfall': -np.mean(total_profit[total_profit < np.percentile(total_profit, 5)])
        }

    def analyze_opportunity(self, yes_price: float, yes_volume: float,
                           no_price: float, no_volume: float,
                           position_size: float = 100.0,
                           transaction_fee: float = 0.02) -> ArbitrageConfidence:
        """
        Complete analysis of an arbitrage opportunity.

        Args:
            yes_price: Current YES price
            yes_volume: YES trading volume
            no_price: Current NO price
            no_volume: NO trading volume
            position_size: Intended position size
            transaction_fee: Fee rate

        Returns:
            ArbitrageConfidence with full analysis
        """
        # Get probability estimates
        yes_est = self.model.estimate_from_price(yes_price, yes_volume)
        no_est = self.model.estimate_from_price(no_price, no_volume)

        # Naive spread
        naive_spread = 1.0 - (yes_price + no_price)

        # Probability arbitrage truly exists
        arb_prob = self.calculate_arbitrage_probability(yes_est, no_est)

        # Profit distribution
        profit_dist = self.calculate_profit_distribution(yes_est, no_est, position_size)

        # Adjust for fees
        total_fees = position_size * (yes_price + no_price) * transaction_fee * 2
        adjusted_expected = profit_dist['mean'] - total_fees

        # Determine if significant
        is_significant = (
            arb_prob >= self.min_confidence and
            adjusted_expected > 0 and
            profit_dist['prob_positive'] > 0.9
        )

        # Generate reason
        if is_significant:
            reason = f"High confidence arbitrage: {arb_prob:.0%} probability, ${adjusted_expected:.2f} expected profit"
        elif arb_prob < self.min_confidence:
            reason = f"Low confidence: only {arb_prob:.0%} probability of true arbitrage"
        elif adjusted_expected <= 0:
            reason = f"Negative expected profit after fees: ${adjusted_expected:.2f}"
        else:
            reason = f"Risk too high: {profit_dist['prob_negative']:.0%} chance of loss"

        return ArbitrageConfidence(
            is_significant=is_significant,
            probability=arb_prob,
            expected_profit=adjusted_expected,
            var_95=profit_dist['var_95'],
            confidence_level=yes_est.confidence * no_est.confidence,
            reason=reason
        )

    def compare_opportunities(self, opportunities: List[Dict]) -> List[Dict]:
        """
        Rank multiple arbitrage opportunities by risk-adjusted return.

        Args:
            opportunities: List of {yes_price, yes_volume, no_price, no_volume, ...}

        Returns:
            Sorted list with analysis added
        """
        analyzed = []

        for opp in opportunities:
            analysis = self.analyze_opportunity(
                yes_price=opp['yes_price'],
                yes_volume=opp.get('yes_volume', 1000),
                no_price=opp['no_price'],
                no_volume=opp.get('no_volume', 1000),
                position_size=opp.get('position_size', 100)
            )

            analyzed.append({
                **opp,
                'arb_probability': analysis.probability,
                'expected_profit': analysis.expected_profit,
                'var_95': analysis.var_95,
                'confidence': analysis.confidence_level,
                'is_significant': analysis.is_significant,
                'reason': analysis.reason,
                # Risk-adjusted score: expected profit / VaR
                'risk_adjusted_score': analysis.expected_profit / max(analysis.var_95, 0.01)
            })

        # Sort by risk-adjusted score
        analyzed.sort(key=lambda x: x['risk_adjusted_score'], reverse=True)

        return analyzed


def calculate_implied_probability_ci(price: float, volume: float,
                                     confidence: float = 0.95) -> Tuple[float, float]:
    """
    Quick helper to get confidence interval for implied probability.

    Args:
        price: Market price
        volume: Trading volume
        confidence: Confidence level (default 95%)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    model = BetaProbabilityModel()
    estimate = model.estimate_from_price(price, volume)

    alpha_level = (1 - confidence) / 2
    lower = stats.beta.ppf(alpha_level, estimate.alpha, estimate.beta_param)
    upper = stats.beta.ppf(1 - alpha_level, estimate.alpha, estimate.beta_param)

    return (lower, upper)


def test_probability_models():
    """Test the probability estimation framework"""
    print("=== Probability Estimation Framework Test ===\n")

    model = BetaProbabilityModel()

    # Test 1: Low vs high volume
    print("Test 1: Same price, different volumes")
    print("-" * 50)

    low_vol = model.estimate_from_price(0.50, 100)
    high_vol = model.estimate_from_price(0.50, 100000)

    print(f"$100 volume:    mean={low_vol.mean:.3f}, 95% CI=[{low_vol.lower_95:.3f}, {low_vol.upper_95:.3f}]")
    print(f"$100k volume:   mean={high_vol.mean:.3f}, 95% CI=[{high_vol.lower_95:.3f}, {high_vol.upper_95:.3f}]")

    # Test 2: Arbitrage analysis
    print("\nTest 2: Arbitrage probability analysis")
    print("-" * 50)

    analyzer = ArbitrageProbabilityAnalyzer()

    # Good arbitrage (high volume, clear spread)
    result = analyzer.analyze_opportunity(
        yes_price=0.48, yes_volume=50000,
        no_price=0.48, no_volume=50000,
        position_size=100
    )
    print(f"Clear arb (0.48 + 0.48 = 0.96):")
    print(f"  Probability: {result.probability:.1%}")
    print(f"  Expected profit: ${result.expected_profit:.2f}")
    print(f"  Significant: {result.is_significant}")

    # Marginal arbitrage (low volume)
    result = analyzer.analyze_opportunity(
        yes_price=0.49, yes_volume=500,
        no_price=0.49, no_volume=500,
        position_size=100
    )
    print(f"\nMarginal arb (0.49 + 0.49 = 0.98, low volume):")
    print(f"  Probability: {result.probability:.1%}")
    print(f"  Expected profit: ${result.expected_profit:.2f}")
    print(f"  Significant: {result.is_significant}")
    print(f"  Reason: {result.reason}")

    # No arbitrage
    result = analyzer.analyze_opportunity(
        yes_price=0.52, yes_volume=10000,
        no_price=0.52, no_volume=10000,
        position_size=100
    )
    print(f"\nNo arb (0.52 + 0.52 = 1.04):")
    print(f"  Probability: {result.probability:.1%}")
    print(f"  Significant: {result.is_significant}")

    # Test 3: Bayesian update
    print("\nTest 3: Bayesian updating")
    print("-" * 50)

    prior = model.estimate_from_price(0.50, 1000)
    print(f"Prior:   mean={prior.mean:.3f}, CI=[{prior.lower_95:.3f}, {prior.upper_95:.3f}]")

    updated = model.bayesian_update(prior, 0.55, 5000)
    print(f"Updated: mean={updated.mean:.3f}, CI=[{updated.lower_95:.3f}, {updated.upper_95:.3f}]")


if __name__ == "__main__":
    test_probability_models()
