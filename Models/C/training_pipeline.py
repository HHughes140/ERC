"""
ML Training Pipeline with Walk-Forward Cross-Validation

Proper time-series training for the ML trading engine.

Key Features:
1. Walk-forward cross-validation (no lookahead)
2. Hyperparameter optimization with grid search
3. Model persistence and versioning
4. Performance metrics tracking

CRITICAL: Standard k-fold CV doesn't work for time series!
Walk-forward validation ensures we only train on past data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training pipeline"""
    # Walk-forward parameters
    n_splits: int = 5
    train_window: int = 500   # Bars of training data
    test_window: int = 100    # Bars of test data
    min_train_size: int = 200 # Minimum training samples

    # Model parameters to search
    param_grid: Dict = None

    # Training settings
    retrain_threshold: int = 500  # Retrain after N new samples
    max_features: int = 11

    def __post_init__(self):
        if self.param_grid is None:
            self.param_grid = {
                'n_neighbors': [5, 8, 12, 16, 20],
                'max_bars_back': [200, 400, 600],
                'confidence_threshold': [0.3, 0.45, 0.6],
                'feature_count': [5, 8, 11]
            }


@dataclass
class FoldResult:
    """Results from a single CV fold"""
    fold_idx: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    params: Dict
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    profit_factor: float
    sharpe_ratio: float
    n_trades: int


@dataclass
class OptimizationResult:
    """Results from hyperparameter optimization"""
    best_params: Dict
    best_score: float
    all_results: List[FoldResult]
    cv_scores: Dict[str, float]  # Mean scores across folds
    timestamp: str


class WalkForwardCV:
    """
    Walk-Forward Cross-Validation for time series.

    Unlike k-fold, walk-forward ensures:
    1. Training data always comes BEFORE test data
    2. No information leakage from future
    3. Realistic simulation of live trading

    Structure:
    |---train1---|test1|
         |---train2---|test2|
              |---train3---|test3|
    """

    def __init__(self, n_splits: int = 5, train_window: int = 500,
                 test_window: int = 100, gap: int = 0):
        """
        Args:
            n_splits: Number of train/test splits
            train_window: Size of training window
            test_window: Size of test window
            gap: Gap between train and test (to prevent lookahead)
        """
        self.n_splits = n_splits
        self.train_window = train_window
        self.test_window = test_window
        self.gap = gap

    def split(self, X: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for walk-forward CV.

        Args:
            X: Feature array (n_samples, n_features)

        Yields:
            Tuples of (train_indices, test_indices)
        """
        n_samples = len(X)
        min_required = self.train_window + self.test_window + self.gap

        if n_samples < min_required:
            raise ValueError(f"Not enough samples: {n_samples} < {min_required}")

        # Calculate step size to cover the data
        total_test = self.n_splits * self.test_window
        available = n_samples - self.train_window - self.gap
        step = max(self.test_window, available // self.n_splits)

        splits = []
        for i in range(self.n_splits):
            test_end = n_samples - i * step
            test_start = test_end - self.test_window
            train_end = test_start - self.gap
            train_start = max(0, train_end - self.train_window)

            if train_start >= train_end or test_start >= test_end:
                continue

            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)

            splits.append((train_idx, test_idx))

        # Reverse to chronological order
        return splits[::-1]


class LorentzianKNNTrainer:
    """
    Trainer for Lorentzian KNN classifier with walk-forward CV.

    Optimizes parameters for the ML trading engine.
    """

    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.cv = WalkForwardCV(
            n_splits=self.config.n_splits,
            train_window=self.config.train_window,
            test_window=self.config.test_window
        )

    def _lorentzian_distance(self, x1: np.ndarray, x2: np.ndarray,
                             weights: Optional[np.ndarray] = None) -> float:
        """Calculate Lorentzian distance between vectors"""
        diff = np.abs(x1 - x2)
        log_diff = np.log(1 + diff)

        if weights is not None:
            log_diff = log_diff * weights

        return np.sum(log_diff)

    def _knn_predict(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_test: np.ndarray, k: int,
                     weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        KNN prediction using Lorentzian distance.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            k: Number of neighbors
            weights: Feature weights

        Returns:
            Predicted labels for X_test
        """
        predictions = []

        for test_point in X_test:
            # Calculate distances to all training points
            distances = []
            for i, train_point in enumerate(X_train):
                dist = self._lorentzian_distance(test_point, train_point, weights)
                distances.append((dist, y_train[i]))

            # Sort by distance and get k nearest
            distances.sort(key=lambda x: x[0])
            k_nearest = distances[:k]

            # Vote (weighted by inverse distance)
            vote_sum = 0
            weight_sum = 0
            for dist, label in k_nearest:
                w = 1.0 / (dist + 1e-6)
                vote_sum += w * label
                weight_sum += w

            # Predict based on weighted vote
            pred = 1 if vote_sum / weight_sum > 0 else -1
            predictions.append(pred)

        return np.array(predictions)

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                           returns: Optional[np.ndarray] = None) -> Dict:
        """Calculate classification and trading metrics"""
        # Filter out zeros (unknown labels)
        mask = y_true != 0
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) == 0:
            return {
                'accuracy': 0, 'precision': 0, 'recall': 0,
                'f1_score': 0, 'profit_factor': 0, 'sharpe_ratio': 0, 'n_trades': 0
            }

        # Classification metrics
        correct = np.sum(y_true == y_pred)
        accuracy = correct / len(y_true)

        # Precision/recall for positive class
        true_pos = np.sum((y_true == 1) & (y_pred == 1))
        pred_pos = np.sum(y_pred == 1)
        actual_pos = np.sum(y_true == 1)

        precision = true_pos / pred_pos if pred_pos > 0 else 0
        recall = true_pos / actual_pos if actual_pos > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Trading metrics (if returns provided)
        profit_factor = 0
        sharpe = 0
        if returns is not None and len(returns) == len(mask):
            returns = returns[mask]
            trade_returns = returns * y_pred

            gains = trade_returns[trade_returns > 0].sum()
            losses = abs(trade_returns[trade_returns < 0].sum())
            profit_factor = gains / losses if losses > 0 else gains

            if len(trade_returns) > 1 and np.std(trade_returns) > 0:
                sharpe = np.mean(trade_returns) / np.std(trade_returns) * np.sqrt(252)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'n_trades': len(y_true)
        }

    def cross_validate(self, X: np.ndarray, y: np.ndarray,
                       params: Dict, returns: Optional[np.ndarray] = None) -> List[FoldResult]:
        """
        Run walk-forward cross-validation with given parameters.

        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,)
            params: Model parameters
            returns: Optional returns for trading metrics

        Returns:
            List of FoldResult for each fold
        """
        splits = self.cv.split(X)
        results = []

        k = params.get('n_neighbors', 8)
        weights = None  # Could add feature weights here

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            # Filter training data to only include known labels
            train_mask = y_train != 0
            X_train = X_train[train_mask]
            y_train = y_train[train_mask]

            if len(X_train) < 10:
                continue

            # Make predictions
            y_pred = self._knn_predict(X_train, y_train, X_test, k, weights)

            # Calculate metrics
            test_returns = returns[test_idx] if returns is not None else None
            metrics = self._calculate_metrics(y_test, y_pred, test_returns)

            results.append(FoldResult(
                fold_idx=fold_idx,
                train_start=train_idx[0],
                train_end=train_idx[-1],
                test_start=test_idx[0],
                test_end=test_idx[-1],
                params=params,
                **metrics
            ))

        return results

    def grid_search(self, X: np.ndarray, y: np.ndarray,
                    returns: Optional[np.ndarray] = None,
                    scoring: str = 'sharpe_ratio') -> OptimizationResult:
        """
        Grid search over parameter combinations.

        Args:
            X: Features
            y: Labels
            returns: Optional returns
            scoring: Metric to optimize ('accuracy', 'f1_score', 'sharpe_ratio', 'profit_factor')

        Returns:
            OptimizationResult with best parameters
        """
        param_grid = self.config.param_grid
        all_results = []
        best_score = float('-inf')
        best_params = None

        # Generate all parameter combinations
        from itertools import product

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        total_combinations = 1
        for v in param_values:
            total_combinations *= len(v)

        logger.info(f"Grid search: {total_combinations} parameter combinations")

        for i, values in enumerate(product(*param_values)):
            params = dict(zip(param_names, values))

            # Subset features if specified
            feature_count = params.get('feature_count', X.shape[1])
            X_subset = X[:, :feature_count]

            # Run CV
            fold_results = self.cross_validate(X_subset, y, params, returns)

            if not fold_results:
                continue

            # Calculate mean score
            scores = [getattr(r, scoring) for r in fold_results]
            mean_score = np.mean(scores)

            all_results.extend(fold_results)

            if mean_score > best_score:
                best_score = mean_score
                best_params = params

            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i+1}/{total_combinations}, best {scoring}: {best_score:.4f}")

        # Calculate CV statistics for best params
        best_results = [r for r in all_results if r.params == best_params]
        cv_scores = {
            'accuracy': np.mean([r.accuracy for r in best_results]),
            'f1_score': np.mean([r.f1_score for r in best_results]),
            'sharpe_ratio': np.mean([r.sharpe_ratio for r in best_results]),
            'profit_factor': np.mean([r.profit_factor for r in best_results]),
        }

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            cv_scores=cv_scores,
            timestamp=datetime.now().isoformat()
        )


class ModelPersistence:
    """
    Save and load optimized model parameters.
    """

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def save_params(self, result: OptimizationResult, name: str = "optimized_params") -> str:
        """Save optimization result to JSON"""
        filepath = self.models_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        data = {
            'best_params': result.best_params,
            'best_score': result.best_score,
            'cv_scores': result.cv_scores,
            'timestamp': result.timestamp,
            'n_folds': len(set(r.fold_idx for r in result.all_results))
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        # Also save as "latest"
        latest_path = self.models_dir / f"{name}_latest.json"
        with open(latest_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved parameters to {filepath}")
        return str(filepath)

    def load_params(self, name: str = "optimized_params_latest") -> Optional[Dict]:
        """Load saved parameters"""
        filepath = self.models_dir / f"{name}.json"

        if not filepath.exists():
            logger.warning(f"No saved parameters found at {filepath}")
            return None

        with open(filepath, 'r') as f:
            data = json.load(f)

        return data

    def list_saved_models(self) -> List[str]:
        """List all saved model files"""
        return [f.name for f in self.models_dir.glob("*.json")]


def run_training_pipeline(ohlcv_data: pd.DataFrame,
                          config: Optional[TrainingConfig] = None) -> OptimizationResult:
    """
    Complete training pipeline.

    Args:
        ohlcv_data: DataFrame with OHLCV data
        config: Training configuration

    Returns:
        OptimizationResult with best parameters
    """
    config = config or TrainingConfig()

    # Import indicators (from our new module)
    try:
        from indicators import TechnicalIndicators
        indicators = TechnicalIndicators(normalize=True)
        features = indicators.generate_all_features(ohlcv_data)
    except ImportError:
        logger.warning("indicators module not found, using basic features")
        # Basic RSI-only features
        close = ohlcv_data['close']
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).ewm(span=14).mean()
        loss = (-delta).where(delta < 0, 0.0).ewm(span=14).mean()
        rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
        features = pd.DataFrame({'rsi': rsi / 100})

    # Create labels (future returns)
    close = ohlcv_data['close']
    forward_returns = close.pct_change(4).shift(-4)
    y = np.where(forward_returns > 0, 1, -1)
    y[-4:] = 0  # Unknown for last 4 bars
    y = np.where(np.isnan(forward_returns.values), 0, y)

    # Prepare arrays
    X = features.fillna(0).values
    returns = close.pct_change().values

    # Run grid search
    trainer = LorentzianKNNTrainer(config)
    result = trainer.grid_search(X, y, returns, scoring='sharpe_ratio')

    # Save results
    persistence = ModelPersistence()
    persistence.save_params(result)

    return result


def test_training_pipeline():
    """Test the training pipeline"""
    print("=== ML Training Pipeline Test ===\n")

    # Generate synthetic data
    np.random.seed(42)
    n = 1000

    # Random walk with mean reversion
    returns = np.random.randn(n) * 0.02
    close = 100 * np.exp(np.cumsum(returns))

    ohlcv = pd.DataFrame({
        'open': close * 0.999,
        'high': close * (1 + np.abs(np.random.randn(n) * 0.01)),
        'low': close * (1 - np.abs(np.random.randn(n) * 0.01)),
        'close': close,
        'volume': np.random.randint(1000, 10000, n).astype(float)
    })

    # Test walk-forward CV
    print("Testing Walk-Forward CV...")
    cv = WalkForwardCV(n_splits=3, train_window=200, test_window=50)
    splits = cv.split(np.zeros((n, 5)))

    for i, (train_idx, test_idx) in enumerate(splits):
        print(f"  Fold {i+1}: train[{train_idx[0]}:{train_idx[-1]}], test[{test_idx[0]}:{test_idx[-1]}]")

    # Test grid search (reduced grid for speed)
    print("\nTesting Grid Search (reduced grid)...")
    config = TrainingConfig(
        n_splits=3,
        train_window=300,
        test_window=50,
        param_grid={
            'n_neighbors': [5, 10],
            'feature_count': [3, 5]
        }
    )

    # Simple features for test
    close_series = ohlcv['close']
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0.0).ewm(span=14).mean()
    loss = (-delta).where(delta < 0, 0.0).ewm(span=14).mean()
    rsi = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).fillna(50) / 100

    X = np.column_stack([
        rsi.values,
        close_series.pct_change(5).fillna(0).values,
        close_series.pct_change(10).fillna(0).values,
        close_series.pct_change(20).fillna(0).values,
        (close_series / close_series.rolling(20).mean() - 1).fillna(0).values
    ])

    forward_returns = close_series.pct_change(4).shift(-4)
    y = np.where(forward_returns > 0, 1, -1)
    y[-4:] = 0
    y = np.where(np.isnan(forward_returns.values), 0, y)

    returns = close_series.pct_change().values

    trainer = LorentzianKNNTrainer(config)
    result = trainer.grid_search(X, y, returns, scoring='accuracy')

    print(f"\nBest parameters: {result.best_params}")
    print(f"Best score: {result.best_score:.4f}")
    print(f"CV scores: {result.cv_scores}")


if __name__ == "__main__":
    test_training_pipeline()
