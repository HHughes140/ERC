"""
Ensemble Classifier for ML Trading Engine

Combines multiple models for more robust predictions.

Key Features:
1. Multiple base classifiers (KNN, XGBoost, RF, Logistic)
2. Dynamic weight updates based on recent performance
3. Confidence-weighted voting
4. Calibrated probability outputs

CRITICAL: Single models overfit. Ensembles are more robust.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelPrediction:
    """Prediction from a single model"""
    direction: int        # 1 = long, -1 = short, 0 = neutral
    confidence: float     # 0-1
    probability: float    # Calibrated probability of direction being correct


@dataclass
class EnsemblePrediction:
    """Combined prediction from ensemble"""
    direction: int
    confidence: float
    probability: float
    model_votes: Dict[str, int]
    model_confidences: Dict[str, float]
    agreement_score: float  # How much models agree (0-1)


class LorentzianKNN:
    """
    Lorentzian KNN classifier (port of main engine).

    Uses log(1 + |x|) distance which is more robust to outliers.
    """

    def __init__(self, k: int = 8, max_bars: int = 400):
        self.k = k
        self.max_bars = max_bars
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model on training data"""
        # Only keep valid labels
        mask = y != 0
        self.X_train = X[mask][-self.max_bars:]
        self.y_train = y[mask][-self.max_bars:]

    def _lorentzian_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return np.sum(np.log(1 + np.abs(x1 - x2)))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of positive class"""
        if self.X_train is None or len(self.X_train) < self.k:
            return np.full(len(X), 0.5)

        probas = []
        for x in X:
            distances = [self._lorentzian_distance(x, x_train)
                        for x_train in self.X_train]
            nearest_idx = np.argsort(distances)[:self.k]

            # Distance-weighted vote
            weights = 1.0 / (np.array(distances)[nearest_idx] + 1e-6)
            labels = self.y_train[nearest_idx]

            pos_weight = np.sum(weights[labels == 1])
            total_weight = np.sum(weights)

            prob = pos_weight / total_weight if total_weight > 0 else 0.5
            probas.append(prob)

        return np.array(probas)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        probas = self.predict_proba(X)
        return np.where(probas > 0.5, 1, -1)


class SimpleLogisticRegression:
    """
    Simple logistic regression without sklearn dependency.

    Uses gradient descent for fitting.
    """

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000,
                 regularization: float = 0.1):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.reg = regularization
        self.weights = None
        self.bias = 0

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit using gradient descent"""
        # Convert labels to 0/1
        mask = y != 0
        X = X[mask]
        y = (y[mask] + 1) / 2  # Convert -1/1 to 0/1

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            z = np.dot(X, self.weights) + self.bias
            predictions = self._sigmoid(z)

            # Gradients
            dw = (1/n_samples) * np.dot(X.T, (predictions - y)) + self.reg * self.weights
            db = (1/n_samples) * np.sum(predictions - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of positive class"""
        if self.weights is None:
            return np.full(len(X), 0.5)
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        probas = self.predict_proba(X)
        return np.where(probas > 0.5, 1, -1)


class SimpleRandomForest:
    """
    Simple random forest using decision stumps.

    Lightweight alternative to sklearn.
    """

    def __init__(self, n_trees: int = 50, max_features: float = 0.7,
                 bootstrap_ratio: float = 0.8):
        self.n_trees = n_trees
        self.max_features = max_features
        self.bootstrap_ratio = bootstrap_ratio
        self.trees = []

    def _fit_stump(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Fit a single decision stump"""
        n_samples, n_features = X.shape

        # Select random features
        n_select = max(1, int(n_features * self.max_features))
        feature_idx = np.random.choice(n_features, n_select, replace=False)

        best_gain = -float('inf')
        best_stump = None

        for feat in feature_idx:
            # Try thresholds at various percentiles
            thresholds = np.percentile(X[:, feat], [25, 50, 75])

            for thresh in thresholds:
                left_mask = X[:, feat] <= thresh
                right_mask = ~left_mask

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                # Calculate information gain
                left_prob = y[left_mask].mean()
                right_prob = y[right_mask].mean()

                # Gini impurity reduction
                parent_gini = 2 * y.mean() * (1 - y.mean())
                left_gini = 2 * left_prob * (1 - left_prob)
                right_gini = 2 * right_prob * (1 - right_prob)

                gain = parent_gini - (left_mask.mean() * left_gini +
                                     right_mask.mean() * right_gini)

                if gain > best_gain:
                    best_gain = gain
                    best_stump = {
                        'feature': feat,
                        'threshold': thresh,
                        'left_prob': left_prob,
                        'right_prob': right_prob
                    }

        return best_stump

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the forest"""
        mask = y != 0
        X = X[mask]
        y = (y[mask] + 1) / 2  # Convert to 0/1

        n_samples = len(X)
        self.trees = []

        for _ in range(self.n_trees):
            # Bootstrap sample
            idx = np.random.choice(n_samples, int(n_samples * self.bootstrap_ratio))
            X_boot, y_boot = X[idx], y[idx]

            stump = self._fit_stump(X_boot, y_boot)
            if stump:
                self.trees.append(stump)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of positive class"""
        if not self.trees:
            return np.full(len(X), 0.5)

        predictions = np.zeros(len(X))

        for tree in self.trees:
            mask = X[:, tree['feature']] <= tree['threshold']
            predictions[mask] += tree['left_prob']
            predictions[~mask] += tree['right_prob']

        return predictions / len(self.trees)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        probas = self.predict_proba(X)
        return np.where(probas > 0.5, 1, -1)


class EnsembleClassifier:
    """
    Ensemble classifier combining multiple models.

    Models are weighted dynamically based on recent performance.
    """

    # Default model weights
    DEFAULT_WEIGHTS = {
        'lorentzian_knn': 0.35,
        'logistic': 0.15,
        'random_forest': 0.20,
        'momentum': 0.15,
        'mean_reversion': 0.15
    }

    def __init__(self, initial_weights: Optional[Dict[str, float]] = None,
                 weight_decay: float = 0.99,
                 confidence_threshold: float = 0.6,
                 min_agreement: float = 0.6):
        """
        Args:
            initial_weights: Starting weights for each model
            weight_decay: Decay factor for historical performance
            confidence_threshold: Minimum confidence to make prediction
            min_agreement: Minimum model agreement to predict
        """
        self.weights = initial_weights or self.DEFAULT_WEIGHTS.copy()
        self.weight_decay = weight_decay
        self.confidence_threshold = confidence_threshold
        self.min_agreement = min_agreement

        # Models
        self.models = {
            'lorentzian_knn': LorentzianKNN(k=8, max_bars=400),
            'logistic': SimpleLogisticRegression(),
            'random_forest': SimpleRandomForest(n_trees=30),
        }

        # Performance tracking for weight updates (include rule-based signals)
        all_signal_names = list(self.models.keys()) + ['momentum', 'mean_reversion']
        self.performance_history = {name: deque(maxlen=100) for name in all_signal_names}

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit all models"""
        for name, model in self.models.items():
            try:
                model.fit(X, y)
            except Exception as e:
                logger.warning(f"Error fitting {name}: {e}")

    def _momentum_signal(self, features: np.ndarray) -> Tuple[float, float]:
        """
        Simple momentum signal based on recent returns.

        Assumes features include normalized momentum indicators.
        """
        # Use first feature (typically RSI or similar)
        if len(features) == 0:
            return 0, 0.5

        rsi = features[0] if len(features.shape) == 1 else features[-1, 0]

        # Strong momentum signals
        if rsi > 0.7:
            return 1, min(0.5 + (rsi - 0.5) * 0.5, 0.8)
        elif rsi < 0.3:
            return -1, min(0.5 + (0.5 - rsi) * 0.5, 0.8)
        else:
            return 0, 0.5

    def _mean_reversion_signal(self, features: np.ndarray) -> Tuple[float, float]:
        """
        Mean reversion signal (opposite of momentum).

        Works in range-bound markets.
        """
        if len(features) == 0:
            return 0, 0.5

        rsi = features[0] if len(features.shape) == 1 else features[-1, 0]

        # Oversold = expect reversion up
        if rsi < 0.25:
            return 1, min(0.5 + (0.25 - rsi) * 2, 0.75)
        # Overbought = expect reversion down
        elif rsi > 0.75:
            return -1, min(0.5 + (rsi - 0.75) * 2, 0.75)
        else:
            return 0, 0.5

    def predict_single(self, features: np.ndarray) -> EnsemblePrediction:
        """
        Make ensemble prediction for single sample.

        Args:
            features: Feature vector (1D array)

        Returns:
            EnsemblePrediction with combined result
        """
        features = features.reshape(1, -1)
        model_votes = {}
        model_confidences = {}

        # Get predictions from ML models
        for name, model in self.models.items():
            try:
                prob = model.predict_proba(features)[0]
                direction = 1 if prob > 0.5 else -1
                confidence = abs(prob - 0.5) * 2  # Scale to 0-1

                model_votes[name] = direction
                model_confidences[name] = confidence
            except Exception as e:
                logger.debug(f"Prediction error for {name}: {e}")
                model_votes[name] = 0
                model_confidences[name] = 0

        # Add rule-based signals
        mom_dir, mom_conf = self._momentum_signal(features[0])
        model_votes['momentum'] = mom_dir
        model_confidences['momentum'] = mom_conf

        mr_dir, mr_conf = self._mean_reversion_signal(features[0])
        model_votes['mean_reversion'] = mr_dir
        model_confidences['mean_reversion'] = mr_conf

        # Weighted vote
        weighted_vote = 0
        weight_sum = 0

        for name, vote in model_votes.items():
            if vote != 0:
                weight = self.weights.get(name, 0.1) * model_confidences.get(name, 0.5)
                weighted_vote += vote * weight
                weight_sum += weight

        # Calculate final prediction
        if weight_sum > 0:
            avg_vote = weighted_vote / weight_sum
        else:
            avg_vote = 0

        # Direction based on weighted vote
        if abs(avg_vote) < 0.1:
            final_direction = 0
        else:
            final_direction = 1 if avg_vote > 0 else -1

        # Confidence from vote magnitude
        final_confidence = min(abs(avg_vote), 1.0)

        # Agreement score
        non_zero_votes = [v for v in model_votes.values() if v != 0]
        if non_zero_votes:
            agreement = abs(sum(non_zero_votes)) / len(non_zero_votes)
        else:
            agreement = 0

        # Apply thresholds
        if final_confidence < self.confidence_threshold or agreement < self.min_agreement:
            final_direction = 0

        return EnsemblePrediction(
            direction=final_direction,
            confidence=final_confidence,
            probability=0.5 + final_direction * final_confidence * 0.5,
            model_votes=model_votes,
            model_confidences=model_confidences,
            agreement_score=agreement
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict for multiple samples"""
        return np.array([self.predict_single(x).direction for x in X])

    def update_weights(self, predictions: Dict[str, int], actual: int):
        """
        Update model weights based on prediction accuracy.

        Models that predicted correctly get weight increased.
        Models that predicted wrong get weight decreased.
        """
        for name, pred in predictions.items():
            if pred == 0:
                continue  # Skip neutral predictions

            correct = (pred == actual)
            self.performance_history[name].append(1 if correct else 0)

            # Calculate recent accuracy
            if len(self.performance_history[name]) >= 10:
                recent_accuracy = np.mean(list(self.performance_history[name]))

                # Adjust weights
                base_weight = self.DEFAULT_WEIGHTS.get(name, 0.1)

                if recent_accuracy > 0.55:
                    # Increase weight for accurate models
                    self.weights[name] = base_weight * (1 + (recent_accuracy - 0.5))
                elif recent_accuracy < 0.45:
                    # Decrease weight for inaccurate models
                    self.weights[name] = base_weight * (0.5 + recent_accuracy)
                else:
                    # Decay toward base weight
                    self.weights[name] = self.weight_decay * self.weights[name] + \
                                         (1 - self.weight_decay) * base_weight

        # Normalize weights
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    def get_model_performance(self) -> Dict[str, Dict]:
        """Get performance statistics for each model"""
        stats = {}
        for name, history in self.performance_history.items():
            if len(history) > 0:
                stats[name] = {
                    'accuracy': np.mean(list(history)),
                    'n_predictions': len(history),
                    'current_weight': self.weights.get(name, 0)
                }
        return stats


def test_ensemble():
    """Test the ensemble classifier"""
    print("=== Ensemble Classifier Test ===\n")

    # Generate synthetic data
    np.random.seed(42)
    n = 500

    # Features with some predictive power
    X = np.random.randn(n, 5)
    # Label based on feature combination
    signal = 0.3 * X[:, 0] - 0.2 * X[:, 1] + 0.1 * X[:, 2]
    y = np.sign(signal + np.random.randn(n) * 0.5)
    y = y.astype(int)

    # Split train/test
    train_size = 400
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Create and fit ensemble
    ensemble = EnsembleClassifier(confidence_threshold=0.3, min_agreement=0.4)
    ensemble.fit(X_train, y_train)

    # Make predictions
    correct = 0
    predictions = []

    for i, (x, y_true) in enumerate(zip(X_test, y_test)):
        pred = ensemble.predict_single(x)
        predictions.append(pred)

        if pred.direction != 0:
            if pred.direction == y_true:
                correct += 1

            # Update weights
            ensemble.update_weights(pred.model_votes, y_true)

    # Calculate accuracy
    non_neutral = [p for p in predictions if p.direction != 0]
    if non_neutral:
        accuracy = correct / len(non_neutral)
        coverage = len(non_neutral) / len(predictions)

        print(f"Accuracy: {accuracy:.1%}")
        print(f"Coverage: {coverage:.1%} ({len(non_neutral)}/{len(predictions)} predictions)")
        print(f"\nModel Performance:")
        for name, stats in ensemble.get_model_performance().items():
            print(f"  {name}: acc={stats['accuracy']:.1%}, weight={stats['current_weight']:.1%}")

        print(f"\nSample prediction:")
        sample = predictions[0]
        print(f"  Direction: {sample.direction}, Confidence: {sample.confidence:.2f}")
        print(f"  Agreement: {sample.agreement_score:.2f}")
        print(f"  Votes: {sample.model_votes}")
    else:
        print("No non-neutral predictions made")


if __name__ == "__main__":
    test_ensemble()
