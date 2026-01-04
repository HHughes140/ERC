"""
Tests for Central_DB/database.py
"""
import pytest
from datetime import datetime


class TestDatabase:
    """Tests for Database class"""

    def test_database_initialization(self, temp_db):
        """Test database creates all required tables"""
        conn = temp_db.connect()
        cursor = conn.cursor()

        # Check all tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}

        expected_tables = {
            'trades', 'positions', 'portfolio_snapshots',
            'strategy_performance', 'risk_metrics',
            'market_outcomes', 'model_states', 'api_cache',
            'sentiment_signals', 'ab_test_results'
        }

        assert expected_tables.issubset(tables), f"Missing tables: {expected_tables - tables}"

    def test_insert_trade(self, temp_db, sample_trade):
        """Test inserting a trade"""
        result = temp_db.insert_trade(sample_trade)
        assert result is True

        # Verify trade was inserted
        trades = temp_db.get_open_trades()
        assert len(trades) == 1
        assert trades[0]['trade_id'] == sample_trade['trade_id']

    def test_update_trade_exit(self, temp_db, sample_trade):
        """Test updating trade with exit info"""
        temp_db.insert_trade(sample_trade)

        result = temp_db.update_trade_exit(
            trade_id=sample_trade['trade_id'],
            exit_price=0.55,
            pnl=10.0
        )
        assert result is True

        # Verify trade is no longer open
        open_trades = temp_db.get_open_trades()
        assert len(open_trades) == 0

    def test_insert_position(self, temp_db, sample_position):
        """Test inserting a position"""
        result = temp_db.insert_position(sample_position)
        assert result is True

        positions = temp_db.get_open_positions()
        assert len(positions) == 1
        assert positions[0]['position_id'] == sample_position['position_id']

    def test_close_position(self, temp_db, sample_position):
        """Test closing a position"""
        temp_db.insert_position(sample_position)

        result = temp_db.close_position(sample_position['position_id'])
        assert result is True

        open_positions = temp_db.get_open_positions()
        assert len(open_positions) == 0

    def test_market_outcomes(self, temp_db):
        """Test market outcomes for calibration"""
        outcome_data = {
            'market_id': 'test_market_001',
            'platform': 'polymarket',
            'market_type': 'directional',
            'entry_price': 0.95,
            'resolution_price': 1.0,
            'won': True,
            'pnl': 0.05,
            'entry_timestamp': datetime.now().isoformat(),
        }

        result = temp_db.record_market_outcome(outcome_data)
        assert result is True

        # Get calibration data
        cal_data = temp_db.get_calibration_data('polymarket', 'directional', 0.90, 1.0)
        assert len(cal_data) == 1
        assert cal_data[0]['won'] == 1

    def test_model_states(self, temp_db):
        """Test model state persistence"""
        model_data = {
            'model_id': 'lorentzian_knn_v1',
            'model_type': 'lorentzian_knn',
            'parameters': {'k': 8, 'lookback': 400},
            'weights': [0.35, 0.30, 0.20, 0.15],
            'feature_names': ['rsi', 'adx', 'cci', 'wavetrend'],
            'training_samples': 1000,
            'performance_metrics': {'sharpe': 1.5, 'win_rate': 0.55}
        }

        result = temp_db.save_model_state(model_data)
        assert result is True

        # Retrieve model
        model = temp_db.get_model_state('lorentzian_knn_v1')
        assert model is not None
        assert model['model_type'] == 'lorentzian_knn'
        assert model['parameters']['k'] == 8

    def test_api_cache(self, temp_db):
        """Test API caching"""
        result = temp_db.cache_set('test_key', {'data': [1, 2, 3]}, 'test_ns', 60)
        assert result is True

        cached = temp_db.cache_get('test_key')
        assert cached == {'data': [1, 2, 3]}

    def test_sentiment_signals(self, temp_db):
        """Test sentiment signal storage"""
        signal_data = {
            'market_id': 'test_market_001',
            'source': 'twitter',
            'text': 'Very bullish!',
            'sentiment_score': 0.85,
            'confidence': 0.7,
            'engagement': 150,
            'keywords': ['bullish'],
        }

        result = temp_db.save_sentiment_signal(signal_data)
        assert result is True

    def test_ab_test_results(self, temp_db):
        """Test A/B test result storage"""
        test_data = {
            'test_id': 'kelly_vs_fixed_001',
            'variant_a': 'kelly_sizing',
            'variant_b': 'fixed_sizing',
            'metric_name': 'sharpe_ratio',
            'samples_a': 50,
            'samples_b': 50,
            'mean_a': 1.5,
            'mean_b': 1.2,
            'p_value': 0.03,
            'effect_size': 0.45,
            'winner': 'kelly_sizing',
            'is_significant': True,
        }

        result = temp_db.save_ab_test_result(test_data)
        assert result is True

        ab_result = temp_db.get_ab_test_result('kelly_vs_fixed_001')
        assert ab_result is not None
        assert ab_result['winner'] == 'kelly_sizing'
