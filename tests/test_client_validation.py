from pydantic import create_model
import pytest
from fastAPI.backend_fastapi import get_available_model_name_version, load_models, ScoringModel
from fastAPI.backend_fastapi import validate_client
from unittest.mock import patch
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import pickle



class AlwaysZeroClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier that always predicts class 0.
    """
    def fit(self, X, y=None):
        # No fitting process needed as it always predicts 0
        return self
    def predict(self, X):
        # Always predict 0 for any input
        return np.zeros(X.shape[0], dtype=int)
    def predict_proba(self, X):
        # Return probabilities: 100% confidence for class 0
        return np.column_stack([np.ones(X.shape[0]), np.zeros(X.shape[0])])
    
class AlwaysOneClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier that always predicts class 0.
    """
    def fit(self, X, y=None):
        # No fitting process needed as it always predicts 1
        return self
    def predict(self, X):
        # Always predict 1 for any input
        return np.ones(X.shape[0], dtype=int)
    def predict_proba(self, X):
        # Return probabilities: 100% confidence for class 1
        return np.column_stack([np.zeros(X.shape[0]), np.ones(X.shape[0])])
    

class RandomClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier that always predicts class 0.
    """
    def fit(self, X, y=None):
        # No fitting process needed as it always predicts random values
        return self
    def predict(self, X):
        # Always predict 0 for any input
        self.predict_proba()
        return np.zeros(X.shape[0], dtype=int)
    def predict_proba(self, X):
        # Return probabilities: 100% confidence for class 0
        x = np.random.random(X.shape[0])
        return np.column_stack([x, 1-x])


@pytest.fixture
def mock_models():
    always_zero = ScoringModel(
        model=AlwaysZeroClassifier(),
        validation_threshold=1.0,
        explainer_path='tmp/always_zero/explainer.pkl',
        shap_values_sample_path='tmp/always_zero/shap_values.pkl',
    )    
    always_one = ScoringModel(
        model=AlwaysOneClassifier(),
        validation_threshold=0.0,
        explainer_path='tmp/always_one/explainer.pkl',
        shap_values_sample_path='tmp/always_one/shap_values.pkl',
    )
    random_classifier = ScoringModel(
        model=RandomClassifier(),
        validation_threshold=0.5,
        explainer_path='tmp/random_classifier/explainer.pkl',
        shap_values_sample_path='tmp/random_classifier/shap_values.pkl',
    )
    models = {'always_zero': always_zero,
              'always_one': always_one,
              'random_classifier': random_classifier,
    } 
    return models


def test_get_available_models(mock_models):    
    with patch('fastAPI.backend_fastapi.models', mock_models):
        result = get_available_model_name_version()
        assert len(result) == 3 
        assert 'always_zero' in result
        assert 'always_one' in result
        assert 'random_classifier' in result


field_types = {str(i): (float,...) for i in range(10)}
MockModelEntries = create_model('MockModelEntries', **field_types)


@pytest.fixture
def mock_model_entries():
    # Return an instance of MockModelEntries with random values for testing
    return MockModelEntries(**{str(i): np.random.random() for i in range(10)})

@pytest.mark.asyncio
async def test_validate_client(mock_models, mock_model_entries):
    with patch('fastAPI.backend_fastapi.models', mock_models):
        with patch('fastAPI.backend_fastapi.ModelEntries', MockModelEntries):
            result = await validate_client(input_data=mock_model_entries, model_name_v='always_zero')            
            assert result['default_probability'][0] == 0
            assert result['validation_threshold'] == 1.0
            assert result['credit_approved'][0]
            
            result = await validate_client(input_data=mock_model_entries, model_name_v='always_one')            
            assert result['default_probability'][0] == 1.0
            assert result['validation_threshold'] == 0.0
            assert not result['credit_approved'][0]
            
            for i in range(1000):
                result = await validate_client(input_data=mock_model_entries, model_name_v='random_classifier')                        
                assert result['credit_approved'][0] == ( result['default_probability'][0] < result['validation_threshold']), f"p={float(result['default_probability'][0]):0.3f} < vt={float(result['validation_threshold']):0.4f}"