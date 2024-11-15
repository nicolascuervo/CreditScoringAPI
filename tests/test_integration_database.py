
import pytest
import sqlite3
from fastAPI.backend_fastapi import get_credit_application
from unittest.mock import patch


@pytest.fixture
def setup_test_db(tmp_path):
    # Create database that's stocked in memory
    db_path = tmp_path / "test_credit_requests.db"

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()    
    cursor.execute(
    """
    CREATE TABLE application_test (
        SK_ID_CURR INTEGER PRIMARY KEY,
        Feature1 INTEGER,
        Feature2 TEXT
        );        
    """
    )
    cursor.execute("""
    CREATE TABLE application_train (
        SK_ID_CURR INTEGER PRIMARY KEY,
        Feature1 INTEGER,
        Feature2 TEXT
        );
    """
    )
    cursor.execute("""
    INSERT INTO application_test 
        (SK_ID_CURR, Feature1, Feature2) 
        VALUES 
        (1001, 101, 'value1');
    """
    )
    cursor.execute("""
    INSERT INTO application_train 
        (SK_ID_CURR) 
        VALUES
        (1002);
    """
    )
    conn.commit()
    yield db_path

    conn.close()

def test_get_credit_application_integration(setup_test_db):
    """
        mocker : object similar to unitest.mock, provided by pytest library `pytest-mock` must be installed.
    """
    with patch('fastAPI.backend_fastapi.credit_requests_db', str(setup_test_db)):
        result = get_credit_application(1001)        
        assert isinstance(result, dict), 'THe result should be a dictionary'
        assert result['Feature1'] == 101, 'Feature1 must be 101'
        assert result['Feature2'] == 'value1', 'Feature2 must be "value1"'

        result = get_credit_application(1002)        
        assert result['Feature1'] is None, "expected None for Feature1"
     


    



