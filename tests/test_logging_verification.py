"""
Test file to verify pytest logging is working correctly.
"""
import pytest
import logging

# Get logger for this test module
logger = logging.getLogger(__name__)

def test_logging_levels():
    """Test that all logging levels work correctly."""
    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    logger.critical("This is a CRITICAL message")
    
    # Test should pass
    assert True

def test_logging_with_variables():
    """Test logging with variables and formatting."""
    test_var = "test_value"
    test_number = 42
    
    logger.info(f"Testing logging with variables: {test_var}, number: {test_number}")
    logger.info("Testing logging with %s formatting", test_var)
    
    assert test_var == "test_value"
    assert test_number == 42

def test_logging_in_exception_handling():
    """Test logging during exception handling."""
    try:
        # Simulate an error
        result = 1 / 0
    except ZeroDivisionError as e:
        logger.error("Caught exception: %s", str(e))
        logger.info("Exception handling completed successfully")
    
    # Test should pass
    assert True

def test_logging_with_context():
    """Test logging with additional context."""
    logger.info("Starting test with context")
    
    # Simulate some work
    for i in range(3):
        logger.debug(f"Processing step {i + 1}/3")
    
    logger.info("Test with context completed")
    
    assert True

@pytest.mark.slow
def test_logging_slow_test():
    """Test logging in a slow test."""
    logger.info("Starting slow test")
    
    # Simulate slow operation
    import time
    time.sleep(0.1)
    
    logger.info("Slow test completed")
    assert True

def test_logging_failure():
    """Test logging when a test fails."""
    logger.info("This test will fail and should show logging")
    logger.warning("About to cause a failure")
    
    # This will fail
    assert False, "Intentional failure to test logging"

def test_logging_success():
    """Test logging when a test passes."""
    logger.info("This test will pass and should show logging")
    logger.info("Test is proceeding normally")
    
    # This will pass
    assert True 