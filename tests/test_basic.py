"""Basic tests for Edison package."""

import sys


def test_package_imports():
    """Test that the edison package can be imported."""
    try:
        import edison  # noqa: F401 - imported for test
    except ImportError:
        raise AssertionError("Failed to import edison package")


def test_basic_functionality():
    """Test basic functionality."""
    assert 1 + 1 == 2


def test_python_version():
    """Test that we're running on a supported Python version."""
    version = sys.version_info
    assert version.major == 3
    assert (
        version.minor >= 8
    ), f"Python 3.8+ required, got {version.major}.{version.minor}"
