class Mem2Error(Exception):
    """Base exception for mem2."""


class ConfigurationError(Mem2Error):
    """Raised for invalid config/registry selections."""


class DataValidationError(Mem2Error):
    """Raised for invalid benchmark data."""
