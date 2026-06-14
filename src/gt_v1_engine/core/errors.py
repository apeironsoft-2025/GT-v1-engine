class GTV1EngineError(Exception):
    """Base exception for expected GT-v1-engine errors."""


class ConfigError(GTV1EngineError):
    """Raised when configuration files cannot be loaded."""


class DataValidationError(GTV1EngineError):
    """Raised when input data fails validation."""


class FileMissingError(GTV1EngineError):
    """Raised when a required file does not exist."""


class UnsupportedFormatError(GTV1EngineError):
    """Raised when a file format is unsupported."""


class RuleConfigError(GTV1EngineError):
    """Raised when Rule171 configuration is invalid."""


class CliArgumentError(GTV1EngineError):
    """Raised when CLI arguments are invalid."""
