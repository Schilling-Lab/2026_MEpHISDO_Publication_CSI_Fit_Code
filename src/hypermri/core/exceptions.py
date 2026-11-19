"""
Custom exception classes for the hypermri package.

This module defines a hierarchy of exceptions that provide clear error
handling throughout the hypermri ecosystem.
"""


class HyperMRIError(Exception):
    """Base exception for all hypermri-related errors."""


class ReconstructionError(HyperMRIError):
    """Raised when reconstruction fails."""

    def __init__(self, message: str, strategy: str = None, sequence: str = None):
        """Capture strategy metadata when reconstruction cannot proceed.

        Parameters
        ----------
        message
            Description of the reconstruction failure.
        strategy
            Fully qualified strategy identifier, if known.
        sequence
            Sequence name associated with the failing reconstruction.
        """
        super().__init__(message)
        self.strategy = strategy
        self.sequence = sequence


class ConfigurationError(HyperMRIError):
    """Raised when configuration is invalid or missing."""

    def __init__(self, message: str, config_type: str = None, config_path: str = None):
        """Capture configuration metadata for debugging.

        Parameters
        ----------
        message
            Error message detailing the configuration issue.
        config_type
            Logical grouping of the configuration file or section.
        config_path
            Filesystem path to the configuration resource.
        """
        super().__init__(message)
        self.config_type = config_type
        self.config_path = config_path


