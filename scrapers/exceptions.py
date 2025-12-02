"""
Exception hierarchy for web scraping operations.

This module defines custom exceptions for handling various scraping-related errors,
including network failures, parsing issues, rate limiting, and configuration problems.
"""


class ScraperException(Exception):
    """
    Base exception class for all scraper-related errors.
    
    All custom scraper exceptions inherit from this base class,
    allowing for unified exception handling when needed.
    """
    
    def __init__(self, message: str, *args, **kwargs):
        """
        Initialize the ScraperException.
        
        Args:
            message: Descriptive error message
            *args: Additional positional arguments passed to Exception
            **kwargs: Additional keyword arguments for context
        """
        self.message = message
        self.context = kwargs
        super().__init__(message, *args)
    
    def __str__(self) -> str:
        """Return string representation of the exception."""
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} ({context_str})"
        return self.message


class NetworkError(ScraperException):
    """
    Exception raised for network-related failures.
    
    This includes connection failures, timeouts, DNS resolution errors,
    and other network-level problems that prevent successful HTTP communication.
    
    Examples:
        - Connection timeout
        - Connection refused
        - DNS resolution failure
        - SSL/TLS errors
    """
    
    def __init__(self, message: str, url: str = None, *args, **kwargs):
        """
        Initialize the NetworkError.
        
        Args:
            message: Descriptive error message
            url: The URL that caused the network error (optional)
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments for context
        """
        if url:
            kwargs['url'] = url
        super().__init__(message, *args, **kwargs)


class ParsingError(ScraperException):
    """
    Exception raised for HTML/data parsing failures.
    
    This occurs when the scraper cannot extract expected data from the HTML,
    such as missing elements, unexpected structure, or malformed data.
    
    Examples:
        - Expected HTML element not found
        - Unable to extract price from text
        - Invalid data format
        - Unexpected page structure
    """
    
    def __init__(self, message: str, element: str = None, *args, **kwargs):
        """
        Initialize the ParsingError.
        
        Args:
            message: Descriptive error message
            element: The HTML element or selector that failed (optional)
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments for context
        """
        if element:
            kwargs['element'] = element
        super().__init__(message, *args, **kwargs)


class RateLimitError(ScraperException):
    """
    Exception raised when rate limits are exceeded.
    
    This occurs when the target server returns a 429 (Too Many Requests) response,
    or when the scraper's internal rate limiting prevents a request.
    
    Examples:
        - HTTP 429 Too Many Requests
        - Exceeded configured request rate
        - Temporary ban detected
    """
    
    def __init__(self, message: str, retry_after: int = None, *args, **kwargs):
        """
        Initialize the RateLimitError.
        
        Args:
            message: Descriptive error message
            retry_after: Suggested wait time in seconds before retrying (optional)
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments for context
        """
        if retry_after is not None:
            kwargs['retry_after'] = retry_after
        super().__init__(message, *args, **kwargs)


class ConfigurationError(ScraperException):
    """
    Exception raised for invalid configuration settings.
    
    This occurs when scraper configuration is missing, invalid, or contains
    incompatible values that prevent the scraper from operating correctly.
    
    Examples:
        - Missing required configuration key
        - Invalid timeout value
        - Malformed URL pattern
        - Empty user agent list
    """
    
    def __init__(self, message: str, config_key: str = None, *args, **kwargs):
        """
        Initialize the ConfigurationError.
        
        Args:
            message: Descriptive error message
            config_key: The configuration key that caused the error (optional)
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments for context
        """
        if config_key:
            kwargs['config_key'] = config_key
        super().__init__(message, *args, **kwargs)
