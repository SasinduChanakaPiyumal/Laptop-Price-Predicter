"""
Web scraping infrastructure for laptop price data collection.

This package provides reusable components for building site-specific scrapers:
- BaseScraper: Abstract base class with common scraping functionality
- HTTPClient: Robust HTTP client with retry logic and error handling
- Exception hierarchy: Standardized exceptions for different error types

Example usage:
    from scrapers import BaseScraper
    from scrapers.exceptions import NetworkError, ParsingError
    
    class MyScraper(BaseScraper):
        def parse_laptop(self, url):
            # Implementation here
            pass
        
        def get_listings_page(self, page=1):
            # Implementation here
            pass
"""

from .base_scraper import BaseScraper
from .http_client import HTTPClient
from .exceptions import (
    ScraperException,
    NetworkError,
    ParsingError,
    RateLimitError,
    ConfigurationError
)

__version__ = "1.0.0"

__all__ = [
    "BaseScraper",
    "HTTPClient",
    "ScraperException",
    "NetworkError",
    "ParsingError",
    "RateLimitError",
    "ConfigurationError",
]
