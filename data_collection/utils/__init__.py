"""
Utils Module

This module contains common utilities and helper functions for web scraping operations.

Utilities include:
    - HTTP request helpers with retry logic
    - Rate limiting decorators
    - User agent rotation
    - Proxy management
    - HTML parsing utilities
    - Data validation and cleaning
    - Logging configuration
    - Error handling utilities

Usage:
    from data_collection.utils import rate_limiter, get_random_user_agent
    
    @rate_limiter(requests_per_minute=30)
    def scrape_page(url):
        headers = {'User-Agent': get_random_user_agent()}
        # scraping logic
"""

__all__ = []  # Will be populated as utility functions are implemented
