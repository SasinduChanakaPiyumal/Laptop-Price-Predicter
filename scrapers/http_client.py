"""
HTTP client with retry logic and error handling for web scraping.

This module provides a robust HTTP client that handles connection pooling,
automatic retries with exponential backoff, timeouts, and common HTTP errors.
"""

import logging
import time
from typing import Dict, Optional, Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .exceptions import NetworkError, RateLimitError


logger = logging.getLogger(__name__)


class HTTPClient:
    """
    Manage HTTP requests with session pooling, retries, and error handling.
    
    This class provides a high-level interface for making HTTP requests with
    automatic retry logic using exponential backoff, configurable timeouts,
    and handling of common HTTP errors.
    
    Attributes:
        session: Requests session for connection pooling
        connect_timeout: Maximum time to wait for connection (seconds)
        read_timeout: Maximum time to wait for response (seconds)
        max_retries: Maximum number of retry attempts
        backoff_factor: Base factor for exponential backoff (1s, 2s, 4s, 8s...)
    """
    
    def __init__(
        self,
        connect_timeout: float = 10.0,
        read_timeout: float = 30.0,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        pool_connections: int = 10,
        pool_maxsize: int = 10
    ):
        """
        Initialize the HTTP client with configurable settings.
        
        Args:
            connect_timeout: Maximum seconds to wait for connection establishment
            read_timeout: Maximum seconds to wait for server response
            max_retries: Maximum number of retry attempts on failure
            backoff_factor: Multiplier for exponential backoff (1.0 = 1s, 2s, 4s, 8s)
            pool_connections: Number of connection pools to cache
            pool_maxsize: Maximum connections to save in the pool
        """
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        
        # Create session with connection pooling
        self.session = requests.Session()
        
        # Configure retry strategy with exponential backoff
        # Retry on connection errors, timeouts, and specific HTTP status codes
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[500, 502, 503, 504],  # Retry on server errors
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],
            raise_on_status=False  # We'll handle status codes manually
        )
        
        # Mount adapter with retry strategy for both HTTP and HTTPS
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=pool_connections,
            pool_maxsize=pool_maxsize
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        logger.debug(
            f"HTTPClient initialized: timeout=({connect_timeout}, {read_timeout}), "
            f"max_retries={max_retries}, backoff_factor={backoff_factor}"
        )
    
    def get(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> requests.Response:
        """
        Perform GET request with retry logic and error handling.
        
        Args:
            url: Target URL to fetch
            headers: Optional HTTP headers to include
            params: Optional query parameters
            **kwargs: Additional arguments passed to requests.get()
        
        Returns:
            Response object from successful request
        
        Raises:
            NetworkError: On connection failures or timeouts
            RateLimitError: On 429 Too Many Requests response
        """
        return self._request("GET", url, headers=headers, params=params, **kwargs)
    
    def post(
        self,
        url: str,
        data: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> requests.Response:
        """
        Perform POST request with retry logic and error handling.
        
        Args:
            url: Target URL to post to
            data: Form data to send in request body
            json: JSON data to send in request body
            headers: Optional HTTP headers to include
            **kwargs: Additional arguments passed to requests.post()
        
        Returns:
            Response object from successful request
        
        Raises:
            NetworkError: On connection failures or timeouts
            RateLimitError: On 429 Too Many Requests response
        """
        return self._request("POST", url, data=data, json=json, headers=headers, **kwargs)
    
    def _request(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> requests.Response:
        """
        Internal method to perform HTTP request with error handling.
        
        This method handles the actual request execution, logging, timeout
        configuration, and error translation to custom exceptions.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: Target URL
            **kwargs: Arguments passed to session.request()
        
        Returns:
            Response object from successful request
        
        Raises:
            NetworkError: On connection failures or timeouts
            RateLimitError: On 429 Too Many Requests response
        """
        # Set timeout if not explicitly provided
        if 'timeout' not in kwargs:
            kwargs['timeout'] = (self.connect_timeout, self.read_timeout)
        
        logger.debug(f"{method} request to {url}")
        start_time = time.time()
        
        try:
            response = self.session.request(method, url, **kwargs)
            elapsed = time.time() - start_time
            
            logger.debug(
                f"{method} {url} -> {response.status_code} "
                f"({elapsed:.2f}s, {len(response.content)} bytes)"
            )
            
            # Handle specific HTTP errors
            self._handle_http_errors(response, url)
            
            return response
            
        except requests.exceptions.Timeout as e:
            elapsed = time.time() - start_time
            error_msg = f"Request timeout after {elapsed:.2f}s: {url}"
            logger.error(error_msg)
            raise NetworkError(error_msg, url=url) from e
        
        except requests.exceptions.ConnectionError as e:
            elapsed = time.time() - start_time
            error_msg = f"Connection failed after {elapsed:.2f}s: {url}"
            logger.error(error_msg)
            raise NetworkError(error_msg, url=url) from e
        
        except requests.exceptions.RequestException as e:
            elapsed = time.time() - start_time
            error_msg = f"Request failed after {elapsed:.2f}s: {url} - {str(e)}"
            logger.error(error_msg)
            raise NetworkError(error_msg, url=url) from e
    
    def _handle_http_errors(self, response: requests.Response, url: str) -> None:
        """
        Handle common HTTP error status codes.
        
        Args:
            response: Response object to check
            url: URL of the request (for error messages)
        
        Raises:
            RateLimitError: On 429 Too Many Requests
            NetworkError: On 404, 500, 503, or other client/server errors
        """
        status_code = response.status_code
        
        # Handle rate limiting
        if status_code == 429:
            retry_after = response.headers.get('Retry-After')
            retry_seconds = int(retry_after) if retry_after and retry_after.isdigit() else None
            
            error_msg = f"Rate limit exceeded for {url}"
            if retry_seconds:
                error_msg += f", retry after {retry_seconds}s"
            
            logger.warning(error_msg)
            raise RateLimitError(error_msg, retry_after=retry_seconds)
        
        # Handle not found
        if status_code == 404:
            error_msg = f"Resource not found: {url}"
            logger.warning(error_msg)
            raise NetworkError(error_msg, url=url, status_code=status_code)
        
        # Handle server errors (500, 503, etc.)
        if status_code >= 500:
            error_msg = f"Server error {status_code} for {url}"
            logger.error(error_msg)
            raise NetworkError(error_msg, url=url, status_code=status_code)
        
        # Handle other client errors (4xx except 404 and 429)
        if 400 <= status_code < 500:
            error_msg = f"Client error {status_code} for {url}"
            logger.warning(error_msg)
            # Don't raise for other 4xx errors, let caller handle
            # But log them for visibility
    
    def close(self) -> None:
        """
        Close the HTTP session and clean up resources.
        
        Should be called when the client is no longer needed to properly
        close all connections and release resources.
        """
        if self.session:
            self.session.close()
            logger.debug("HTTPClient session closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically close session."""
        self.close()
