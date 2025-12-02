"""
Abstract base class for site-specific web scrapers.

This module provides the BaseScraper class that handles common scraping
functionality including HTTP requests, rate limiting, robots.txt checking,
user agent rotation, and logging configuration.
"""

import logging
import logging.handlers
import os
import time
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

from .http_client import HTTPClient
from .exceptions import NetworkError, ParsingError, ConfigurationError, RateLimitError


class BaseScraper(ABC):
    """
    Abstract base class for site-specific laptop scrapers.
    
    This class provides common functionality for web scraping including:
    - HTTP request handling with retry logic
    - Rate limiting between requests
    - Robots.txt compliance checking
    - User agent rotation
    - Standardized logging with rotating file handlers
    
    Subclasses must implement:
    - parse_laptop(): Extract laptop details from a product page
    - get_listings_page(): Fetch and parse a listings/search page
    
    Attributes:
        name: Scraper identifier name
        base_url: Base URL of the target website
        user_agents: List of user agent strings to rotate through
        rate_limit_delay: Minimum seconds to wait between requests
        logger: Logger instance for this scraper
        http_client: HTTPClient instance for making requests
        robots_parser: RobotFileParser for checking robots.txt rules
    """
    
    def __init__(
        self,
        name: str,
        base_url: str,
        user_agents: Optional[List[str]] = None,
        rate_limit_delay: float = 1.0,
        connect_timeout: float = 10.0,
        read_timeout: float = 30.0,
        max_retries: int = 3,
        respect_robots_txt: bool = True,
        log_dir: str = "logs",
        log_level: str = "INFO"
    ):
        """
        Initialize the base scraper.
        
        Args:
            name: Identifier for this scraper (used in logging)
            base_url: Base URL of the target website
            user_agents: List of user agent strings (defaults to common browsers)
            rate_limit_delay: Minimum seconds between requests (default: 1.0)
            connect_timeout: Connection timeout in seconds
            read_timeout: Read timeout in seconds
            max_retries: Maximum retry attempts for failed requests
            respect_robots_txt: Whether to check and respect robots.txt
            log_dir: Directory for log files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        
        Raises:
            ConfigurationError: If configuration is invalid
        """
        self.name = name
        self.base_url = base_url.rstrip('/')
        self.rate_limit_delay = rate_limit_delay
        self.respect_robots_txt = respect_robots_txt
        
        # Validate configuration
        self._validate_config()
        
        # Set up user agents
        self.user_agents = user_agents or self._get_default_user_agents()
        if not self.user_agents:
            raise ConfigurationError(
                "User agent list cannot be empty",
                config_key="user_agents"
            )
        self._current_user_agent_index = 0
        
        # Initialize HTTP client
        self.http_client = HTTPClient(
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            max_retries=max_retries,
            backoff_factor=1.0  # Exponential backoff: 1s, 2s, 4s, 8s...
        )
        
        # Set up logging
        self.logger = self._setup_logging(log_dir, log_level)
        
        # Track last request time for rate limiting
        self._last_request_time = 0.0
        
        # Initialize robots.txt parser
        self.robots_parser: Optional[RobotFileParser] = None
        if self.respect_robots_txt:
            self._load_robots_txt()
        
        self.logger.info(
            f"Initialized {self.name} scraper: "
            f"base_url={self.base_url}, "
            f"rate_limit={self.rate_limit_delay}s, "
            f"robots_txt={'enabled' if self.respect_robots_txt else 'disabled'}"
        )
    
    def _validate_config(self) -> None:
        """
        Validate configuration parameters.
        
        Raises:
            ConfigurationError: If configuration is invalid
        """
        if not self.name:
            raise ConfigurationError("Scraper name cannot be empty", config_key="name")
        
        if not self.base_url:
            raise ConfigurationError("Base URL cannot be empty", config_key="base_url")
        
        if self.rate_limit_delay < 0:
            raise ConfigurationError(
                f"Rate limit delay must be non-negative, got {self.rate_limit_delay}",
                config_key="rate_limit_delay"
            )
    
    def _get_default_user_agents(self) -> List[str]:
        """
        Get default user agent strings for common browsers.
        
        Returns:
            List of user agent strings
        """
        return [
            # Chrome on Windows
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            # Firefox on Windows
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) "
            "Gecko/20100101 Firefox/121.0",
            # Chrome on macOS
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            # Safari on macOS
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
            "(KHTML, like Gecko) Version/17.1 Safari/605.1.15",
            # Edge on Windows
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0"
        ]
    
    def _get_next_user_agent(self) -> str:
        """
        Get the next user agent from the rotation list.
        
        Returns:
            User agent string
        """
        user_agent = self.user_agents[self._current_user_agent_index]
        self._current_user_agent_index = (
            (self._current_user_agent_index + 1) % len(self.user_agents)
        )
        return user_agent
    
    def _setup_logging(self, log_dir: str, log_level: str) -> logging.Logger:
        """
        Configure logging with rotating file handler.
        
        Sets up a logger with:
        - Rotating file handler (10MB max, 5 backups)
        - Console handler for development
        - Standard format with timestamp, name, level, and message
        
        Args:
            log_dir: Directory for log files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        
        Returns:
            Configured logger instance
        """
        # Create logs directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Create logger
        logger = logging.getLogger(f"scraper.{self.name}")
        logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        
        # Avoid duplicate handlers if logger already configured
        if logger.handlers:
            return logger
        
        # Create formatters
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        # Rotating file handler (10MB max, 5 backups)
        log_file = os.path.join(log_dir, "scraper.log")
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console handler for development/debugging
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def _load_robots_txt(self) -> None:
        """
        Load and parse robots.txt from the target website.
        
        Attempts to fetch and parse robots.txt. If it fails, logs a warning
        but continues (assumes allowed by default).
        """
        try:
            robots_url = f"{self.base_url}/robots.txt"
            self.robots_parser = RobotFileParser()
            self.robots_parser.set_url(robots_url)
            self.robots_parser.read()
            self.logger.info(f"Successfully loaded robots.txt from {robots_url}")
        except Exception as e:
            self.logger.warning(
                f"Failed to load robots.txt from {self.base_url}: {e}. "
                "Proceeding with assumption that scraping is allowed."
            )
            self.robots_parser = None
    
    def _can_fetch(self, url: str) -> bool:
        """
        Check if the URL can be fetched according to robots.txt.
        
        Args:
            url: URL to check
        
        Returns:
            True if URL can be fetched, False otherwise
        """
        if not self.respect_robots_txt or not self.robots_parser:
            return True
        
        user_agent = self._get_next_user_agent()
        can_fetch = self.robots_parser.can_fetch(user_agent, url)
        
        if not can_fetch:
            self.logger.warning(f"robots.txt disallows fetching: {url}")
        
        return can_fetch
    
    def _enforce_rate_limit(self) -> None:
        """
        Enforce rate limiting by waiting if necessary.
        
        Ensures minimum delay between requests based on rate_limit_delay.
        Uses time.sleep() to pause execution.
        """
        if self.rate_limit_delay <= 0:
            return
        
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - elapsed
            self.logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
    
    def fetch_page(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> requests.Response:
        """
        Fetch a web page with rate limiting, user agent rotation, and robots.txt checking.
        
        This method:
        1. Checks robots.txt compliance
        2. Enforces rate limiting
        3. Rotates user agent
        4. Makes HTTP request with retry logic
        5. Updates last request time
        
        Args:
            url: URL to fetch
            headers: Optional custom headers (user-agent will be added/overridden)
            params: Optional query parameters
        
        Returns:
            Response object
        
        Raises:
            ConfigurationError: If robots.txt disallows the URL
            NetworkError: On connection failures
            RateLimitError: On rate limit exceeded
        """
        # Check robots.txt
        if not self._can_fetch(url):
            raise ConfigurationError(
                f"robots.txt disallows access to {url}",
                config_key="robots_txt"
            )
        
        # Enforce rate limiting
        self._enforce_rate_limit()
        
        # Prepare headers with rotated user agent
        request_headers = headers.copy() if headers else {}
        request_headers['User-Agent'] = self._get_next_user_agent()
        
        # Make request
        try:
            response = self.http_client.get(url, headers=request_headers, params=params)
            self._last_request_time = time.time()
            return response
        except Exception as e:
            self.logger.error(f"Failed to fetch {url}: {e}")
            raise
    
    def parse_html(self, html_content: str, parser: str = "lxml") -> BeautifulSoup:
        """
        Parse HTML content into BeautifulSoup object.
        
        Args:
            html_content: Raw HTML string
            parser: HTML parser to use (default: lxml)
        
        Returns:
            BeautifulSoup object
        
        Raises:
            ParsingError: If HTML parsing fails
        """
        try:
            return BeautifulSoup(html_content, parser)
        except Exception as e:
            self.logger.error(f"Failed to parse HTML: {e}")
            raise ParsingError(f"HTML parsing failed: {e}") from e
    
    @abstractmethod
    def parse_laptop(self, url: str) -> Dict[str, Any]:
        """
        Extract laptop details from a product page.
        
        This method must be implemented by subclasses to extract
        laptop-specific information from individual product pages.
        
        Args:
            url: URL of the laptop product page
        
        Returns:
            Dictionary containing laptop details (model, price, specs, etc.)
        
        Raises:
            ParsingError: If required data cannot be extracted
            NetworkError: On connection failures
        """
        pass
    
    @abstractmethod
    def get_listings_page(self, page: int = 1) -> List[str]:
        """
        Fetch and parse a listings/search page to get product URLs.
        
        This method must be implemented by subclasses to extract
        product URLs from listing pages.
        
        Args:
            page: Page number to fetch (default: 1)
        
        Returns:
            List of product URLs found on the page
        
        Raises:
            ParsingError: If listing page cannot be parsed
            NetworkError: On connection failures
        """
        pass
    
    def close(self) -> None:
        """
        Clean up resources (close HTTP client session).
        
        Should be called when scraping is complete to properly
        release network resources.
        """
        if self.http_client:
            self.http_client.close()
            self.logger.info(f"{self.name} scraper closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically close resources."""
        self.close()
