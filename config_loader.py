#!/usr/bin/env python3
"""
Configuration Loader Module

This module provides a robust configuration loading system for the laptop price scraper.
It handles YAML-based configuration files with comprehensive validation, type checking,
and error handling.

Classes:
    ConfigurationError: Custom exception for configuration-related errors
    ConfigLoader: Main configuration loader with validation capabilities

Usage:
    >>> from config_loader import ConfigLoader
    >>> loader = ConfigLoader()
    >>> scraper_config = loader.load_scraper_config()
    >>> database_config = loader.load_database_config()
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Union


class ConfigurationError(Exception):
    """
    Custom exception raised when configuration loading or validation fails.
    
    This exception provides detailed error messages to help diagnose
    configuration issues such as missing files, invalid YAML syntax,
    missing required fields, or type mismatches.
    
    Attributes:
        message (str): Human-readable error description
        config_file (str): Path to the configuration file that caused the error
        field (str): Specific field that failed validation (if applicable)
    """
    
    def __init__(self, message: str, config_file: Optional[str] = None, field: Optional[str] = None):
        """
        Initialize ConfigurationError with detailed context.
        
        Args:
            message: Description of the configuration error
            config_file: Path to the problematic config file
            field: Name of the field that failed validation
        """
        self.message = message
        self.config_file = config_file
        self.field = field
        
        # Build comprehensive error message
        error_parts = [message]
        if config_file:
            error_parts.append(f"Config file: {config_file}")
        if field:
            error_parts.append(f"Field: {field}")
        
        super().__init__(" | ".join(error_parts))


class ConfigLoader:
    """
    Configuration loader with validation and type checking.
    
    This class handles loading YAML configuration files for both scraper settings
    and database configuration. It provides robust validation including:
    - Required field checking
    - Type validation (int, float, bool, str)
    - Range validation for numeric values
    - Default value application for optional fields
    - Helpful error messages for debugging
    
    Attributes:
        config_dir (Path): Directory containing configuration files
        scraper_config_path (Path): Path to scrapers.yaml
        database_config_path (Path): Path to database.yaml
    
    Configuration Schema:
        
        scrapers.yaml:
            - global: Global scraper settings
                - user_agent (str): HTTP User-Agent header
                - request_timeout (int): Request timeout in seconds
                - retry_attempts (int): Number of retry attempts
                - delay_between_requests (float): Delay between requests
                - respect_robots_txt (bool): Whether to respect robots.txt
            - sites: Dictionary of site configurations
                - [site_name]:
                    - enabled (bool): Whether site is enabled
                    - base_url (str): Base URL of the site
                    - search_url (str): Search page URL
                    - selectors (dict): CSS selectors for data extraction
                    - delays (dict): Site-specific delay settings
        
        database.yaml:
            - storage: Storage configuration
                - csv: CSV file settings
                    - path (str): Path to CSV file
                    - encoding (str): File encoding
                    - backup_enabled (bool): Enable backups
                - sqlite: SQLite database settings
                    - path (str): Path to SQLite database
                    - backup_enabled (bool): Enable backups
                    - backup_frequency (str): Backup frequency
    
    Example:
        >>> loader = ConfigLoader()
        >>> config = loader.load_scraper_config()
        >>> print(config['global']['user_agent'])
        >>> 
        >>> # Access site-specific configuration
        >>> amazon_config = config['sites']['amazon']
        >>> if amazon_config['enabled']:
        >>>     selectors = amazon_config['selectors']
    """
    
    # Default values for optional configuration fields
    DEFAULT_GLOBAL_CONFIG = {
        'user_agent': 'Mozilla/5.0 (compatible; LaptopScraper/1.0)',
        'request_timeout': 30,
        'retry_attempts': 3,
        'delay_between_requests': 2.0,
        'respect_robots_txt': True,
        'log_level': 'INFO',
        'log_requests': True,
        'min_price': 100,
        'max_price': 10000
    }
    
    DEFAULT_SITE_DELAYS = {
        'request_delay': 2.0,
        'page_delay': 5.0
    }
    
    DEFAULT_PAGINATION = {
        'enabled': True,
        'max_pages': 10,
        'next_button_selector': None
    }
    
    # Required fields for validation
    REQUIRED_GLOBAL_FIELDS = ['user_agent', 'request_timeout', 'retry_attempts']
    REQUIRED_SITE_FIELDS = ['enabled', 'base_url', 'selectors']
    REQUIRED_SELECTOR_FIELDS = [
        'company', 'product', 'type_name', 'inches', 'screen_resolution',
        'cpu', 'ram', 'memory', 'gpu', 'os', 'weight', 'price'
    ]
    REQUIRED_CSV_FIELDS = ['path', 'encoding']
    REQUIRED_SQLITE_FIELDS = ['path', 'backup_frequency']
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize ConfigLoader with configuration directory path.
        
        Args:
            config_dir: Path to directory containing configuration files
                       (default: "config")
        
        Raises:
            ConfigurationError: If config directory doesn't exist
        """
        self.config_dir = Path(config_dir)
        
        # Validate config directory exists
        if not self.config_dir.exists():
            raise ConfigurationError(
                f"Configuration directory not found: {self.config_dir}",
                config_file=str(self.config_dir)
            )
        
        # Set configuration file paths
        self.scraper_config_path = self.config_dir / "scrapers.yaml"
        self.database_config_path = self.config_dir / "database.yaml"
    
    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Load and parse a YAML file with error handling.
        
        Args:
            file_path: Path to the YAML file to load
        
        Returns:
            Dictionary containing parsed YAML data
        
        Raises:
            ConfigurationError: If file doesn't exist, can't be read, or contains invalid YAML
        """
        # Check file exists
        if not file_path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {file_path}",
                config_file=str(file_path)
            )
        
        # Try to read and parse YAML
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            # Validate that YAML parsed to a dictionary
            if not isinstance(config, dict):
                raise ConfigurationError(
                    f"Configuration file must contain a YAML dictionary, got {type(config).__name__}",
                    config_file=str(file_path)
                )
            
            return config
            
        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Invalid YAML syntax: {str(e)}",
                config_file=str(file_path)
            )
        except Exception as e:
            raise ConfigurationError(
                f"Error reading configuration file: {str(e)}",
                config_file=str(file_path)
            )
    
    def _validate_type(self, value: Any, expected_type: type, field_name: str, config_file: str) -> None:
        """
        Validate that a value matches the expected type.
        
        Args:
            value: Value to validate
            expected_type: Expected Python type (int, float, str, bool, dict, list)
            field_name: Name of the field being validated
            config_file: Path to configuration file for error messages
        
        Raises:
            ConfigurationError: If type doesn't match
        """
        # Handle Union types (e.g., int or float)
        if expected_type == Union[int, float]:
            if not isinstance(value, (int, float)):
                raise ConfigurationError(
                    f"Field '{field_name}' must be a number, got {type(value).__name__}",
                    config_file=config_file,
                    field=field_name
                )
        elif not isinstance(value, expected_type):
            raise ConfigurationError(
                f"Field '{field_name}' must be {expected_type.__name__}, got {type(value).__name__}",
                config_file=config_file,
                field=field_name
            )
    
    def _validate_range(self, value: Union[int, float], min_val: Optional[float] = None,
                       max_val: Optional[float] = None, field_name: str = "", 
                       config_file: str = "") -> None:
        """
        Validate that a numeric value falls within a specified range.
        
        Args:
            value: Numeric value to validate
            min_val: Minimum allowed value (inclusive)
            max_val: Maximum allowed value (inclusive)
            field_name: Name of the field being validated
            config_file: Path to configuration file for error messages
        
        Raises:
            ConfigurationError: If value is out of range
        """
        if min_val is not None and value < min_val:
            raise ConfigurationError(
                f"Field '{field_name}' must be >= {min_val}, got {value}",
                config_file=config_file,
                field=field_name
            )
        
        if max_val is not None and value > max_val:
            raise ConfigurationError(
                f"Field '{field_name}' must be <= {max_val}, got {value}",
                config_file=config_file,
                field=field_name
            )
    
    def _apply_defaults(self, config: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply default values for missing optional fields.
        
        Args:
            config: Configuration dictionary (may be incomplete)
            defaults: Dictionary of default values
        
        Returns:
            Configuration dictionary with defaults applied
        """
        result = config.copy()
        for key, default_value in defaults.items():
            if key not in result:
                result[key] = default_value
        return result
    
    def _validate_required_fields(self, config: Dict[str, Any], required_fields: List[str],
                                  config_file: str, section: str = "") -> None:
        """
        Validate that all required fields are present in the configuration.
        
        Args:
            config: Configuration dictionary to validate
            required_fields: List of required field names
            config_file: Path to configuration file for error messages
            section: Section name for better error messages
        
        Raises:
            ConfigurationError: If any required field is missing
        """
        missing_fields = [field for field in required_fields if field not in config]
        
        if missing_fields:
            section_msg = f" in section '{section}'" if section else ""
            raise ConfigurationError(
                f"Missing required fields{section_msg}: {', '.join(missing_fields)}",
                config_file=config_file
            )
    
    def validate_scraper_config(self, config: Dict[str, Any]) -> None:
        """
        Validate the scraper configuration structure and values.
        
        Performs comprehensive validation including:
        - Required fields presence
        - Type checking for all fields
        - Range validation for timeouts, delays, and retry attempts
        - Selector completeness for each site
        
        Args:
            config: Scraper configuration dictionary to validate
        
        Raises:
            ConfigurationError: If validation fails
        """
        config_file = str(self.scraper_config_path)
        
        # Validate top-level structure
        if 'global' not in config:
            raise ConfigurationError(
                "Missing required 'global' section in scraper configuration",
                config_file=config_file
            )
        
        if 'sites' not in config:
            raise ConfigurationError(
                "Missing required 'sites' section in scraper configuration",
                config_file=config_file
            )
        
        # Validate global configuration
        global_config = config['global']
        self._validate_required_fields(
            global_config, self.REQUIRED_GLOBAL_FIELDS, config_file, "global"
        )
        
        # Type validation for global config
        self._validate_type(global_config['user_agent'], str, 'global.user_agent', config_file)
        self._validate_type(global_config['request_timeout'], int, 'global.request_timeout', config_file)
        self._validate_type(global_config['retry_attempts'], int, 'global.retry_attempts', config_file)
        
        # Range validation for global config
        self._validate_range(
            global_config['request_timeout'], min_val=1, max_val=300,
            field_name='global.request_timeout', config_file=config_file
        )
        self._validate_range(
            global_config['retry_attempts'], min_val=0, max_val=10,
            field_name='global.retry_attempts', config_file=config_file
        )
        
        # Validate optional global fields if present
        if 'delay_between_requests' in global_config:
            self._validate_type(global_config['delay_between_requests'], (int, float),
                              'global.delay_between_requests', config_file)
            self._validate_range(
                global_config['delay_between_requests'], min_val=0.0, max_val=60.0,
                field_name='global.delay_between_requests', config_file=config_file
            )
        
        if 'respect_robots_txt' in global_config:
            self._validate_type(global_config['respect_robots_txt'], bool,
                              'global.respect_robots_txt', config_file)
        
        # Validate each site configuration
        sites = config['sites']
        if not isinstance(sites, dict):
            raise ConfigurationError(
                "'sites' must be a dictionary of site configurations",
                config_file=config_file
            )
        
        if not sites:
            raise ConfigurationError(
                "At least one site configuration is required in 'sites' section",
                config_file=config_file
            )
        
        for site_name, site_config in sites.items():
            self._validate_site_config(site_name, site_config, config_file)
    
    def _validate_site_config(self, site_name: str, site_config: Dict[str, Any], config_file: str) -> None:
        """
        Validate a single site configuration.
        
        Args:
            site_name: Name of the site
            site_config: Site configuration dictionary
            config_file: Path to configuration file for error messages
        
        Raises:
            ConfigurationError: If validation fails
        """
        section = f"sites.{site_name}"
        
        # Validate required fields
        self._validate_required_fields(
            site_config, self.REQUIRED_SITE_FIELDS, config_file, section
        )
        
        # Type validation
        self._validate_type(site_config['enabled'], bool, f'{section}.enabled', config_file)
        self._validate_type(site_config['base_url'], str, f'{section}.base_url', config_file)
        self._validate_type(site_config['selectors'], dict, f'{section}.selectors', config_file)
        
        # Validate URLs start with http/https
        if not site_config['base_url'].startswith(('http://', 'https://')):
            raise ConfigurationError(
                f"base_url must start with http:// or https://",
                config_file=config_file,
                field=f'{section}.base_url'
            )
        
        # Validate selectors
        selectors = site_config['selectors']
        self._validate_required_fields(
            selectors, self.REQUIRED_SELECTOR_FIELDS, config_file, f"{section}.selectors"
        )
        
        # Validate all selectors are strings
        for selector_name, selector_value in selectors.items():
            self._validate_type(
                selector_value, str, f'{section}.selectors.{selector_name}', config_file
            )
        
        # Validate delays if present
        if 'delays' in site_config:
            delays = site_config['delays']
            self._validate_type(delays, dict, f'{section}.delays', config_file)
            
            for delay_name, delay_value in delays.items():
                self._validate_type(
                    delay_value, (int, float), f'{section}.delays.{delay_name}', config_file
                )
                self._validate_range(
                    delay_value, min_val=0.0, max_val=120.0,
                    field_name=f'{section}.delays.{delay_name}', config_file=config_file
                )
        
        # Validate pagination if present
        if 'pagination' in site_config:
            pagination = site_config['pagination']
            self._validate_type(pagination, dict, f'{section}.pagination', config_file)
            
            if 'enabled' in pagination:
                self._validate_type(pagination['enabled'], bool,
                                  f'{section}.pagination.enabled', config_file)
            
            if 'max_pages' in pagination:
                self._validate_type(pagination['max_pages'], int,
                                  f'{section}.pagination.max_pages', config_file)
                self._validate_range(
                    pagination['max_pages'], min_val=1, max_val=1000,
                    field_name=f'{section}.pagination.max_pages', config_file=config_file
                )
    
    def validate_database_config(self, config: Dict[str, Any]) -> None:
        """
        Validate the database configuration structure and values.
        
        Performs validation including:
        - Required storage sections presence
        - Type checking for all fields
        - Path validation
        - Encoding validation
        
        Args:
            config: Database configuration dictionary to validate
        
        Raises:
            ConfigurationError: If validation fails
        """
        config_file = str(self.database_config_path)
        
        # Validate top-level structure
        if 'storage' not in config:
            raise ConfigurationError(
                "Missing required 'storage' section in database configuration",
                config_file=config_file
            )
        
        storage = config['storage']
        
        # Validate CSV configuration if present
        if 'csv' in storage:
            csv_config = storage['csv']
            self._validate_required_fields(
                csv_config, self.REQUIRED_CSV_FIELDS, config_file, "storage.csv"
            )
            
            self._validate_type(csv_config['path'], str, 'storage.csv.path', config_file)
            self._validate_type(csv_config['encoding'], str, 'storage.csv.encoding', config_file)
            
            if 'backup_enabled' in csv_config:
                self._validate_type(csv_config['backup_enabled'], bool,
                                  'storage.csv.backup_enabled', config_file)
        
        # Validate SQLite configuration if present
        if 'sqlite' in storage:
            sqlite_config = storage['sqlite']
            self._validate_required_fields(
                sqlite_config, self.REQUIRED_SQLITE_FIELDS, config_file, "storage.sqlite"
            )
            
            self._validate_type(sqlite_config['path'], str, 'storage.sqlite.path', config_file)
            self._validate_type(sqlite_config['backup_frequency'], str,
                              'storage.sqlite.backup_frequency', config_file)
            
            # Validate backup frequency value
            valid_frequencies = ['daily', 'weekly', 'monthly']
            if sqlite_config['backup_frequency'] not in valid_frequencies:
                raise ConfigurationError(
                    f"backup_frequency must be one of {valid_frequencies}, "
                    f"got '{sqlite_config['backup_frequency']}'",
                    config_file=config_file,
                    field='storage.sqlite.backup_frequency'
                )
            
            if 'backup_enabled' in sqlite_config:
                self._validate_type(sqlite_config['backup_enabled'], bool,
                                  'storage.sqlite.backup_enabled', config_file)
            
            if 'timeout' in sqlite_config:
                self._validate_type(sqlite_config['timeout'], (int, float),
                                  'storage.sqlite.timeout', config_file)
                self._validate_range(
                    sqlite_config['timeout'], min_val=1.0, max_val=300.0,
                    field_name='storage.sqlite.timeout', config_file=config_file
                )
    
    def load_scraper_config(self) -> Dict[str, Any]:
        """
        Load and validate the scraper configuration from scrapers.yaml.
        
        This method:
        1. Loads the YAML configuration file
        2. Applies default values for optional fields
        3. Validates all required fields and types
        4. Returns a validated configuration dictionary
        
        Returns:
            Dictionary containing complete and validated scraper configuration
            with the following structure:
            {
                'global': {
                    'user_agent': str,
                    'request_timeout': int,
                    'retry_attempts': int,
                    'delay_between_requests': float,
                    ...
                },
                'sites': {
                    'site_name': {
                        'enabled': bool,
                        'base_url': str,
                        'search_url': str,
                        'selectors': {...},
                        'delays': {...},
                        'pagination': {...}
                    },
                    ...
                }
            }
        
        Raises:
            ConfigurationError: If file doesn't exist, contains invalid YAML,
                              or fails validation
        
        Example:
            >>> loader = ConfigLoader()
            >>> config = loader.load_scraper_config()
            >>> user_agent = config['global']['user_agent']
            >>> amazon_enabled = config['sites']['amazon']['enabled']
        """
        # Load YAML file
        config = self._load_yaml_file(self.scraper_config_path)
        
        # Apply defaults to global configuration
        if 'global' in config:
            config['global'] = self._apply_defaults(
                config['global'], self.DEFAULT_GLOBAL_CONFIG
            )
        
        # Apply defaults to each site configuration
        if 'sites' in config:
            for site_name, site_config in config['sites'].items():
                # Apply delay defaults
                if 'delays' not in site_config:
                    site_config['delays'] = self.DEFAULT_SITE_DELAYS.copy()
                else:
                    site_config['delays'] = self._apply_defaults(
                        site_config['delays'], self.DEFAULT_SITE_DELAYS
                    )
                
                # Apply pagination defaults
                if 'pagination' not in site_config:
                    site_config['pagination'] = self.DEFAULT_PAGINATION.copy()
                else:
                    site_config['pagination'] = self._apply_defaults(
                        site_config['pagination'], self.DEFAULT_PAGINATION
                    )
        
        # Validate configuration
        self.validate_scraper_config(config)
        
        return config
    
    def load_database_config(self) -> Dict[str, Any]:
        """
        Load and validate the database configuration from database.yaml.
        
        This method:
        1. Loads the YAML configuration file
        2. Validates all required fields and types
        3. Returns a validated configuration dictionary
        
        Returns:
            Dictionary containing complete and validated database configuration
            with the following structure:
            {
                'storage': {
                    'csv': {
                        'path': str,
                        'encoding': str,
                        'backup_enabled': bool,
                        ...
                    },
                    'sqlite': {
                        'path': str,
                        'backup_enabled': bool,
                        'backup_frequency': str,
                        ...
                    }
                },
                'validation': {...}
            }
        
        Raises:
            ConfigurationError: If file doesn't exist, contains invalid YAML,
                              or fails validation
        
        Example:
            >>> loader = ConfigLoader()
            >>> config = loader.load_database_config()
            >>> csv_path = config['storage']['csv']['path']
            >>> encoding = config['storage']['csv']['encoding']
        """
        # Load YAML file
        config = self._load_yaml_file(self.database_config_path)
        
        # Validate configuration
        self.validate_database_config(config)
        
        return config
    
    def validate_config(self, config_type: str = "scraper") -> bool:
        """
        Validate a configuration without loading it into memory.
        
        Useful for pre-deployment validation or configuration testing.
        
        Args:
            config_type: Type of configuration to validate ("scraper" or "database")
        
        Returns:
            True if validation succeeds
        
        Raises:
            ConfigurationError: If validation fails
            ValueError: If config_type is invalid
        
        Example:
            >>> loader = ConfigLoader()
            >>> try:
            >>>     loader.validate_config("scraper")
            >>>     print("Scraper configuration is valid!")
            >>> except ConfigurationError as e:
            >>>     print(f"Configuration error: {e}")
        """
        if config_type == "scraper":
            config = self._load_yaml_file(self.scraper_config_path)
            self.validate_scraper_config(config)
        elif config_type == "database":
            config = self._load_yaml_file(self.database_config_path)
            self.validate_database_config(config)
        else:
            raise ValueError(f"Invalid config_type: {config_type}. Must be 'scraper' or 'database'")
        
        return True


# Convenience function for quick configuration loading
def load_config(config_type: str = "scraper", config_dir: str = "config") -> Dict[str, Any]:
    """
    Convenience function to quickly load a configuration.
    
    Args:
        config_type: Type of configuration to load ("scraper" or "database")
        config_dir: Path to configuration directory
    
    Returns:
        Loaded and validated configuration dictionary
    
    Raises:
        ConfigurationError: If loading or validation fails
    
    Example:
        >>> scraper_config = load_config("scraper")
        >>> database_config = load_config("database")
    """
    loader = ConfigLoader(config_dir)
    
    if config_type == "scraper":
        return loader.load_scraper_config()
    elif config_type == "database":
        return loader.load_database_config()
    else:
        raise ValueError(f"Invalid config_type: {config_type}. Must be 'scraper' or 'database'")


if __name__ == "__main__":
    """
    Module test code - validates both configuration files.
    Run this module directly to test configuration loading.
    """
    print("=" * 70)
    print("Configuration Loader Test")
    print("=" * 70)
    
    try:
        loader = ConfigLoader()
        
        # Test scraper configuration
        print("\n📝 Loading scraper configuration...")
        scraper_config = loader.load_scraper_config()
        print(f"✅ Scraper configuration loaded successfully!")
        print(f"   - Global settings: {len(scraper_config['global'])} fields")
        print(f"   - Sites configured: {len(scraper_config['sites'])}")
        
        enabled_sites = [name for name, cfg in scraper_config['sites'].items() if cfg['enabled']]
        print(f"   - Enabled sites: {', '.join(enabled_sites)}")
        
        # Test database configuration
        print("\n📝 Loading database configuration...")
        database_config = loader.load_database_config()
        print(f"✅ Database configuration loaded successfully!")
        print(f"   - CSV path: {database_config['storage']['csv']['path']}")
        print(f"   - SQLite path: {database_config['storage']['sqlite']['path']}")
        
        print("\n" + "=" * 70)
        print("✅ All configurations validated successfully!")
        print("=" * 70)
        
    except ConfigurationError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("=" * 70)
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        print("=" * 70)
