"""
Configuration management for the trading system.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from .exceptions import ConfigError


class ConfigManager(BaseSettings):
    """
    Centralized configuration management using Pydantic settings.
    Loads from environment variables and YAML config files.
    """
    
    # System settings
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Database
    database_url: str = Field(env="DATABASE_URL")
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    
    # API Keys
    alpha_vantage_api_key: Optional[str] = Field(default=None, env="ALPHA_VANTAGE_API_KEY")
    quandl_api_key: Optional[str] = Field(default=None, env="QUANDL_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    
    # Configuration file
    config_file_path: str = Field(default="config/config.yaml")
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._config_data = self._load_config_file()
    
    def _load_config_file(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_path = Path(self.config_file_path)
        
        if not config_path.exists():
            raise ConfigError(f"Config file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"Error parsing config file: {e}")
        except Exception as e:
            raise ConfigError(f"Error loading config file: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., 'database.pool_size')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        try:
            keys = key.split('.')
            value = self._config_data
            
            for k in keys:
                value = value[k]
            
            return value
        except (KeyError, TypeError):
            return default
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get an entire configuration section.
        
        Args:
            section: Section name
            
        Returns:
            Dictionary containing section configuration
        """
        return self._config_data.get(section, {})
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == "development"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"


# Global configuration instance
config = ConfigManager()