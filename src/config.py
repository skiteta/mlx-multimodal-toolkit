import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import logging


class Config:
    """Configuration management for the toolkit"""
    
    DEFAULT_CONFIG = {
        "models": {
            "default_llm": "Phi-3.5-mini-instruct-4bit",
            "default_vlm": "SmolVLM-Instruct-bf16",
            "default_alm": "whisper-large-v3-mlx",
        },
        "generation": {
            "max_tokens": 500,
            "temperature": 0.8,
            "top_p": 0.95,
            "repetition_penalty": 1.0,
        },
        "system": {
            "cache_dir": "~/.cache/mlx_models",
            "logging_level": "INFO",
            "trust_remote_code": True,
        },
        "batch": {
            "max_concurrent": 4,
            "timeout": 300,
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration
        
        Args:
            config_path: Path to config file. If None, looks for config.yaml in current dir
        """
        self.config_path = config_path or "config.yaml"
        self.config = self.DEFAULT_CONFIG.copy()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load config from file
        self.load_config()
        
        # Override with environment variables
        self.load_env_vars()
        
    def load_config(self):
        """Load configuration from YAML file"""
        config_file = Path(self.config_path)
        
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    file_config = yaml.safe_load(f)
                    
                if file_config:
                    self._merge_config(self.config, file_config)
                    self.logger.info(f"Loaded config from {config_file}")
                    
            except Exception as e:
                self.logger.error(f"Failed to load config file {config_file}: {e}")
                self.logger.info("Using default configuration")
        else:
            self.logger.info(f"Config file {config_file} not found, using defaults")
            
    def load_env_vars(self):
        """Load configuration from environment variables"""
        env_mappings = {
            "MLX_CACHE_DIR": ("system", "cache_dir"),
            "MLX_LOG_LEVEL": ("system", "logging_level"),
            "MLX_MAX_TOKENS": ("generation", "max_tokens"),
            "MLX_TEMPERATURE": ("generation", "temperature"),
            "MLX_TOP_P": ("generation", "top_p"),
            "MLX_DEFAULT_LLM": ("models", "default_llm"),
            "MLX_DEFAULT_VLM": ("models", "default_vlm"),
            "MLX_DEFAULT_ALM": ("models", "default_alm"),
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                # Convert to appropriate type
                if key in ["max_tokens", "max_concurrent", "timeout"]:
                    value = int(value)
                elif key in ["temperature", "top_p", "repetition_penalty"]:
                    value = float(value)
                elif key == "trust_remote_code":
                    value = value.lower() in ("true", "1", "yes", "on")
                    
                self.config[section][key] = value
                self.logger.debug(f"Set {section}.{key} = {value} from environment")
                
    def _merge_config(self, base: Dict[str, Any], override: Dict[str, Any]):
        """Recursively merge configuration dictionaries"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
                
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(section, {}).get(key, default)
        
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section"""
        return self.config.get(section, {}).copy()
        
    def set(self, section: str, key: str, value: Any):
        """Set configuration value"""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
        
    def save_config(self, path: Optional[str] = None):
        """Save current configuration to file"""
        output_path = path or self.config_path
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            self.logger.info(f"Configuration saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save config to {output_path}: {e}")
            raise
            
    def validate_config(self) -> bool:
        """Validate configuration values"""
        errors = []
        
        # Validate generation parameters
        gen_config = self.config.get("generation", {})
        
        if gen_config.get("max_tokens", 0) <= 0:
            errors.append("max_tokens must be positive")
            
        if not (0 < gen_config.get("temperature", 0.8) <= 2.0):
            errors.append("temperature must be between 0 and 2.0")
            
        if not (0 < gen_config.get("top_p", 0.95) <= 1.0):
            errors.append("top_p must be between 0 and 1.0")
            
        # Validate system settings
        sys_config = self.config.get("system", {})
        cache_dir = sys_config.get("cache_dir")
        
        if cache_dir:
            try:
                Path(cache_dir).expanduser().mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Invalid cache_dir: {e}")
                
        if errors:
            self.logger.error(f"Configuration validation errors: {errors}")
            return False
            
        return True
        
    def create_default_config(self, path: str = "config.yaml"):
        """Create a default configuration file"""
        try:
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(self.DEFAULT_CONFIG, f, default_flow_style=False, indent=2)
            self.logger.info(f"Created default config at {path}")
            
        except Exception as e:
            self.logger.error(f"Failed to create default config: {e}")
            raise
            
    def __repr__(self):
        return f"Config(path={self.config_path})"