"""
Configuration module for Jarvis AI Assistant.
This module handles all configuration settings for the assistant.
"""

import os
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class Config:
    """Configuration class for Jarvis AI Assistant."""
    
    # Default configuration values
    DEFAULT_CONFIG = {
        "llm": {
            "model_name": "bigscience/bloom-560m",  # A smaller model for faster loading
            "model_path": "models/llm",
            "use_gpu": False,
            "max_new_tokens": 100,
            "temperature": 0.7
        },
        "speech": {
            "stt_model": "tiny",  # Options: tiny, base, small, medium, large
            "stt_model_path": "models/whisper",
            "tts_voice": None,  # None means system default
            "tts_rate": 175,
            "tts_volume": 1.0
        },
        "memory": {
            "vector_db_path": "data/vector_db",
            "history_size": 20,
            "embedding_model": "all-MiniLM-L6-v2"
        },
        "application": {
            "name": "Jarvis",
            "listening_indicator": "Listening...",
            "thinking_indicator": "Thinking...",
            "speaking_indicator": "Speaking...",
            "assistant_prompt": "I am Jarvis, your AI assistant. I run completely offline on your machine. I can help you with various tasks, answer questions, and execute commands for you."
        }
    }
    
    def __init__(self, config_path="config.json"):
        """Initialize configuration from file or defaults."""
        self.config_path = config_path
        self.config = self._load_config()
        
        # Create necessary directories
        self._create_directories()
    
    def _load_config(self):
        """Load configuration from file or create default if not exists."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    logger.info(f"Loading configuration from {self.config_path}")
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
                logger.info("Using default configuration")
                return self.DEFAULT_CONFIG
        else:
            logger.info(f"Configuration file not found. Creating default at {self.config_path}")
            config = self.DEFAULT_CONFIG
            self._save_config(config)
            return config
    
    def _save_config(self, config):
        """Save configuration to file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def _create_directories(self):
        """Create necessary directories for the application."""
        directories = [
            self.get_model_path(),
            self.get_vector_db_path(),
            "data",
            "logs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def save(self):
        """Save current configuration to file."""
        self._save_config(self.config)
    
    def get_llm_config(self):
        """Get LLM configuration."""
        return self.config["llm"]
    
    def get_speech_config(self):
        """Get speech configuration."""
        return self.config["speech"]
    
    def get_memory_config(self):
        """Get memory configuration."""
        return self.config["memory"]
    
    def get_app_config(self):
        """Get application configuration."""
        return self.config["application"]
    
    def get_model_path(self):
        """Get model path."""
        return self.config["llm"]["model_path"]
    
    def get_whisper_model_path(self):
        """Get Whisper model path."""
        return self.config["speech"]["stt_model_path"]
    
    def get_vector_db_path(self):
        """Get vector DB path."""
        return self.config["memory"]["vector_db_path"]
    
    def update(self, section, key, value):
        """Update a specific configuration value."""
        if section in self.config and key in self.config[section]:
            self.config[section][key] = value
            self.save()
            return True
        return False
