"""
Utility functions for the Jarvis AI Assistant.
"""

import os
import logging
import time
import subprocess
import platform
import datetime
import shutil
from pathlib import Path
import requests
import tqdm

logger = logging.getLogger(__name__)

def download_required_models(config):
    """
    Download required models for the application.
    
    Args:
        config: Configuration object
    """
    from huggingface_hub import snapshot_download
    import torch
    from transformers import pipeline
    
    logger.info("Checking and downloading required models...")
    
    # Create model directories if they don't exist
    os.makedirs(config.get_model_path(), exist_ok=True)
    os.makedirs(config.get_whisper_model_path(), exist_ok=True)
    
    # Download LLM model
    llm_config = config.get_llm_config()
    model_name = llm_config["model_name"]
    
    logger.info(f"Downloading LLM model: {model_name}")
    try:
        # This will download the model to the Hugging Face cache and can be used without internet later
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Save the model to specified path to ensure we have a local copy
        model.save_pretrained(config.get_model_path())
        tokenizer.save_pretrained(config.get_model_path())
        
        logger.info(f"LLM model downloaded to {config.get_model_path()}")
    except Exception as e:
        logger.error(f"Error downloading LLM model: {e}")
    
    # Download Whisper model
    stt_config = config.get_speech_config()
    whisper_model = stt_config["stt_model"]
    
    logger.info(f"Downloading Whisper model: {whisper_model}")
    try:
        # This will download the model to the Hugging Face cache
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        
        processor = WhisperProcessor.from_pretrained(f"openai/whisper-{whisper_model}")
        model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{whisper_model}")
        
        # Save the model to specified path
        model.save_pretrained(os.path.join(config.get_whisper_model_path(), whisper_model))
        processor.save_pretrained(os.path.join(config.get_whisper_model_path(), whisper_model))
        
        logger.info(f"Whisper model downloaded to {config.get_whisper_model_path()}")
    except Exception as e:
        logger.error(f"Error downloading Whisper model: {e}")
    
    # Download embedding model for memory
    memory_config = config.get_memory_config()
    embedding_model = memory_config["embedding_model"]
    
    logger.info(f"Downloading embedding model: {embedding_model}")
    try:
        from sentence_transformers import SentenceTransformer
        
        # This will download the model to the Hugging Face cache
        model = SentenceTransformer(embedding_model)
        
        logger.info(f"Embedding model downloaded")
    except Exception as e:
        logger.error(f"Error downloading embedding model: {e}")
    
    logger.info("All required models downloaded successfully")

def execute_system_command(command, return_output=False):
    """
    Execute a system command and optionally return the output.
    
    Args:
        command (str): Command to execute
        return_output (bool): Whether to return the command output
        
    Returns:
        str or None: Command output if return_output is True, None otherwise
    """
    try:
        if return_output:
            result = subprocess.check_output(command, shell=True, text=True)
            return result.strip()
        else:
            subprocess.run(command, shell=True, check=True)
            return None
    except subprocess.SubprocessError as e:
        logger.error(f"Error executing command '{command}': {e}")
        return f"Error: {e}" if return_output else None

def open_application(app_name):
    """
    Open an application based on platform.
    
    Args:
        app_name (str): Name of the application to open
        
    Returns:
        bool: True if successful, False otherwise
    """
    system = platform.system().lower()
    
    try:
        if system == 'windows':
            os.startfile(app_name)
        elif system == 'darwin':  # macOS
            subprocess.run(['open', '-a', app_name], check=True)
        else:  # Linux
            subprocess.run([app_name], check=True)
        return True
    except Exception as e:
        logger.error(f"Error opening application '{app_name}': {e}")
        return False

def format_reminder(reminder_text, time_str=None):
    """
    Format a reminder with text and optional time.
    
    Args:
        reminder_text (str): The reminder text
        time_str (str, optional): Time string for the reminder
        
    Returns:
        tuple: (reminder text, datetime object or None)
    """
    reminder_time = None
    
    if time_str:
        try:
            # Parse time string into datetime object
            current_time = datetime.datetime.now()
            
            # Handle relative time like "in 5 minutes"
            if "in" in time_str.lower() and "minute" in time_str.lower():
                try:
                    minutes = int(''.join(filter(str.isdigit, time_str)))
                    reminder_time = current_time + datetime.timedelta(minutes=minutes)
                except ValueError:
                    pass
            
            # Handle relative time like "in 2 hours"
            elif "in" in time_str.lower() and "hour" in time_str.lower():
                try:
                    hours = int(''.join(filter(str.isdigit, time_str)))
                    reminder_time = current_time + datetime.timedelta(hours=hours)
                except ValueError:
                    pass
            
            # Try to parse exact time
            elif ":" in time_str:
                time_parts = time_str.split(":")
                if len(time_parts) == 2:
                    hour, minute = map(int, time_parts)
                    reminder_time = current_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
                    
                    # If the time is in the past, assume it's for tomorrow
                    if reminder_time < current_time:
                        reminder_time += datetime.timedelta(days=1)
            
            # Try to handle "tomorrow" or specific dates
            elif "tomorrow" in time_str.lower():
                reminder_time = current_time + datetime.timedelta(days=1)
                reminder_time = reminder_time.replace(hour=9, minute=0, second=0, microsecond=0)  # Default to 9 AM
        
        except Exception as e:
            logger.error(f"Error parsing reminder time '{time_str}': {e}")
    
    return (reminder_text, reminder_time)

def get_system_info():
    """
    Get system information.
    
    Returns:
        dict: System information
    """
    info = {
        'platform': platform.platform(),
        'system': platform.system(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'hostname': platform.node()
    }
    
    # Add memory info if psutil is available
    try:
        import psutil
        memory = psutil.virtual_memory()
        info['memory_total'] = f"{memory.total / (1024**3):.2f} GB"
        info['memory_available'] = f"{memory.available / (1024**3):.2f} GB"
        info['memory_percent'] = f"{memory.percent}%"
        
        # Add disk info
        disk = psutil.disk_usage('/')
        info['disk_total'] = f"{disk.total / (1024**3):.2f} GB"
        info['disk_free'] = f"{disk.free / (1024**3):.2f} GB"
        info['disk_percent'] = f"{disk.percent}%"
        
        # Add CPU info
        info['cpu_cores'] = psutil.cpu_count(logical=False)
        info['cpu_threads'] = psutil.cpu_count(logical=True)
        info['cpu_percent'] = f"{psutil.cpu_percent()}%"
    except ImportError:
        pass
    
    return info

def chunked_text(text, chunk_size=4000):
    """
    Split text into chunks of a specified size.
    
    Args:
        text (str): Text to split
        chunk_size (int): Maximum size of each chunk
        
    Returns:
        list: List of text chunks
    """
    # Split by sentences to avoid cutting in the middle of a sentence
    sentences = text.replace('\n', ' ').split('. ')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # Add period back except for the last sentence if it doesn't have one
        if not sentence.endswith('.') and sentence != sentences[-1]:
            sentence += '.'
            
        # If adding this sentence would exceed chunk size, start a new chunk
        if len(current_chunk) + len(sentence) + 1 > chunk_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks
