"""
Speech processor component for Jarvis AI Assistant.

Handles speech-to-text and text-to-speech functionalities.
"""

import os
import logging
import tempfile
import queue
import threading
import wave
import time
import numpy as np
from pathlib import Path
import pyttsx3
import pyaudio
import torch
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration

logger = logging.getLogger(__name__)

class SpeechProcessor:
    """Handles speech-to-text and text-to-speech functionalities."""
    
    def __init__(self, config):
        """
        Initialize the speech processor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.speech_config = config.get_speech_config()
        
        # Initialize TTS engine
        self.tts_engine = pyttsx3.init()
        self._configure_tts()
        
        # Initialize STT
        self.stt_model = self.speech_config["stt_model"]
        self.stt_model_path = self.speech_config["stt_model_path"]
        self.whisper = None  # Lazy load to save memory
        
        # Audio recording settings
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.CHUNK = 1024
        self.audio = pyaudio.PyAudio()
        
        # Recording state
        self.recording = False
        self.audio_queue = queue.Queue()
        
        # Check if CUDA is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
    
    def _configure_tts(self):
        """Configure Text-to-Speech engine."""
        voice = self.speech_config["tts_voice"]
        rate = self.speech_config["tts_rate"]
        volume = self.speech_config["tts_volume"]
        
        # Set voice if specified
        if voice:
            voices = self.tts_engine.getProperty('voices')
            for v in voices:
                if voice.lower() in v.name.lower():
                    self.tts_engine.setProperty('voice', v.id)
                    break
        
        # Set rate and volume
        self.tts_engine.setProperty('rate', rate)
        self.tts_engine.setProperty('volume', volume)
    
    def _load_whisper_model(self):
        """Load Whisper model for speech recognition."""
        if self.whisper is None:
            logger.info(f"Loading Whisper model: {self.stt_model}")
            
            try:
                # Try to load from local path first
                local_model_path = os.path.join(self.stt_model_path, self.stt_model)
                
                if os.path.exists(local_model_path):
                    logger.info(f"Loading Whisper model from local path: {local_model_path}")
                    processor = WhisperProcessor.from_pretrained(local_model_path)
                    model = WhisperForConditionalGeneration.from_pretrained(local_model_path).to(self.device)
                    self.whisper = pipeline(
                        "automatic-speech-recognition",
                        model=model,
                        tokenizer=processor.tokenizer,
                        feature_extractor=processor.feature_extractor,
                        chunk_length_s=30,
                        device=self.device
                    )
                else:
                    # Fall back to downloading from Hugging Face
                    logger.info(f"Loading Whisper model from Hugging Face: {self.stt_model}")
                    self.whisper = pipeline(
                        "automatic-speech-recognition",
                        model=f"openai/whisper-{self.stt_model}",
                        chunk_length_s=30,
                        device=self.device
                    )
            except Exception as e:
                logger.error(f"Error loading Whisper model: {e}")
                raise RuntimeError(f"Failed to load Whisper model: {e}")
    
    def start_recording(self):
        """Start recording audio from microphone."""
        if self.recording:
            return
        
        self.recording = True
        self.audio_queue = queue.Queue()
        
        # Start recording thread
        self.recording_thread = threading.Thread(target=self._record_audio)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        logger.info("Started recording audio")
    
    def stop_recording(self):
        """Stop recording audio and return the transcription."""
        if not self.recording:
            return ""
        
        self.recording = False
        self.recording_thread.join(timeout=2.0)
        
        # Get all audio data from queue
        audio_data = []
        while not self.audio_queue.empty():
            audio_data.append(self.audio_queue.get())
        
        if not audio_data:
            logger.warning("No audio data recorded")
            return ""
        
        # Combine all audio chunks
        audio_data = b''.join(audio_data)
        
        # Save to temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            
            with wave.open(temp_path, 'wb') as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
                wf.setframerate(self.RATE)
                wf.writeframes(audio_data)
        
        logger.info(f"Audio saved to temporary file: {temp_path}")
        
        # Transcribe audio
        try:
            text = self.transcribe_audio(temp_path)
            logger.info(f"Transcription: {text}")
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except:
                pass
        
        return text
    
    def _record_audio(self):
        """Record audio from microphone and put chunks in queue."""
        stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )
        
        try:
            while self.recording:
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                self.audio_queue.put(data)
        finally:
            stream.stop_stream()
            stream.close()
    
    def transcribe_audio(self, audio_file):
        """
        Transcribe audio file to text.
        
        Args:
            audio_file (str): Path to audio file
            
        Returns:
            str: Transcribed text
        """
        # Load model if not loaded yet
        if self.whisper is None:
            self._load_whisper_model()
        
        try:
            result = self.whisper(audio_file)
            return result["text"].strip()
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return ""
    
    def speak(self, text):
        """
        Convert text to speech.
        
        Args:
            text (str): Text to speak
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not text:
            return False
        
        try:
            # Break long text into sentences for more natural speech
            sentences = text.split('. ')
            
            for sentence in sentences:
                if not sentence.strip():
                    continue
                    
                # Add period if missing
                if not sentence.endswith(('.', '?', '!')):
                    sentence += '.'
                
                self.tts_engine.say(sentence)
                self.tts_engine.runAndWait()
                
                # Small pause between sentences
                time.sleep(0.1)
            
            return True
        except Exception as e:
            logger.error(f"Error in text-to-speech: {e}")
            return False
    
    def change_voice(self, voice_name):
        """
        Change TTS voice.
        
        Args:
            voice_name (str): Name of the voice to use
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            voices = self.tts_engine.getProperty('voices')
            for voice in voices:
                if voice_name.lower() in voice.name.lower():
                    self.tts_engine.setProperty('voice', voice.id)
                    self.config.update("speech", "tts_voice", voice_name)
                    return True
            
            logger.warning(f"Voice '{voice_name}' not found")
            return False
        except Exception as e:
            logger.error(f"Error changing voice: {e}")
            return False
    
    def set_speech_rate(self, rate):
        """
        Set TTS speech rate.
        
        Args:
            rate (int): Speech rate (words per minute)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.tts_engine.setProperty('rate', rate)
            self.config.update("speech", "tts_rate", rate)
            return True
        except Exception as e:
            logger.error(f"Error setting speech rate: {e}")
            return False
    
    def set_speech_volume(self, volume):
        """
        Set TTS volume.
        
        Args:
            volume (float): Volume level (0.0 to 1.0)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            volume = max(0.0, min(1.0, volume))  # Clamp between 0 and 1
            self.tts_engine.setProperty('volume', volume)
            self.config.update("speech", "tts_volume", volume)
            return True
        except Exception as e:
            logger.error(f"Error setting speech volume: {e}")
            return False
    
    def list_available_voices(self):
        """
        List all available TTS voices.
        
        Returns:
            list: List of available voice names
        """
        try:
            voices = self.tts_engine.getProperty('voices')
            return [voice.name for voice in voices]
        except Exception as e:
            logger.error(f"Error listing voices: {e}")
            return []
    
    def cleanup(self):
        """Clean up resources."""
        if self.recording:
            self.stop_recording()
        
        try:
            self.audio.terminate()
        except:
            pass
