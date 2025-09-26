#!/usr/bin/env python3
"""
audio_processor.py
Audio input and preprocessing module for SoundSense project.
Handles real-time audio capture and preprocessing for YAMNet model inference.
"""

import numpy as np
import threading
import queue
import time
import logging
from typing import Generator, Optional, Tuple
from dataclasses import dataclass

# Audio processing libraries
try:
    import pyaudio
except ImportError:
    print("PyAudio not found. Install with: pip install pyaudio")
    print("On Raspberry Pi, you may need: sudo apt-get install portaudio19-dev python3-pyaudio")
    raise

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AudioConfig:
    """Configuration for audio processing matching YAMNet requirements."""
    
    # YAMNet specific requirements
    SAMPLE_RATE: int = 16000  # YAMNet expects 16kHz audio
    WINDOW_LENGTH_SECONDS: float = 0.96  # YAMNet sliding window length
    HOP_LENGTH_SECONDS: float = 0.48  # YAMNet hop length
    
    # Calculated values
    WINDOW_LENGTH_SAMPLES: int = None
    HOP_LENGTH_SAMPLES: int = None
    
    # Audio capture settings
    CHANNELS: int = 1  # Mono audio
    CHUNK_SIZE: int = 1024  # PyAudio buffer size
    FORMAT: int = pyaudio.paFloat32  # 32-bit float format
    
    def __post_init__(self):
        """Calculate sample-based values from time-based parameters."""
        self.WINDOW_LENGTH_SAMPLES = int(self.WINDOW_LENGTH_SECONDS * self.SAMPLE_RATE)
        self.HOP_LENGTH_SAMPLES = int(self.HOP_LENGTH_SECONDS * self.SAMPLE_RATE)
        
        logger.info(f"Audio Config initialized:")
        logger.info(f"  Sample rate: {self.SAMPLE_RATE} Hz")
        logger.info(f"  Window: {self.WINDOW_LENGTH_SECONDS}s ({self.WINDOW_LENGTH_SAMPLES} samples)")
        logger.info(f"  Hop: {self.HOP_LENGTH_SECONDS}s ({self.HOP_LENGTH_SAMPLES} samples)")


class AudioProcessor:
    """
    Handles audio capture and preprocessing for YAMNet model.
    
    This class manages real-time audio capture from microphone and preprocesses
    it into the format expected by YAMNet, following the MemryX AsyncAccl pattern.
    """
    
    def __init__(self, device_index: Optional[int] = None, 
                 buffer_duration: float = 2.0):
        """
        Initialize the audio processor.
        
        Args:
            device_index: PyAudio device index for microphone. None for default.
            buffer_duration: Duration of audio buffer in seconds for continuous processing.
        """
        self.config = AudioConfig()
        self.device_index = device_index
        self.buffer_duration = buffer_duration
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # Audio buffer for continuous capture
        self.buffer_size = int(self.config.SAMPLE_RATE * self.buffer_duration)
        self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.buffer_lock = threading.Lock()
        
        # Queue for processed audio windows
        self.window_queue = queue.Queue(maxsize=10)
        
        # Control flags
        self.is_capturing = False
        self.capture_thread = None
        
        # Track position in buffer for sliding window
        self.current_position = 0
        
        logger.info(f"AudioProcessor initialized with buffer duration: {buffer_duration}s")
        
    def list_audio_devices(self):
        """List available audio input devices."""
        info = self.audio.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount')
        
        print("\nAvailable audio input devices:")
        print("-" * 40)
        for i in range(num_devices):
            info = self.audio.get_device_info_by_host_api_device_index(0, i)
            if info.get('maxInputChannels') > 0:
                print(f"Device {i}: {info.get('name')} - {info.get('maxInputChannels')} channels")
    
    def start_capture(self):
        """Start audio capture in a separate thread."""
        if self.is_capturing:
            logger.warning("Audio capture already started")
            return
        
        try:
            # Open audio stream
            self.stream = self.audio.open(
                format=self.config.FORMAT,
                channels=self.config.CHANNELS,
                rate=self.config.SAMPLE_RATE,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.config.CHUNK_SIZE,
                stream_callback=self._audio_callback
            )
            
            self.is_capturing = True
            self.stream.start_stream()
            
            # Start processing thread
            self.capture_thread = threading.Thread(target=self._process_audio_windows)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            
            logger.info("Audio capture started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start audio capture: {e}")
            raise
    
    def stop_capture(self):
        """Stop audio capture and cleanup resources."""
        if not self.is_capturing:
            return
        
        self.is_capturing = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        
        logger.info("Audio capture stopped")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """
        PyAudio callback for continuous audio capture.
        
        This callback is called by PyAudio when new audio data is available.
        """
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        # Convert byte data to numpy array
        audio_chunk = np.frombuffer(in_data, dtype=np.float32)
        
        # Add to circular buffer
        with self.buffer_lock:
            chunk_size = len(audio_chunk)
            if self.current_position + chunk_size <= self.buffer_size:
                self.audio_buffer[self.current_position:self.current_position + chunk_size] = audio_chunk
            else:
                # Wrap around circular buffer
                overflow = (self.current_position + chunk_size) - self.buffer_size
                self.audio_buffer[self.current_position:] = audio_chunk[:-overflow]
                self.audio_buffer[:overflow] = audio_chunk[-overflow:]
            
            self.current_position = (self.current_position + chunk_size) % self.buffer_size
        
        return (None, pyaudio.paContinue)
    
    def _process_audio_windows(self):
        """
        Process audio buffer to extract sliding windows for YAMNet.
        
        This runs in a separate thread and creates sliding windows
        according to YAMNet specifications.
        """
        last_window_time = 0
        
        while self.is_capturing:
            current_time = time.time()
            
            # Check if enough time has passed for next window (hop length)
            if current_time - last_window_time >= self.config.HOP_LENGTH_SECONDS:
                with self.buffer_lock:
                    # Extract window from circular buffer
                    window = self._extract_window()
                
                if window is not None:
                    # Preprocess the window
                    processed_window = self._preprocess_window(window)
                    
                    # Add to queue (non-blocking)
                    try:
                        self.window_queue.put_nowait(processed_window)
                        last_window_time = current_time
                    except queue.Full:
                        # Skip if queue is full (prevents memory issues)
                        logger.debug("Window queue full, skipping frame")
            
            # Small sleep to prevent CPU spinning
            time.sleep(0.01)
    
    def _extract_window(self) -> Optional[np.ndarray]:
        """
        Extract a window of audio from the circular buffer.
        
        Returns:
            Audio window of WINDOW_LENGTH_SAMPLES or None if not enough data.
        """
        window_samples = self.config.WINDOW_LENGTH_SAMPLES
        
        # Calculate start position for window
        start_pos = (self.current_position - window_samples) % self.buffer_size
        
        if start_pos < self.current_position:
            # Window is contiguous in buffer
            window = self.audio_buffer[start_pos:self.current_position].copy()
        else:
            # Window wraps around buffer
            window = np.concatenate([
                self.audio_buffer[start_pos:],
                self.audio_buffer[:self.current_position]
            ])
        
        # Ensure we have enough samples
        if len(window) < window_samples:
            # Pad with zeros if needed (for startup)
            window = np.pad(window, (0, window_samples - len(window)), mode='constant')
        elif len(window) > window_samples:
            window = window[:window_samples]
        
        return window
    
    def _preprocess_window(self, window: np.ndarray) -> np.ndarray:
        """
        Preprocess audio window for YAMNet model.
        
        Args:
            window: Raw audio window
            
        Returns:
            Preprocessed audio ready for YAMNet input
        """
        # Ensure audio is in the range [-1.0, +1.0]
        # PyAudio with paFloat32 should already be in this range,
        # but we'll clip for safety
        window = np.clip(window, -1.0, 1.0)
        
        # YAMNet expects shape (1, num_samples) for batch dimension
        window = window.reshape(1, -1)
        
        return window.astype(np.float32)
    
    def audio_generator(self) -> Generator[np.ndarray, None, None]:
        """
        Generator function for AsyncAccl input pipeline.
        
        This follows the MemryX pattern for input functions,
        yielding preprocessed audio windows for model inference.
        
        Yields:
            Preprocessed audio windows ready for YAMNet
        """
        logger.info("Starting audio generator for AsyncAccl")
        
        while self.is_capturing:
            try:
                # Get next window from queue (blocking with timeout)
                window = self.window_queue.get(timeout=1.0)
                
                # Log occasionally to show it's working
                if np.random.random() < 0.01:  # Log ~1% of windows
                    logger.debug(f"Yielding window shape: {window.shape}, "
                               f"range: [{window.min():.3f}, {window.max():.3f}]")
                
                yield window
                
            except queue.Empty:
                # No window available, continue waiting
                continue
            except Exception as e:
                logger.error(f"Error in audio generator: {e}")
                break
        
        logger.info("Audio generator stopped")
        return None
    
    def get_single_window(self) -> Optional[np.ndarray]:
        """
        Get a single preprocessed audio window (for testing).
        
        Returns:
            Single preprocessed audio window or None if not available
        """
        try:
            return self.window_queue.get_nowait()
        except queue.Empty:
            return None
    
    def __enter__(self):
        """Context manager entry."""
        self.start_capture()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_capture()
        if self.audio:
            self.audio.terminate()


def test_audio_capture():
    """Test function to verify audio capture is working."""
    print("Testing audio capture for SoundSense...")
    
    processor = AudioProcessor()
    
    # List available devices
    processor.list_audio_devices()
    
    # Use context manager for automatic cleanup
    with processor:
        print("\nCapturing audio for 5 seconds...")
        print("Speak or make sounds to test the microphone")
        
        # Capture for 5 seconds
        start_time = time.time()
        window_count = 0
        
        while time.time() - start_time < 5:
            window = processor.get_single_window()
            if window is not None:
                window_count += 1
                print(f"Window {window_count}: shape={window.shape}, "
                      f"mean={window.mean():.4f}, std={window.std():.4f}")
            time.sleep(0.1)
    
    print(f"\nTest complete. Captured {window_count} windows.")


if __name__ == "__main__":
    # Run test when executed directly
    test_audio_capture()