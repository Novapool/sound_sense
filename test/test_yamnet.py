#!/usr/bin/env python3
"""
Test script for YAMNet audio classification using MemryX accelerator.
This script tests the model with various audio files and logs detailed results.

Usage:
    python test_yamnet.py [audio_file_path]
    
If no audio file is provided, it will test all audio files in the test/samples directory.
"""

import os
import sys
import json
import numpy as np
import librosa
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import csv

# MemryX imports
from memryx import AsyncAccl

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
TEST_DIR = Path(__file__).parent
SAMPLES_DIR = TEST_DIR / "samples"
LOGS_DIR = TEST_DIR / "logs" 
RESULTS_DIR = TEST_DIR / "results"
MODEL_PATH = PROJECT_ROOT / "models" / "Audio_classification_YamNet_96_64_1_tflite.dfp"

# Create directories if they don't exist
for directory in [SAMPLES_DIR, LOGS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / f"yamnet_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class YAMNetTester:
    """
    YAMNet audio classification tester using MemryX accelerator.
    
    This class handles audio preprocessing, model inference, and result analysis
    following the MemryX AsyncAccl pattern from the tutorials.
    """
    
    def __init__(self):
        """Initialize the YAMNet tester with model configuration."""
        # YAMNet model specifications
        self.sample_rate = 16000  # YAMNet expects 16kHz audio
        self.window_length_seconds = 0.96  # YAMNet sliding window length
        self.hop_length_seconds = 0.48     # YAMNet hop length
        
        # Calculate samples for windowing
        self.window_length_samples = int(self.window_length_seconds * self.sample_rate)
        self.hop_length_samples = int(self.hop_length_seconds * self.sample_rate)
        
        # Model outputs
        self.num_classes = 521  # YAMNet predicts 521 AudioSet classes
        
        # Load class labels (AudioSet ontology)
        self.class_labels = self._load_audioset_labels()
        
        # Results storage
        self.current_audio_file = None
        self.results = []
        self.accelerator = None
        
        logger.info(f"YAMNet Tester initialized")
        logger.info(f"Sample rate: {self.sample_rate} Hz")
        logger.info(f"Window length: {self.window_length_seconds}s ({self.window_length_samples} samples)")
        logger.info(f"Hop length: {self.hop_length_seconds}s ({self.hop_length_samples} samples)")
    
    def _load_audioset_labels(self) -> Dict[int, str]:
        """
        Load AudioSet class labels. Since we don't have the CSV file,
        we'll create a mapping with some common sound categories for testing.
        
        In a production system, you'd load this from the official YAMNet class map CSV.
        """
        # Common sound categories for testing (partial AudioSet ontology)
        common_labels = {
            0: "Speech",
            1: "Male speech, man speaking",
            2: "Female speech, woman speaking", 
            3: "Child speech, kid speaking",
            10: "Music",
            137: "Telephone",
            310: "Fire alarm",
            320: "Smoke detector, smoke alarm",
            322: "Buzzer",
            324: "Alarm",
            325: "Siren",
            384: "Dog",
            385: "Bark",
            400: "Car",
            402: "Vehicle",
            420: "Door",
            421: "Doorbell",
            440: "Breaking",
            441: "Glass breaking",
            460: "Emergency vehicle",
            475: "Applause",
            480: "Laughter",
            485: "Crying",
            490: "Footsteps"
        }
        
        # Fill in remaining indices with generic labels
        labels = {}
        for i in range(self.num_classes):
            if i in common_labels:
                labels[i] = common_labels[i]
            else:
                labels[i] = f"AudioSet_Class_{i}"
        
        logger.info(f"Loaded {len(labels)} class labels")
        return labels
    
    def preprocess_audio(self, audio_file_path: str) -> np.ndarray:
        """
        Preprocess audio file to match YAMNet input requirements.
        
        The model expects input shape [1, 96, 64, 1] which appears to be a mel spectrogram.
        We need to convert the raw audio waveform into this format.
        
        Args:
            audio_file_path: Path to the audio file
            
        Returns:
            Preprocessed mel spectrogram as numpy array with shape [1, 96, 64, 1]
        """
        logger.info(f"Preprocessing audio file: {audio_file_path}")
        
        try:
            # Load audio file and resample to 16kHz
            waveform, original_sr = librosa.load(audio_file_path, sr=self.sample_rate, mono=True)
            
            logger.info(f"Original sample rate: {original_sr} Hz")
            logger.info(f"Audio duration: {len(waveform) / self.sample_rate:.2f} seconds")
            logger.info(f"Raw audio shape: {waveform.shape}")
            logger.info(f"Audio range: [{waveform.min():.3f}, {waveform.max():.3f}]")
            
            # Ensure audio is in the correct range [-1.0, +1.0]
            if np.abs(waveform).max() > 1.0:
                waveform = waveform / np.abs(waveform).max()
                logger.warning("Audio was normalized to [-1.0, +1.0] range")
            
            # Convert to mel spectrogram to match expected input shape [1, 96, 64, 1]
            # YAMNet typically uses 64 mel bins and specific time frames
            
            # Calculate mel spectrogram
            # Using hop_length and n_fft values typical for YAMNet
            hop_length = 512  # ~32ms at 16kHz
            n_fft = 2048      # ~128ms window
            n_mels = 64       # 64 mel bins as expected by model
            
            mel_spectrogram = librosa.feature.melspectrogram(
                y=waveform,
                sr=self.sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                fmin=0.0,
                fmax=self.sample_rate // 2
            )
            
            # Convert to log scale (common for audio models)
            log_mel = librosa.power_to_db(mel_spectrogram, ref=np.max)
            
            # Normalize to [0, 1] range
            log_mel_normalized = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min())
            
            logger.info(f"Mel spectrogram shape: {log_mel_normalized.shape}")
            
            # The model expects exactly 96 time frames
            # If we have more, we'll take chunks of 96 frames
            # If we have less, we'll pad with zeros
            
            target_time_frames = 96
            n_mels, time_frames = log_mel_normalized.shape
            
            if time_frames < target_time_frames:
                # Pad with zeros if too short
                padding = target_time_frames - time_frames
                log_mel_padded = np.pad(log_mel_normalized, ((0, 0), (0, padding)), mode='constant')
                logger.info(f"Padded spectrogram from {time_frames} to {target_time_frames} frames")
            else:
                # Take the first 96 frames if too long (we could also take multiple chunks)
                log_mel_padded = log_mel_normalized[:, :target_time_frames]
                if time_frames > target_time_frames:
                    logger.info(f"Truncated spectrogram from {time_frames} to {target_time_frames} frames")
            
            # Reshape to match expected input: [1, 96, 64, 1]
            # Current shape is (64, 96), we need (1, 96, 64, 1)
            log_mel_reshaped = log_mel_padded.T  # Transpose to (96, 64)
            log_mel_final = log_mel_reshaped[np.newaxis, :, :, np.newaxis]  # Add batch and channel dims
            
            logger.info(f"Final input shape: {log_mel_final.shape}")
            logger.info(f"Final input range: [{log_mel_final.min():.3f}, {log_mel_final.max():.3f}]")
            
            return log_mel_final.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error preprocessing audio file {audio_file_path}: {e}")
            raise
    
    def audio_generator(self):
        """
        Generator function that yields preprocessed mel spectrograms.
        This follows the MemryX pattern like the input functions in tutorials.
        
        The model expects input shape [1, 96, 64, 1] which is a mel spectrogram,
        not raw audio waveform.
        """
        try:
            if self.current_audio_file is None:
                logger.error("No audio file set for processing")
                return None
            
            # Preprocess the audio file to mel spectrogram
            mel_spectrogram = self.preprocess_audio(self.current_audio_file)
            
            logger.info(f"Yielding mel spectrogram of shape {mel_spectrogram.shape}")
            yield mel_spectrogram
            
        except Exception as e:
            logger.error(f"Error in audio generator: {e}")
            return None
    
    def process_model_output(self, *outputs):
        """
        Process YAMNet model outputs and log results.
        This is like the postprocess_and_show_frame() function from tutorials,
        but adapted for audio classification results.
        
        Note: The actual model output format may differ from the standard YAMNet
        description since this is a compiled version. We'll analyze what we receive.
        
        Args:
            *outputs: Model outputs from the compiled YAMNet model
        """
        try:
            logger.info(f"Received {len(outputs)} outputs from model")
            
            # Log the shape and type of each output to understand the format
            for i, output in enumerate(outputs):
                logger.info(f"Output {i}: shape={output.shape}, dtype={output.dtype}, "
                          f"range=[{output.min():.3f}, {output.max():.3f}]")
            
            if len(outputs) >= 1:
                # Assume the first output contains the classification scores
                scores = outputs[0]
                logger.info(f"Processing scores with shape: {scores.shape}")
                
                # Handle different possible output shapes
                if len(scores.shape) == 4:
                    # If output is 4D, flatten to get the classification scores
                    # This might be the case for models that output spatial features
                    scores_flat = scores.flatten()
                    logger.info(f"Flattened 4D output to shape: {scores_flat.shape}")
                    
                    # If we have exactly 521 values, these are likely the class scores
                    if len(scores_flat) == self.num_classes:
                        class_scores = scores_flat
                    else:
                        # Take the last 521 values or repeat pattern
                        if len(scores_flat) >= self.num_classes:
                            class_scores = scores_flat[-self.num_classes:]
                            logger.info(f"Extracted last {self.num_classes} values as class scores")
                        else:
                            logger.warning(f"Output too small ({len(scores_flat)}) for {self.num_classes} classes")
                            # Pad with zeros if needed
                            class_scores = np.zeros(self.num_classes)
                            class_scores[:len(scores_flat)] = scores_flat
                
                elif len(scores.shape) == 2:
                    # If 2D, might be (batch_size, num_classes) or (time_frames, num_classes)
                    if scores.shape[1] == self.num_classes:
                        # Take mean across time frames or batch
                        class_scores = np.mean(scores, axis=0)
                        logger.info(f"Averaged across dimension 0, final shape: {class_scores.shape}")
                    elif scores.shape[0] == self.num_classes:
                        # Take mean across the other dimension
                        class_scores = np.mean(scores, axis=1)
                        logger.info(f"Averaged across dimension 1, final shape: {class_scores.shape}")
                    else:
                        logger.warning(f"Unexpected 2D shape: {scores.shape}")
                        class_scores = scores.flatten()[:self.num_classes]
                
                elif len(scores.shape) == 1:
                    # 1D output - use directly if it matches expected size
                    if len(scores) == self.num_classes:
                        class_scores = scores
                    else:
                        logger.warning(f"1D output size ({len(scores)}) doesn't match expected classes ({self.num_classes})")
                        class_scores = np.zeros(self.num_classes)
                        min_len = min(len(scores), self.num_classes)
                        class_scores[:min_len] = scores[:min_len]
                
                else:
                    logger.error(f"Unexpected output shape: {scores.shape}")
                    return
                
                # Ensure we have valid class scores
                if len(class_scores) != self.num_classes:
                    logger.error(f"Class scores length ({len(class_scores)}) doesn't match expected ({self.num_classes})")
                    return
                
                logger.info(f"Final class scores shape: {class_scores.shape}")
                logger.info(f"Class scores range: [{class_scores.min():.3f}, {class_scores.max():.3f}]")
                
                # Get top predictions
                top_indices = np.argsort(class_scores)[-10:][::-1]  # Top 10 predictions
                top_scores = class_scores[top_indices]
                
                predictions = []
                for idx, score in zip(top_indices, top_scores):
                    prediction = {
                        'class_index': int(idx),
                        'class_name': self.class_labels.get(idx, f"Unknown_{idx}"),
                        'raw_score': float(score),
                        'confidence_percent': float(score * 100)  # Convert to percentage
                    }
                    predictions.append(prediction)
                
                # Store results
                result = {
                    'audio_file': str(self.current_audio_file),
                    'timestamp': datetime.now().isoformat(),
                    'model_output_shapes': [list(output.shape) for output in outputs],
                    'predictions': predictions,
                    'raw_scores_stats': {
                        'min': float(class_scores.min()),
                        'max': float(class_scores.max()),
                        'mean': float(class_scores.mean()),
                        'std': float(class_scores.std())
                    }
                }
                
                self.results.append(result)
                
                # Log top predictions
                logger.info(f"\n=== TOP PREDICTIONS FOR {Path(self.current_audio_file).name} ===")
                for i, pred in enumerate(predictions, 1):
                    logger.info(f"{i:2d}. {pred['class_name']:<30} {pred['confidence_percent']:6.2f}% (raw: {pred['raw_score']:.4f})")
                
                # Check for important sounds
                self._check_important_sounds(predictions)
                
            else:
                logger.error("No outputs received from model")
                
        except Exception as e:
            logger.error(f"Error processing model output: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _check_important_sounds(self, predictions: List[Dict]):
        """
        Check if any important/emergency sounds were detected.
        This simulates the notification logic that would be in notification_handler.py.
        """
        # Define important sound categories and their thresholds
        important_sounds = {
            'fire alarm': ['fire alarm', 'smoke detector', 'alarm', 'buzzer'],
            'emergency': ['siren', 'emergency vehicle'],
            'security': ['breaking', 'glass breaking'],
            'communication': ['telephone', 'doorbell']
        }
        
        threshold = 0.1  # Minimum raw score for alert (adjusted for raw model outputs)
        
        alerts = []
        for pred in predictions:
            class_name_lower = pred['class_name'].lower()
            confidence = pred['raw_score']  # Use raw score instead of percentage
            
            if confidence >= threshold:
                for category, keywords in important_sounds.items():
                    if any(keyword in class_name_lower for keyword in keywords):
                        alert = {
                            'category': category,
                            'detected_sound': pred['class_name'],
                            'raw_score': confidence,
                            'confidence_percent': pred['confidence_percent'],
                            'urgency': 'HIGH' if category in ['fire alarm', 'emergency'] else 'MEDIUM'
                        }
                        alerts.append(alert)
                        break
        
        if alerts:
            logger.warning(f"\n🚨 IMPORTANT SOUNDS DETECTED:")
            for alert in alerts:
                logger.warning(f"   {alert['urgency']} - {alert['category'].upper()}: "
                             f"{alert['detected_sound']} (raw: {alert['raw_score']:.3f}, "
                             f"{alert['confidence_percent']:.1f}%)")
        else:
            logger.info("No critical sounds detected above threshold")
    
    def test_audio_file(self, audio_file_path: str):
        """
        Test the YAMNet model with a single audio file.
        
        Args:
            audio_file_path: Path to the audio file to test
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"TESTING AUDIO FILE: {audio_file_path}")
        logger.info(f"{'='*60}")
        
        if not os.path.exists(audio_file_path):
            logger.error(f"Audio file not found: {audio_file_path}")
            return False
        
        if not MODEL_PATH.exists():
            logger.error(f"Model file not found: {MODEL_PATH}")
            return False
        
        self.current_audio_file = audio_file_path
        
        try:
            # Initialize the MemryX accelerator
            logger.info("Initializing MemryX accelerator...")
            self.accelerator = AsyncAccl(str(MODEL_PATH), local_mode=False)
            
            # Connect input and output functions (following AsyncAccl pattern)
            logger.info("Connecting input and output functions...")
            self.accelerator.connect_input(self.audio_generator)
            self.accelerator.connect_output(self.process_model_output)
            
            # Start inference
            logger.info("Starting inference...")
            self.accelerator.wait()
            
            logger.info("✅ Audio file processed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error testing audio file: {e}")
            return False
        
        finally:
            # Clean up
            if self.accelerator:
                try:
                    self.accelerator.stop()
                except:
                    pass
                self.accelerator = None
    
    def test_all_samples(self):
        """Test all audio files in the samples directory."""
        audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
        
        sample_files = []
        for ext in audio_extensions:
            sample_files.extend(SAMPLES_DIR.glob(f"*{ext}"))
        
        if not sample_files:
            logger.warning(f"No audio files found in {SAMPLES_DIR}")
            logger.info("Please add some test audio files to the samples directory")
            return
        
        logger.info(f"Found {len(sample_files)} audio files to test")
        
        successful_tests = 0
        for audio_file in sample_files:
            success = self.test_audio_file(str(audio_file))
            if success:
                successful_tests += 1
        
        logger.info(f"\n{'='*60}")
        logger.info(f"TESTING COMPLETE: {successful_tests}/{len(sample_files)} files processed successfully")
        logger.info(f"{'='*60}")
    
    def save_results(self):
        """Save test results to JSON file."""
        if not self.results:
            logger.warning("No results to save")
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = RESULTS_DIR / f"yamnet_test_results_{timestamp}.json"
        
        try:
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            logger.info(f"Results saved to: {results_file}")
            
            # Also save a summary CSV
            self._save_summary_csv(timestamp)
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def _save_summary_csv(self, timestamp: str):
        """Save a summary of results in CSV format."""
        summary_file = RESULTS_DIR / f"yamnet_summary_{timestamp}.csv"
        
        try:
            with open(summary_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Audio File', 'Top Prediction', 'Confidence %', 'Second Prediction', 'Confidence %'])
                
                for result in self.results:
                    audio_file = Path(result['audio_file']).name
                    predictions = result['predictions']
                    
                    if len(predictions) >= 2:
                        row = [
                            audio_file,
                            predictions[0]['class_name'],
                            f"{predictions[0]['confidence_percent']:.2f}",
                            predictions[1]['class_name'], 
                            f"{predictions[1]['confidence_percent']:.2f}"
                        ]
                    elif len(predictions) >= 1:
                        row = [
                            audio_file,
                            predictions[0]['class_name'],
                            f"{predictions[0]['confidence_percent']:.2f}",
                            "N/A",
                            "N/A"
                        ]
                    else:
                        row = [audio_file, "No predictions", "0.00", "N/A", "N/A"]
                    
                    writer.writerow(row)
            
            logger.info(f"Summary saved to: {summary_file}")
            
        except Exception as e:
            logger.error(f"Error saving summary CSV: {e}")


def main():
    """Main function to run the YAMNet test."""
    print(f"""
    🎵 YAMNet Audio Classification Test
    {'='*50}
    
    This script tests the YAMNet model using MemryX accelerator.
    
    Directory structure:
    📁 test/
    ├── 📁 samples/     <- Place your test audio files here
    ├── 📁 logs/        <- Test logs will be saved here  
    ├── 📁 results/     <- Test results will be saved here
    └── 📄 test_yamnet.py <- This script
    
    Supported audio formats: WAV, MP3, FLAC, OGG, M4A, AAC
    """)
    
    # Initialize tester
    tester = YAMNetTester()
    
    try:
        # Check if specific audio file was provided as command line argument
        if len(sys.argv) > 1:
            audio_file_path = sys.argv[1]
            if not os.path.exists(audio_file_path):
                logger.error(f"Audio file not found: {audio_file_path}")
                sys.exit(1)
            
            logger.info(f"Testing single audio file: {audio_file_path}")
            success = tester.test_audio_file(audio_file_path)
            
            if success:
                tester.save_results()
            else:
                sys.exit(1)
        else:
            # Test all audio files in samples directory
            logger.info("Testing all audio files in samples directory...")
            tester.test_all_samples()
            tester.save_results()
        
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()