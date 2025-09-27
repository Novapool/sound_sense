#!/usr/bin/env python3
"""
Improved YAMNet test script with proper AudioSet class mapping and model output handling.
This script addresses all the key issues identified:
1. Loads complete AudioSet class mapping from CSV
2. Handles YAMNet's 3-tuple output format correctly (scores, embeddings, log_mel_spectrogram)
3. Processes raw audio waveforms (not mel spectrograms)
4. Implements proper score interpretation (not treating as probabilities)
5. Uses correct AudioSet indices for fire alarm detection
6. Implements multi-frame score aggregation

Usage:
    python test_yamnet_improved.py [audio_file_path]
"""

import os
import sys
import json
import numpy as np
import librosa
import logging
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import pandas as pd

# MemryX imports
from memryx import AsyncAccl

# Import audio processing logic from our audio_processor
sys.path.append(str(Path(__file__).parent.parent / "src"))
from audio_processor import AudioConfig

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
TEST_DIR = Path(__file__).parent
SAMPLES_DIR = TEST_DIR / "samples"
LOGS_DIR = TEST_DIR / "logs" 
RESULTS_DIR = TEST_DIR / "results"
MODEL_PATH = PROJECT_ROOT / "models" / "Audio_classification_YamNet_96_64_1_tflite.dfp"
CLASS_MAP_PATH = TEST_DIR / "yamnet_class_map.csv"

# Create directories if they don't exist
for directory in [SAMPLES_DIR, LOGS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / f"yamnet_improved_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class YAMNetImprovedTester:
    """
    Improved YAMNet audio classification tester with proper AudioSet mapping.
    
    This version correctly handles:
    - Complete AudioSet class mapping from CSV
    - YAMNet's 3-tuple output format (scores, embeddings, log_mel_spectrogram)
    - Raw audio waveform processing
    - Non-probability score interpretation
    - Multi-frame score aggregation
    - Proper fire alarm detection using correct AudioSet indices
    """
    
    def __init__(self):
        """Initialize the improved YAMNet tester."""
        # Use the same audio configuration as audio_processor
        self.config = AudioConfig()
        
        # Model specifications
        self.num_classes = 521  # YAMNet predicts 521 AudioSet classes
        
        # Load complete AudioSet class labels from CSV
        self.class_labels = self._load_audioset_labels_from_csv()
        
        # Define critical sound categories with proper AudioSet indices
        self.critical_sound_indices = self._define_critical_sounds()
        
        # Results storage
        self.current_audio_file = None
        self.results = []
        self.accelerator = None
        self.current_windows = []  # Store windows for current file
        self.window_results = []   # Store per-window results
        
        logger.info(f"Improved YAMNet Tester initialized")
        logger.info(f"Sample rate: {self.config.SAMPLE_RATE} Hz")
        logger.info(f"Window length: {self.config.WINDOW_LENGTH_SECONDS}s ({self.config.WINDOW_LENGTH_SAMPLES} samples)")
        logger.info(f"Hop length: {self.config.HOP_LENGTH_SECONDS}s ({self.config.HOP_LENGTH_SAMPLES} samples)")
        logger.info(f"Loaded {len(self.class_labels)} AudioSet classes")
    
    def _load_audioset_labels_from_csv(self) -> Dict[int, str]:
        """
        Load complete AudioSet class labels from the CSV file.
        
        Returns:
            Dictionary mapping class indices to display names
        """
        if not CLASS_MAP_PATH.exists():
            logger.error(f"AudioSet class map CSV not found: {CLASS_MAP_PATH}")
            raise FileNotFoundError(f"Please ensure {CLASS_MAP_PATH} exists")
        
        class_labels = {}
        
        try:
            with open(CLASS_MAP_PATH, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    index = int(row['index'])
                    display_name = row['display_name']
                    class_labels[index] = display_name
            
            logger.info(f"Successfully loaded {len(class_labels)} AudioSet class labels from CSV")
            
            # Log some key fire alarm related classes
            fire_alarm_classes = [(k, v) for k, v in class_labels.items() 
                                 if any(term in v.lower() for term in ['fire alarm', 'smoke detector', 'alarm', 'siren', 'buzzer'])]
            logger.info("Key alarm/emergency classes found:")
            for idx, name in fire_alarm_classes:
                logger.info(f"  {idx}: {name}")
            
            return class_labels
            
        except Exception as e:
            logger.error(f"Error loading AudioSet class labels from CSV: {e}")
            raise
    
    def _define_critical_sounds(self) -> Dict[str, Dict]:
        """
        Define critical sound categories with their AudioSet indices and detection parameters.
        
        Returns:
            Dictionary of critical sound categories with detection configuration
        """
        critical_sounds = {
            'fire_alarm': {
                'indices': [394],  # Fire alarm
                'keywords': ['fire alarm'],
                'threshold': 0.1,  # Raw score threshold (not probability)
                'urgency': 'CRITICAL',
                'description': 'Fire alarm detection'
            },
            'smoke_detector': {
                'indices': [393],  # Smoke detector, smoke alarm
                'keywords': ['smoke detector', 'smoke alarm'],
                'threshold': 0.1,
                'urgency': 'CRITICAL',
                'description': 'Smoke detector alarm'
            },
            'general_alarm': {
                'indices': [382, 389],  # Alarm, Alarm clock
                'keywords': ['alarm'],
                'threshold': 0.15,
                'urgency': 'HIGH',
                'description': 'General alarm sounds'
            },
            'emergency_siren': {
                'indices': [390, 391, 317, 318, 319],  # Siren, Civil defense siren, Police car, Ambulance, Fire engine
                'keywords': ['siren', 'emergency vehicle', 'police car', 'ambulance', 'fire engine'],
                'threshold': 0.12,
                'urgency': 'HIGH',
                'description': 'Emergency vehicle sirens'
            },
            'buzzer': {
                'indices': [392],  # Buzzer
                'keywords': ['buzzer'],
                'threshold': 0.15,
                'urgency': 'MEDIUM',
                'description': 'Buzzer sounds'
            },
            'security_alarm': {
                'indices': [304],  # Car alarm
                'keywords': ['car alarm'],
                'threshold': 0.12,
                'urgency': 'MEDIUM',
                'description': 'Security alarms'
            }
        }
        
        logger.info("Critical sound categories defined:")
        for category, config in critical_sounds.items():
            indices_names = [f"{idx}:{self.class_labels.get(idx, 'Unknown')}" for idx in config['indices']]
            logger.info(f"  {category}: {config['description']} - Indices: {indices_names}")
        
        return critical_sounds
    
    def preprocess_audio_file(self, audio_file_path: str) -> List[np.ndarray]:
        """
        Preprocess audio file into raw audio waveforms for YAMNet.
        
        YAMNet expects raw audio waveforms, not mel spectrograms.
        The model performs its own internal feature extraction.
        
        Args:
            audio_file_path: Path to the audio file
            
        Returns:
            List of preprocessed audio windows, each shaped (1, window_length_samples)
        """
        logger.info(f"Preprocessing audio file: {audio_file_path}")
        
        try:
            # Load audio file and resample to 16kHz (YAMNet requirement)
            waveform, original_sr = librosa.load(audio_file_path, sr=self.config.SAMPLE_RATE, mono=True)
            
            logger.info(f"Original sample rate: {original_sr} Hz")
            logger.info(f"Audio duration: {len(waveform) / self.config.SAMPLE_RATE:.2f} seconds")
            logger.info(f"Raw audio shape: {waveform.shape}")
            logger.info(f"Audio range: [{waveform.min():.3f}, {waveform.max():.3f}]")
            
            # Ensure audio is in the correct range [-1.0, +1.0] as required by YAMNet
            waveform = np.clip(waveform, -1.0, 1.0)
            
            # Create sliding windows using YAMNet parameters
            windows = []
            window_length = self.config.WINDOW_LENGTH_SAMPLES
            hop_length = self.config.HOP_LENGTH_SAMPLES
            
            # Calculate number of windows
            if len(waveform) <= window_length:
                # Short audio: pad and create single window
                padded_waveform = np.pad(waveform, (0, window_length - len(waveform)), mode='constant')
                window = padded_waveform.reshape(1, -1).astype(np.float32)
                windows.append(window)
                logger.info(f"Short audio: created 1 window with padding")
            else:
                # Long audio: create sliding windows
                start = 0
                while start + window_length <= len(waveform):
                    window_data = waveform[start:start + window_length]
                    window = window_data.reshape(1, -1).astype(np.float32)
                    windows.append(window)
                    start += hop_length
                
                # Handle remaining audio if any
                if start < len(waveform):
                    remaining = waveform[start:]
                    padded_remaining = np.pad(remaining, (0, window_length - len(remaining)), mode='constant')
                    window = padded_remaining.reshape(1, -1).astype(np.float32)
                    windows.append(window)
                
                logger.info(f"Long audio: created {len(windows)} sliding windows")
            
            # Log window details
            for i, window in enumerate(windows[:3]):  # Log first 3 windows
                logger.info(f"Window {i}: shape={window.shape}, range=[{window.min():.3f}, {window.max():.3f}]")
            
            if len(windows) > 3:
                logger.info(f"... and {len(windows) - 3} more windows")
            
            return windows
            
        except Exception as e:
            logger.error(f"Error preprocessing audio file {audio_file_path}: {e}")
            raise
    
    def audio_generator(self):
        """
        Generator function that yields preprocessed raw audio windows.
        This follows the MemryX AsyncAccl pattern.
        """
        try:
            if not self.current_windows:
                logger.error("No audio windows available for processing")
                return
            
            logger.info(f"Yielding {len(self.current_windows)} raw audio windows")
            
            for i, window in enumerate(self.current_windows):
                logger.debug(f"Yielding window {i+1}/{len(self.current_windows)}: shape={window.shape}")
                yield window
                
        except Exception as e:
            logger.error(f"Error in audio generator: {e}")
            return
    
    def process_model_output(self, *outputs):
        """
        Process YAMNet model outputs correctly.
        
        YAMNet returns a 3-tuple: (scores, embeddings, log_mel_spectrogram)
        - scores: Shape (N, 521) - Classification scores for each AudioSet class
        - embeddings: Shape (N, 1024) - Feature embeddings
        - log_mel_spectrogram: Shape (num_frames, 64) - Log mel spectrogram
        
        We want the scores for classification.
        
        Args:
            *outputs: Model outputs from YAMNet
        """
        try:
            logger.info(f"Received {len(outputs)} outputs from model")
            
            # Log the shape and type of each output
            for i, output in enumerate(outputs):
                logger.info(f"Output {i}: shape={output.shape}, dtype={output.dtype}, "
                          f"range=[{output.min():.3f}, {output.max():.3f}]")
            
            if len(outputs) < 1:
                logger.error("Expected at least 1 output from YAMNet model")
                return
            
            # YAMNet should return 3 outputs, but compiled models might vary
            # We need to identify which output contains the classification scores (shape should be N, 521)
            scores_output = None
            embeddings_output = None
            
            for i, output in enumerate(outputs):
                if len(output.shape) == 2:
                    if output.shape[1] == self.num_classes:  # (N, 521)
                        scores_output = output
                        logger.info(f"Found classification scores in output {i}: shape={output.shape}")
                    elif output.shape[1] == 1024:  # (N, 1024)
                        embeddings_output = output
                        logger.info(f"Found embeddings in output {i}: shape={output.shape}")
            
            # If we didn't find the expected format, try to work with what we have
            if scores_output is None:
                logger.warning("Could not identify classification scores output, using first output")
                scores_output = outputs[0]
                
                # Try to reshape or extract scores
                if len(scores_output.shape) == 4:
                    # Flatten 4D output
                    scores_flat = scores_output.flatten()
                    if len(scores_flat) >= self.num_classes:
                        # Take the last 521 values or reshape appropriately
                        scores_output = scores_flat[-self.num_classes:].reshape(1, -1)
                        logger.info(f"Reshaped 4D output to classification scores: {scores_output.shape}")
                elif len(scores_output.shape) == 1:
                    if len(scores_output) == self.num_classes:
                        scores_output = scores_output.reshape(1, -1)
                    else:
                        logger.error(f"1D output size ({len(scores_output)}) doesn't match expected classes ({self.num_classes})")
                        return
            
            # Extract classification scores
            if len(scores_output.shape) == 2 and scores_output.shape[1] == self.num_classes:
                # Perfect: (N, 521) shape - take mean across frames if multiple frames
                if scores_output.shape[0] > 1:
                    class_scores = np.mean(scores_output, axis=0)  # Average across frames
                    logger.info(f"Averaged scores across {scores_output.shape[0]} frames")
                else:
                    class_scores = scores_output[0]  # Single frame
            else:
                logger.error(f"Unexpected classification scores shape: {scores_output.shape}")
                return
            
            logger.info(f"Final classification scores shape: {class_scores.shape}")
            logger.info(f"Classification scores range: [{class_scores.min():.3f}, {class_scores.max():.3f}]")
            
            # Important: YAMNet scores are NOT calibrated probabilities
            # We work with raw scores and use relative ranking
            
            # Get top predictions based on raw scores
            top_indices = np.argsort(class_scores)[-20:][::-1]  # Top 20 predictions
            top_scores = class_scores[top_indices]
            
            predictions = []
            for idx, score in zip(top_indices, top_scores):
                prediction = {
                    'class_index': int(idx),
                    'class_name': self.class_labels.get(idx, f"Unknown_{idx}"),
                    'raw_score': float(score),
                    'relative_confidence': float(score / np.max(class_scores) * 100)  # Relative to max score
                }
                predictions.append(prediction)
            
            # Store per-window result
            window_result = {
                'window_index': len(self.window_results),
                'predictions': predictions,
                'raw_scores_stats': {
                    'min': float(class_scores.min()),
                    'max': float(class_scores.max()),
                    'mean': float(class_scores.mean()),
                    'std': float(class_scores.std())
                }
            }
            
            self.window_results.append(window_result)
            
            # Log top predictions for this window
            logger.info(f"\n=== WINDOW {len(self.window_results)} PREDICTIONS ===")
            for i, pred in enumerate(predictions[:10], 1):  # Show top 10
                logger.info(f"{i:2d}. {pred['class_name']:<35} "
                          f"Raw: {pred['raw_score']:7.3f} "
                          f"Rel: {pred['relative_confidence']:5.1f}%")
            
            # Check for critical sounds in this window
            self._check_critical_sounds(predictions, len(self.window_results))
                
        except Exception as e:
            logger.error(f"Error processing model output: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _check_critical_sounds(self, predictions: List[Dict], window_index: int):
        """
        Check if any critical/emergency sounds were detected in this window.
        
        Uses the proper AudioSet indices and raw score thresholds.
        """
        alerts = []
        
        for pred in predictions:
            class_idx = pred['class_index']
            class_name = pred['class_name']
            raw_score = pred['raw_score']
            
            # Check against each critical sound category
            for category, config in self.critical_sound_indices.items():
                if class_idx in config['indices']:
                    if raw_score >= config['threshold']:
                        alert = {
                            'window_index': window_index,
                            'category': category,
                            'detected_sound': class_name,
                            'class_index': class_idx,
                            'raw_score': raw_score,
                            'relative_confidence': pred['relative_confidence'],
                            'urgency': config['urgency'],
                            'threshold_used': config['threshold'],
                            'description': config['description']
                        }
                        alerts.append(alert)
                        break  # Only match first category
        
        if alerts:
            logger.warning(f"\n🚨 CRITICAL SOUNDS DETECTED IN WINDOW {window_index}:")
            for alert in alerts:
                logger.warning(f"   {alert['urgency']} - {alert['category'].upper()}: "
                             f"{alert['detected_sound']} "
                             f"(raw: {alert['raw_score']:.3f}, rel: {alert['relative_confidence']:.1f}%)")
        else:
            logger.debug(f"No critical sounds detected in window {window_index}")
    
    def _aggregate_results(self) -> Dict:
        """
        Aggregate results across all windows for the current audio file.
        
        This provides overall predictions by combining window-level results.
        """
        if not self.window_results:
            return {}
        
        logger.info(f"Aggregating results from {len(self.window_results)} windows")
        
        # Collect all scores across windows for each class
        all_class_scores = {}
        all_alerts = []
        
        for window_result in self.window_results:
            # Aggregate scores for each class
            for pred in window_result['predictions']:
                class_idx = pred['class_index']
                score = pred['raw_score']
                
                if class_idx not in all_class_scores:
                    all_class_scores[class_idx] = []
                all_class_scores[class_idx].append(score)
        
        # Calculate aggregated statistics for each class
        aggregated_predictions = []
        for class_idx, scores in all_class_scores.items():
            class_name = self.class_labels.get(class_idx, f"Unknown_{class_idx}")
            
            aggregated_pred = {
                'class_index': class_idx,
                'class_name': class_name,
                'mean_score': float(np.mean(scores)),
                'max_score': float(np.max(scores)),
                'min_score': float(np.min(scores)),
                'std_score': float(np.std(scores)),
                'detection_count': len(scores),
                'detection_rate': len(scores) / len(self.window_results)
            }
            aggregated_predictions.append(aggregated_pred)
        
        # Sort by mean score
        aggregated_predictions.sort(key=lambda x: x['mean_score'], reverse=True)
        
        # Get top aggregated predictions
        top_aggregated = aggregated_predictions[:15]
        
        # Check for critical sounds in aggregated results
        aggregated_alerts = self._check_aggregated_critical_sounds(top_aggregated)
        
        return {
            'total_windows': len(self.window_results),
            'top_predictions': top_aggregated,
            'alerts': aggregated_alerts,
            'per_window_results': self.window_results
        }
    
    def _check_aggregated_critical_sounds(self, predictions: List[Dict]) -> List[Dict]:
        """Check for critical sounds in aggregated results."""
        alerts = []
        
        for pred in predictions:
            class_idx = pred['class_index']
            class_name = pred['class_name']
            mean_score = pred['mean_score']
            max_score = pred['max_score']
            detection_rate = pred['detection_rate']
            
            # Check against each critical sound category
            for category, config in self.critical_sound_indices.items():
                if class_idx in config['indices']:
                    # Use lower threshold for aggregated results, but require some consistency
                    threshold = config['threshold'] * 0.7  # Lower threshold for aggregated
                    
                    if (mean_score >= threshold and detection_rate >= 0.1) or \
                       (max_score >= config['threshold'] and detection_rate >= 0.2):
                        
                        confidence_level = 'HIGH' if detection_rate >= 0.5 else \
                                         'MEDIUM' if detection_rate >= 0.3 else 'LOW'
                        
                        alert = {
                            'category': category,
                            'detected_sound': class_name,
                            'class_index': class_idx,
                            'mean_score': mean_score,
                            'max_score': max_score,
                            'detection_rate': detection_rate,
                            'detection_count': pred['detection_count'],
                            'urgency': config['urgency'],
                            'confidence_level': confidence_level,
                            'description': config['description']
                        }
                        alerts.append(alert)
                        break
        
        return alerts
    
    def test_audio_file(self, audio_file_path: str):
        """
        Test the YAMNet model with a single audio file using improved processing.
        
        Args:
            audio_file_path: Path to the audio file to test
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"TESTING AUDIO FILE (IMPROVED): {audio_file_path}")
        logger.info(f"{'='*70}")
        
        if not os.path.exists(audio_file_path):
            logger.error(f"Audio file not found: {audio_file_path}")
            return False
        
        if not MODEL_PATH.exists():
            logger.error(f"Model file not found: {MODEL_PATH}")
            return False
        
        self.current_audio_file = audio_file_path
        self.window_results = []  # Reset for new file
        
        try:
            # Preprocess audio file into raw audio windows
            logger.info("Preprocessing audio file into raw audio windows...")
            self.current_windows = self.preprocess_audio_file(audio_file_path)
            
            if not self.current_windows:
                logger.error("No audio windows generated")
                return False
            
            # Initialize the MemryX accelerator
            logger.info("Initializing MemryX accelerator...")
            self.accelerator = AsyncAccl(str(MODEL_PATH), local_mode=False)
            
            # Connect input and output functions
            logger.info("Connecting input and output functions...")
            self.accelerator.connect_input(self.audio_generator)
            self.accelerator.connect_output(self.process_model_output)
            
            # Start inference
            logger.info("Starting inference...")
            self.accelerator.wait()
            
            # Aggregate results
            logger.info("Aggregating results across windows...")
            aggregated_result = self._aggregate_results()
            
            # Store final result
            final_result = {
                'audio_file': str(self.current_audio_file),
                'timestamp': datetime.now().isoformat(),
                'model_info': 'YAMNet with proper AudioSet class mapping and raw audio processing',
                'processing_info': 'Raw audio waveforms processed with sliding windows, scores aggregated across frames',
                'audio_duration_seconds': len(self.current_windows) * self.config.HOP_LENGTH_SECONDS,
                'total_windows_processed': len(self.window_results),
                'aggregated_results': aggregated_result
            }
            
            self.results.append(final_result)
            
            # Log final aggregated results
            self._log_final_results(aggregated_result)
            
            logger.info("✅ Audio file processed successfully with improved implementation!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error testing audio file: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
        
        finally:
            # Clean up
            if self.accelerator:
                try:
                    self.accelerator.stop()
                except:
                    pass
                self.accelerator = None
            self.current_windows = []
    
    def _log_final_results(self, aggregated_result: Dict):
        """Log the final aggregated results in a clear format."""
        logger.info(f"\n{'='*70}")
        logger.info(f"FINAL AGGREGATED RESULTS")
        logger.info(f"{'='*70}")
        
        if 'top_predictions' in aggregated_result:
            logger.info(f"\nTOP AGGREGATED PREDICTIONS:")
            for i, pred in enumerate(aggregated_result['top_predictions'][:12], 1):
                logger.info(f"{i:2d}. {pred['class_name']:<35} "
                          f"Mean: {pred['mean_score']:6.3f} "
                          f"Max: {pred['max_score']:6.3f} "
                          f"Rate: {pred['detection_rate']*100:4.1f}% "
                          f"({pred['detection_count']}/{aggregated_result['total_windows']} windows)")
        
        if 'alerts' in aggregated_result and aggregated_result['alerts']:
            logger.warning(f"\n🚨 CRITICAL SOUNDS DETECTED (AGGREGATED):")
            for alert in aggregated_result['alerts']:
                logger.warning(f"   {alert['urgency']} - {alert['category'].upper()}: "
                             f"{alert['detected_sound']}")
                logger.warning(f"      {alert['description']}")
                logger.warning(f"      Mean score: {alert['mean_score']:.3f}, "
                             f"Max: {alert['max_score']:.3f}, "
                             f"Detection rate: {alert['detection_rate']*100:.1f}% "
                             f"({alert['detection_count']}/{aggregated_result['total_windows']} windows)")
                logger.warning(f"      Confidence level: {alert['confidence_level']}")
        else:
            logger.info("\nNo critical sounds detected in aggregated results")
    
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
        
        logger.info(f"\n{'='*70}")
        logger.info(f"TESTING COMPLETE: {successful_tests}/{len(sample_files)} files processed successfully")
        logger.info(f"{'='*70}")
    
    def save_results(self):
        """Save test results to JSON file."""
        if not self.results:
            logger.warning("No results to save")
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = RESULTS_DIR / f"yamnet_improved_results_{timestamp}.json"
        
        try:
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            logger.info(f"Improved results saved to: {results_file}")
            
            # Also save a summary CSV
            self._save_summary_csv(timestamp)
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def _save_summary_csv(self, timestamp: str):
        """Save a summary of results in CSV format."""
        summary_file = RESULTS_DIR / f"yamnet_improved_summary_{timestamp}.csv"
        
        try:
            with open(summary_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Audio File', 'Duration (s)', 'Windows Processed', 
                    'Top Prediction', 'Mean Score', 'Detection Rate %',
                    'Second Prediction', 'Mean Score', 'Detection Rate %',
                    'Critical Alerts', 'Alert Details'
                ])
                
                for result in self.results:
                    audio_file = Path(result['audio_file']).name
                    duration = result.get('audio_duration_seconds', 0)
                    windows = result.get('total_windows_processed', 0)
                    
                    aggregated = result.get('aggregated_results', {})
                    predictions = aggregated.get('top_predictions', [])
                    alerts = aggregated.get('alerts', [])
                    
                    # Get top 2 predictions
                    if len(predictions) >= 2:
                        pred1 = predictions[0]
                        pred2 = predictions[1]
                        row = [
                            audio_file, f"{duration:.1f}", windows,
                            pred1['class_name'], f"{pred1['mean_score']:.3f}", f"{pred1['detection_rate']*100:.1f}",
                            pred2['class_name'], f"{pred2['mean_score']:.3f}", f"{pred2['detection_rate']*100:.1f}",
                            f"{len(alerts)} alerts" if alerts else "None",
                            "; ".join([f"{a['category']}:{a['urgency']}" for a in alerts[:3]]) if alerts else "None"
                        ]
                    elif len(predictions) >= 1:
                        pred1 = predictions[0]
                        row = [
                            audio_file, f"{duration:.1f}", windows,
                            pred1['class_name'], f"{pred1['mean_score']:.3f}", f"{pred1['detection_rate']*100:.1f}",
                            "N/A", "N/A", "N/A",
                            f"{len(alerts)} alerts" if alerts else "None",
                            "; ".join([f"{a['category']}:{a['urgency']}" for a in alerts[:3]]) if alerts else "None"
                        ]
                    else:
                        row = [audio_file, f"{duration:.1f}", windows, "No predictions", "0.000", "0.0", "N/A", "N/A", "N/A", "None", "None"]
                    
                    writer.writerow(row)
            
            logger.info(f"Improved summary saved to: {summary_file}")
            
        except Exception as e:
            logger.error(f"Error saving summary CSV: {e}")


def main():
    """Main function to run the improved YAMNet test."""
    print(f"""
    🎵 YAMNet Audio Classification Test (IMPROVED VERSION)
    {'='*70}
    
    This is the IMPROVED version with all fixes:
    ✅ Complete AudioSet class mapping from CSV (521 classes)
    ✅ Proper YAMNet 3-tuple output handling (scores, embeddings, log_mel_spectrogram)
    ✅ Raw audio waveform processing (not mel spectrograms)
    ✅ Non-probability score interpretation with relative confidence
    ✅ Multi-frame score aggregation across sliding windows
    ✅ Correct AudioSet indices for fire alarm detection:
        - Index 394: Fire alarm
        - Index 393: Smoke detector, smoke alarm
        - Index 382: Alarm
        - Index 390: Siren
        - Index 392: Buzzer
    
    Directory structure:
    📁 test/
    ├── 📁 samples/           <- Place your test audio files here
    ├── 📁 logs/              <- Test logs will be saved here  
    ├── 📁 results/           <- Test results will be saved here
    ├── 📄 yamnet_class_map.csv <- AudioSet class mapping (521 classes)
    └── 📄 test_yamnet_improved.py <- This improved script
    
    Supported audio formats: WAV, MP3, FLAC, OGG, M4A, AAC
    """)
    
    # Initialize improved tester
    tester = YAMNetImprovedTester()
    
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
