#!/usr/bin/env python3
"""
Improved Real-Time Audio Classification using AsyncAccl
Further fixes to address callback timing and data flow issues
"""

import numpy as np
import sounddevice as sd
import csv
import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter
from memryx import AsyncAccl
from collections import Counter, deque
import time
import threading
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImprovedRealTimeAudioClassifier:
    """
    Improved real-time audio classifier using AsyncAccl.
    Addresses callback timing and data flow issues.
    """

    def __init__(self, class_map_csv_text, preprocess_model_path, model_dfp_path, postprocess_model_path):
        self.class_map_csv_text = class_map_csv_text
        self.preprocess_model_path = preprocess_model_path
        self.model_dfp_path = model_dfp_path
        self.postprocess_model_path = postprocess_model_path

        # YAMNet parameters (same as working classifier)
        self.num_samples_per_frame = 15600
        self.hop_size = 7800
        self.sample_rate = 16000
        
        # Load class labels
        self.labels = self._class_names_from_csv()

        # Initialize interpreters for manual preprocessing/postprocessing
        self._pre_interpreter = Interpreter(self.preprocess_model_path)
        self._post_interpreter = Interpreter(self.postprocess_model_path)

        # Audio capture setup
        self.audio_buffer = deque(maxlen=self.sample_rate * 10)  # 10 second buffer
        self.is_running = False
        self.audio_stream = None
        self.buffer_lock = threading.Lock()
        
        # Results tracking
        self.current_prediction = "Listening..."
        self.prediction_history = deque(maxlen=20)
        self.frame_count = 0
        self.last_process_time = time.time()
        
        # AsyncAccl setup
        self.accl = None
        self.frames_processed = 0
        self.max_frames = 100  # Process 100 frames then stop for testing

    def _class_names_from_csv(self):
        """Load class names from CSV (same as working classifier)"""
        class_names = []
        with tf.io.gfile.GFile(self.class_map_csv_text) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                class_names.append(row['display_name'])
        return class_names

    def _run_preprocess_model(self, waveform):
        """Preprocess audio data (same as working classifier)"""
        input_details = self._pre_interpreter.get_input_details()
        waveform_input_index = input_details[0]['index']

        output_details = self._pre_interpreter.get_output_details()
        preprocessed_waveform = output_details[1]['index']  # Use index 1 like working classifier

        self._pre_interpreter.resize_tensor_input(waveform_input_index, [waveform.size], strict=True)
        self._pre_interpreter.allocate_tensors()
        self._pre_interpreter.set_tensor(waveform_input_index, waveform)

        self._pre_interpreter.invoke()

        preprocessed_wav_data = self._pre_interpreter.get_tensor(preprocessed_waveform)
        return preprocessed_wav_data

    def _run_postprocess_model(self, output):
        """Postprocess MXA output (same as working classifier)"""
        input_details = self._post_interpreter.get_input_details()
        waveform_input_index = input_details[0]['index']

        output_details = self._post_interpreter.get_output_details()
        postprocessed_waveform = output_details[0]['index']

        self._post_interpreter.resize_tensor_input(waveform_input_index, list(output.shape), strict=True)
        self._post_interpreter.allocate_tensors()
        self._post_interpreter.set_tensor(waveform_input_index, output)

        self._post_interpreter.invoke()

        postprocessed_wav_data = self._post_interpreter.get_tensor(postprocessed_waveform)
        return postprocessed_wav_data

    def audio_callback(self, indata, frames, time, status):
        """Audio input callback for sounddevice"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        # Convert to mono if stereo
        audio_data = indata[:, 0] if len(indata.shape) > 1 else indata
        
        # Debug: Check audio levels (sounddevice returns float32 in [-1, 1])
        audio_level = np.abs(audio_data).mean()
        if audio_level > 0.001:  # Only log if there's significant audio
            logger.info(f"Audio detected - level: {audio_level:.6f}, Max: {np.abs(audio_data).max():.6f}")
        
        # Add to buffer with thread safety
        with self.buffer_lock:
            self.audio_buffer.extend(audio_data.flatten())

    def input_callback(self):
        """
        Input callback for AsyncAccl - returns one frame per call
        """
        if not self.is_running:
            logger.info("Input callback: not running, returning None")
            return None
            
        if self.frames_processed >= self.max_frames:
            logger.info(f"Input callback: processed {self.frames_processed} frames, stopping")
            return None
            
        # Wait for enough audio data
        max_wait_time = 5.0  # Maximum wait time in seconds
        start_wait = time.time()
        
        while time.time() - start_wait < max_wait_time:
            with self.buffer_lock:
                if len(self.audio_buffer) >= self.num_samples_per_frame:
                    # Get the most recent frame
                    audio_frame = np.array(list(self.audio_buffer)[-self.num_samples_per_frame:], dtype=np.float32)
                    break
            time.sleep(0.01)  # Small delay to avoid busy waiting
        else:
            logger.warning("Input callback: timeout waiting for audio data")
            return None
        
        # Check if audio is already normalized (sounddevice returns [-1, 1])
        if np.abs(audio_frame).max() <= 1.0:
            normalized_audio = audio_frame
        else:
            normalized_audio = audio_frame / tf.int16.max
        
        logger.info(f"Input callback: audio frame stats - mean: {np.mean(normalized_audio):.6f}, max: {np.max(np.abs(normalized_audio)):.6f}")
        
        # Preprocess the frame manually
        try:
            preprocessed_data = self._run_preprocess_model(normalized_audio)
            self.frame_count += 1
            self.frames_processed += 1
            logger.info(f"Input callback: returning preprocessed frame {self.frame_count}, shape: {preprocessed_data.shape}")
            return preprocessed_data
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            return None

    def output_callback(self, *mxa_outputs):
        """
        Output callback for AsyncAccl - processes one result per call
        """
        try:
            logger.info(f"Output callback: received {len(mxa_outputs)} outputs")
            for i, output in enumerate(mxa_outputs):
                logger.info(f"  Output {i}: shape {output.shape}, dtype {output.dtype}")
            
            # Use the correct output index (same as working classifier)
            mxa_output = mxa_outputs[1]
            
            # Run through postprocessing model manually
            scores = self._run_postprocess_model(mxa_output)
            
            # Get prediction (same as working classifier)
            top_class_index = scores.argmax()
            predicted_class = self.labels[top_class_index]
            confidence = float(scores[0][top_class_index])
            
            # Update results
            self.current_prediction = predicted_class
            self.prediction_history.append(predicted_class)
            
            # Calculate processing rate
            current_time = time.time()
            fps = 1.0 / (current_time - self.last_process_time) if self.last_process_time else 0
            self.last_process_time = current_time
            
            logger.info(f"PREDICTION: {predicted_class} (confidence: {confidence:.3f}, FPS: {fps:.1f})")
            
            # Print top 3 predictions for debugging
            top_3_indices = np.argsort(scores[0])[-3:][::-1]
            logger.info("Top 3 predictions:")
            for i, idx in enumerate(top_3_indices):
                logger.info(f"  {i+1}. {self.labels[idx]}: {scores[0][idx]:.3f}")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Error in output processing: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

    def start_realtime_classification(self):
        """Start real-time classification with proper AsyncAccl setup"""
        if self.is_running:
            logger.warning("Already running!")
            return
            
        logger.info("Starting improved real-time audio classification...")
        self.is_running = True
        self.frames_processed = 0
        
        # Start audio capture
        logger.info("Starting audio capture...")
        self.audio_stream = sd.InputStream(
            callback=self.audio_callback,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=1600,  # Small block size for low latency
            dtype=np.float32  # Explicitly request float32
        )
        self.audio_stream.start()
        logger.info("Audio capture started")
        
        # Wait a moment for audio buffer to fill
        logger.info("Waiting for audio buffer to fill...")
        time.sleep(2.0)
        
        # Initialize AsyncAccl
        logger.info("Initializing AsyncAccl...")
        self.accl = AsyncAccl(dfp=self.model_dfp_path)
        
        # Connect AsyncAccl callbacks
        logger.info("Connecting AsyncAccl callbacks...")
        self.accl.connect_input(self.input_callback)
        self.accl.connect_output(self.output_callback)
        
        logger.info("AsyncAccl callbacks connected")
        logger.info(f"Listening for audio... Will process {self.max_frames} frames")
        
        # Wait for AsyncAccl to finish (this blocks)
        try:
            self.accl.wait()
            logger.info("AsyncAccl processing completed")
        except KeyboardInterrupt:
            logger.info("\nInterrupted by user")
        except Exception as e:
            logger.error(f"Error during AsyncAccl processing: {e}")
        finally:
            self.stop_classification()

    def stop_classification(self):
        """Stop real-time classification"""
        if not self.is_running:
            return
            
        logger.info("Stopping classification...")
        self.is_running = False
        
        # Stop audio capture
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
            logger.info("Audio capture stopped")
        
        # Stop AsyncAccl
        if self.accl:
            try:
                self.accl.stop()
                self.accl.shutdown()
                logger.info("AsyncAccl stopped and shutdown")
            except Exception as e:
                logger.error(f"Error stopping AsyncAccl: {e}")
        
        logger.info("Classification stopped")

    def get_prediction_summary(self):
        """Get summary of recent predictions"""
        if not self.prediction_history:
            return {}
        
        counts = Counter(self.prediction_history)
        total = len(self.prediction_history)
        
        summary = {
            class_name: (count / total) * 100 
            for class_name, count in counts.most_common(5)
        }
        
        return summary


def list_audio_devices():
    """List available audio input devices"""
    logger.info("Available audio devices:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            logger.info(f"  {i}: {device['name']} (inputs: {device['max_input_channels']})")
    print()


def main():
    """Main function to run the improved realtime classifier"""
    
    # List available audio devices
    list_audio_devices()
    
    # Model paths (same as working classifier)
    class_map_csv_text = '../../assets/yamnet_class_map.csv'
    preprocess_model_path = '../../models/Audio_classification_YamNet_96_64_1_tflite_pre.tflite'
    model_dfp_path = "../../models/Audio_classification_YamNet_96_64_1_tflite.dfp"
    postprocess_model_path = '../../models/Audio_classification_YamNet_96_64_1_tflite_post.tflite'
    
    # Create improved classifier
    classifier = ImprovedRealTimeAudioClassifier(
        class_map_csv_text,
        preprocess_model_path, 
        model_dfp_path,
        postprocess_model_path
    )
    
    try:
        # Start classification
        classifier.start_realtime_classification()
        
        # Print summary
        summary = classifier.get_prediction_summary()
        if summary:
            logger.info("\nPrediction Summary:")
            for class_name, percentage in summary.items():
                logger.info(f"  {class_name}: {percentage:.1f}%")
        else:
            logger.info("No predictions were made")
            
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
    finally:
        if 'classifier' in locals():
            classifier.stop_classification()


if __name__ == "__main__":
    main()
