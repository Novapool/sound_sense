#!/usr/bin/env python3
"""
Fixed Web Interface for Real-Time Audio Classification
Uses the corrected AsyncAccl implementation with Flask and SocketIO
"""

from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
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
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask and SocketIO setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Model paths
class_map_csv_text = '../../assets/yamnet_class_map.csv'
preprocess_model_path = '../../models/Audio_classification_YamNet_96_64_1_tflite_pre.tflite'
model_dfp_path = "../../models/Audio_classification_YamNet_96_64_1_tflite.dfp"
postprocess_model_path = '../../models/Audio_classification_YamNet_96_64_1_tflite_post.tflite'


class FixedWebRealTimeAudioClassifier:
    """
    Fixed web-enabled real-time audio classifier using AsyncAccl.
    Incorporates all the fixes from the improved implementation.
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
        self.current_confidence = 0.0
        self.prediction_history = deque(maxlen=50)
        self.frame_count = 0
        self.last_process_time = time.time()
        
        # AsyncAccl setup
        self.accl = None
        self.accl_thread = None
        
        # Alert system - simple and minimal
        self.alert_sounds = {
            'critical': {  # Red alerts
                'Fire alarm': 394,
                'Smoke detector, smoke alarm': 393,
                'Siren': 390,
                'Civil defense siren': 391
            },
            'high': {  # Orange alerts  
                'Police car (siren)': 317,
                'Ambulance (siren)': 318,
                'Fire engine, fire truck (siren)': 319
            },
            'medium': {  # Yellow alerts
                'Doorbell': 349,
                'Telephone bell ringing': 384,
                'Vehicle horn, car horn, honking': 302,
                'Baby cry, infant cry': 20
            }
        }
        self.last_alert_time = 0
        self.alert_cooldown = 5.0  # 5 seconds between alerts
        
        # Discord webhook URL
        self.discord_webhook_url = "https://discord.com/api/webhooks/1421703732013568010/wUHkW-uw4UvBUGCfuHWrNUTzCcwpRk1VMoqvsxyVnKDSaO--HF-kSbxV3n7nP7GlzGRP"

    def _check_for_alert(self, predicted_class, confidence):
        """Check if current prediction should trigger an alert"""
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.last_alert_time < self.alert_cooldown:
            return None
            
        # Check if this sound is in our alert list
        for priority, sounds in self.alert_sounds.items():
            for sound_name, sound_index in sounds.items():
                if predicted_class == sound_name and confidence > 0.5:  # Simple confidence threshold
                    self.last_alert_time = current_time
                    return {
                        'type': 'alert',
                        'priority': priority,
                        'sound': sound_name,
                        'confidence': confidence,
                        'timestamp': current_time
                    }
        return None

    def _send_discord_alert(self, alert):
        """Send alert to Discord webhook"""
        try:
            # Priority color mapping
            priority_colors = {
                'critical': 0xFF0000,  # Red
                'high': 0xFF8C00,      # Orange
                'medium': 0xFFD700     # Yellow
            }
            
            # Format timestamp
            from datetime import datetime
            timestamp = datetime.fromtimestamp(alert['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            
            # Prepare Discord message data
            data = {
                "content": f"🚨 **{alert['priority'].upper()} ALERT** 🚨 <@275014760590999552>",
                "username": "Sound Sense Alert System",
                "embeds": [
                    {
                        "title": f"{alert['priority'].capitalize()} Priority Alert",
                        "description": f"**Sound Detected:** {alert['sound']}\n**Confidence:** {alert['confidence']:.1%}\n**Time:** {timestamp}",
                        "color": priority_colors.get(alert['priority'], 0x808080),
                        "fields": [
                            {
                                "name": "Priority Level",
                                "value": alert['priority'].upper(),
                                "inline": True
                            },
                            {
                                "name": "Detection Confidence",
                                "value": f"{alert['confidence']:.1%}",
                                "inline": True
                            }
                        ]
                    }
                ]
            }
            
            # Send to Discord webhook
            result = requests.post(self.discord_webhook_url, json=data, timeout=5)
            result.raise_for_status()
            logger.info(f"Discord alert sent successfully for {alert['sound']} (code {result.status_code})")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send Discord alert: {e}")
        except Exception as e:
            logger.error(f"Error sending Discord alert: {e}")

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
        
        # Add to buffer with thread safety
        with self.buffer_lock:
            self.audio_buffer.extend(audio_data.flatten())

    def input_callback(self):
        """
        FIXED: Input callback for AsyncAccl - returns one frame per call
        """
        if not self.is_running:
            return None
            
        # Wait for enough audio data
        max_wait_time = 2.0  # Maximum wait time in seconds
        start_wait = time.time()
        
        while time.time() - start_wait < max_wait_time:
            with self.buffer_lock:
                if len(self.audio_buffer) >= self.num_samples_per_frame:
                    # Get the most recent frame
                    audio_frame = np.array(list(self.audio_buffer)[-self.num_samples_per_frame:], dtype=np.float32)
                    break
            time.sleep(0.01)  # Small delay to avoid busy waiting
        else:
            # No audio data available, return None
            return None
        
        # FIXED: Proper audio normalization
        # sounddevice returns float32 in [-1, 1], so check if already normalized
        if np.abs(audio_frame).max() <= 1.0:
            normalized_audio = audio_frame
        else:
            normalized_audio = audio_frame / tf.int16.max
        
        # Preprocess the frame manually
        try:
            preprocessed_data = self._run_preprocess_model(normalized_audio)
            self.frame_count += 1
            return preprocessed_data
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            return None

    def output_callback(self, *mxa_outputs):
        """
        FIXED: Output callback for AsyncAccl - processes one result per call
        """
        try:
            # FIXED: Use the correct output index (same as working classifier)
            mxa_output = mxa_outputs[1]
            
            # Run through postprocessing model manually
            scores = self._run_postprocess_model(mxa_output)
            
            # Get prediction (same as working classifier)
            top_class_index = scores.argmax()
            predicted_class = self.labels[top_class_index]
            confidence = float(scores[0][top_class_index])
            
            # Update results
            self.current_prediction = predicted_class
            self.current_confidence = confidence
            self.prediction_history.append(predicted_class)
            
            # Calculate processing rate
            current_time = time.time()
            fps = 1.0 / (current_time - self.last_process_time) if self.last_process_time else 0
            self.last_process_time = current_time
            
            # Get top 5 predictions for web display
            top_5_indices = np.argsort(scores[0])[-5:][::-1]
            top_scores = {
                self.labels[idx]: float(scores[0][idx]) 
                for idx in top_5_indices
            }
            
            # Get prediction history summary
            history_summary = self._get_prediction_summary()
            
            # Check for alerts
            alert = self._check_for_alert(predicted_class, confidence)
            
            # Emit update to web interface
            update_data = {
                'class': predicted_class,
                'confidence': confidence,
                'scores': top_scores,
                'history': history_summary,
                'fps': fps,
                'frame_count': self.frame_count
            }
            
            # Add alert data if there's an alert
            if alert:
                update_data['alert'] = alert
                logger.info(f"ALERT: {alert['priority'].upper()} - {alert['sound']} (confidence: {alert['confidence']:.3f})")
                # Send Discord alert
                self._send_discord_alert(alert)
            
            socketio.emit('classification_update', update_data)
            
        except Exception as e:
            logger.error(f"Error in output processing: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

    def _get_prediction_summary(self):
        """Get summary of recent predictions"""
        if not self.prediction_history:
            return {}
        
        counts = Counter(self.prediction_history)
        total = len(self.prediction_history)
        
        summary = {
            class_name: (count / total) * 100 
            for class_name, count in counts.most_common(10)
        }
        
        return summary

    def start_classification(self):
        """Start real-time classification with proper AsyncAccl setup"""
        if self.is_running:
            logger.warning("Already running!")
            return False
            
        logger.info("Starting fixed web real-time audio classification...")
        self.is_running = True
        
        try:
            # Start audio capture
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
            time.sleep(1.0)
            
            # Initialize AsyncAccl
            self.accl = AsyncAccl(dfp=self.model_dfp_path)
            
            # Connect AsyncAccl callbacks with the fixed pattern
            self.accl.connect_input(self.input_callback)
            self.accl.connect_output(self.output_callback)
            
            logger.info("AsyncAccl callbacks connected")
            
            # Start AsyncAccl in background thread
            self.accl_thread = threading.Thread(target=self._run_accl)
            self.accl_thread.daemon = True
            self.accl_thread.start()
            
            logger.info("Web classification started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting classification: {e}")
            self.stop_classification()
            return False

    def _run_accl(self):
        """Run AsyncAccl in background thread"""
        try:
            self.accl.wait()
        except Exception as e:
            logger.error(f"Error in AsyncAccl thread: {e}")
        finally:
            logger.info("AsyncAccl thread finished")

    def stop_classification(self):
        """Stop real-time classification"""
        if not self.is_running:
            return False
            
        logger.info("Stopping web classification...")
        self.is_running = False
        
        # Stop audio capture
        if self.audio_stream:
            try:
                self.audio_stream.stop()
                self.audio_stream.close()
                logger.info("Audio capture stopped")
            except Exception as e:
                logger.error(f"Error stopping audio: {e}")
        
        # Stop AsyncAccl
        if self.accl:
            try:
                self.accl.stop()
                self.accl.shutdown()
                logger.info("AsyncAccl stopped and shutdown")
            except Exception as e:
                logger.error(f"Error stopping AsyncAccl: {e}")
        
        # Wait for thread to finish
        if self.accl_thread and self.accl_thread.is_alive():
            self.accl_thread.join(timeout=5)
        
        logger.info("Web classification stopped")
        return True

    def get_status(self):
        """Get current status"""
        return {
            'is_running': self.is_running,
            'current_class': self.current_prediction,
            'confidence': self.current_confidence,
            'frame_count': self.frame_count
        }


# Global classifier instance
classifier = None


@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index_realtime.html')


@app.route('/status')
def get_status():
    """Get current classifier status"""
    if classifier:
        return jsonify(classifier.get_status())
    return jsonify({'is_running': False, 'current_class': 'Not initialized'})


@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info("Client connected")
    emit('connected', {'data': 'Connected to fixed real-time classifier'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info("Client disconnected")


@socketio.on('start_classification')
def handle_start():
    """Start real-time classification"""
    global classifier
    
    if classifier is None:
        classifier = FixedWebRealTimeAudioClassifier(
            class_map_csv_text,
            preprocess_model_path,
            model_dfp_path,
            postprocess_model_path
        )
    
    if not classifier.is_running:
        success = classifier.start_classification()
        if success:
            emit('status_update', {'status': 'running'})
        else:
            emit('status_update', {'status': 'error'})
    else:
        emit('status_update', {'status': 'already_running'})


@socketio.on('stop_classification')
def handle_stop():
    """Stop real-time classification"""
    global classifier
    
    if classifier and classifier.is_running:
        success = classifier.stop_classification()
        if success:
            emit('status_update', {'status': 'stopped'})
        else:
            emit('status_update', {'status': 'error'})
    else:
        emit('status_update', {'status': 'not_running'})


if __name__ == '__main__':
    logger.info("Starting Fixed Real-Time Audio Classification Web App...")
    logger.info("Open your browser to: http://localhost:5000")
    
    # Run the Flask-SocketIO app
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
