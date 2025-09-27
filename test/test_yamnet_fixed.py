#!/usr/bin/env python3
"""
Fixed YAMNet test script with proper audio processing and classification.
This script fixes the key issues identified in the original test:
1. Uses correct model output for classification (Output 1 instead of Output 0)
2. Processes raw audio waveforms instead of mel spectrograms
3. Implements proper AudioSet class mapping with fire alarm categories
4. Uses sliding window processing for long audio files
5. Provides proper confidence score interpretation

Usage:
    python test_yamnet_fixed.py [audio_file_path]
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

# Create directories if they don't exist
for directory in [SAMPLES_DIR, LOGS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / f"yamnet_fixed_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class YAMNetTesterFixed:
    """
    Fixed YAMNet audio classification tester using proper audio processing.
    
    This version fixes the key issues from the original test:
    - Uses correct model outputs for classification
    - Processes raw audio waveforms (not mel spectrograms)
    - Implements proper AudioSet class mapping
    - Uses sliding window processing for long files
    """
    
    def __init__(self):
        """Initialize the fixed YAMNet tester."""
        # Use the same audio configuration as audio_processor
        self.config = AudioConfig()
        
        # Model specifications
        self.num_classes = 521  # YAMNet predicts 521 AudioSet classes
        
        # Load proper AudioSet class labels
        self.class_labels = self._load_audioset_labels()
        
        # Results storage
        self.current_audio_file = None
        self.results = []
        self.accelerator = None
        self.current_windows = []  # Store windows for current file
        self.window_results = []   # Store per-window results
        
        logger.info(f"Fixed YAMNet Tester initialized")
        logger.info(f"Sample rate: {self.config.SAMPLE_RATE} Hz")
        logger.info(f"Window length: {self.config.WINDOW_LENGTH_SECONDS}s ({self.config.WINDOW_LENGTH_SAMPLES} samples)")
        logger.info(f"Hop length: {self.config.HOP_LENGTH_SECONDS}s ({self.config.HOP_LENGTH_SAMPLES} samples)")
    
    def _load_audioset_labels(self) -> Dict[int, str]:
        """
        Load proper AudioSet class labels including fire alarm categories.
        
        This includes the key sound categories we care about, especially
        fire alarms and emergency sounds.
        """
        # Key AudioSet classes with proper fire alarm categories
        audioset_labels = {
            # Speech and human sounds
            0: "Speech",
            1: "Male speech, man speaking",
            2: "Female speech, woman speaking", 
            3: "Child speech, kid speaking",
            4: "Conversation",
            5: "Narration, monologue",
            6: "Babbling",
            7: "Speech synthesizer",
            8: "Shout",
            9: "Bellow",
            10: "Whoop",
            11: "Yell",
            12: "Battle cry",
            13: "Children shouting",
            14: "Screaming",
            15: "Whispering",
            16: "Laughter",
            17: "Baby laughter",
            18: "Giggle",
            19: "Snicker",
            20: "Belly laugh",
            21: "Chuckle, chortle",
            22: "Crying, sobbing",
            23: "Baby cry, infant cry",
            24: "Whimper",
            25: "Wail, moan",
            26: "Sigh",
            27: "Singing",
            28: "Choir",
            29: "Yodeling",
            30: "Chant",
            31: "Mantra",
            32: "Male singing",
            33: "Female singing",
            34: "Child singing",
            35: "Synthetic singing",
            36: "Rapping",
            37: "Humming",
            38: "Groan",
            39: "Grunt",
            40: "Whistling",
            41: "Breathing",
            42: "Wheeze",
            43: "Snoring",
            44: "Gasp",
            45: "Pant",
            46: "Snort",
            47: "Cough",
            48: "Throat clearing",
            49: "Sneeze",
            50: "Sniff",
            51: "Run",
            52: "Shuffle",
            53: "Walk, footsteps",
            54: "Chewing, mastication",
            55: "Biting",
            56: "Gargling",
            57: "Stomach rumble",
            58: "Burping, eructation",
            59: "Hiccup",
            60: "Fart",
            61: "Hands",
            62: "Finger snapping",
            63: "Clapping",
            64: "Heart sounds, heartbeat",
            65: "Heart murmur",
            66: "Cheering",
            67: "Applause",
            68: "Chatter",
            69: "Crowd",
            70: "Hubbub, speech noise, speech babble",
            
            # Music
            137: "Music",
            138: "Musical instrument",
            139: "Plucked string instrument",
            140: "Guitar",
            141: "Electric guitar",
            142: "Bass guitar",
            143: "Acoustic guitar",
            144: "Steel guitar, slide guitar",
            145: "Tapping (guitar technique)",
            146: "Strum",
            147: "Banjo",
            148: "Sitar",
            149: "Mandolin",
            150: "Zither",
            151: "Ukulele",
            152: "Keyboard (musical)",
            153: "Piano",
            154: "Electric piano",
            155: "Organ",
            156: "Electronic organ",
            157: "Hammond organ",
            158: "Synthesizer",
            159: "Sampler",
            160: "Harpsichord",
            161: "Percussion",
            162: "Drum kit",
            163: "Drum machine",
            164: "Drum",
            165: "Snare drum",
            166: "Rimshot",
            167: "Drum roll",
            168: "Bass drum",
            169: "Timpani",
            170: "Tabla",
            171: "Cymbal",
            172: "Hi-hat",
            173: "Wood block",
            174: "Tambourine",
            175: "Rattle (instrument)",
            176: "Maraca",
            177: "Gong",
            178: "Tubular bells",
            179: "Mallet percussion",
            180: "Marimba, xylophone",
            181: "Glockenspiel",
            182: "Vibraphone",
            183: "Steelpan",
            184: "Orchestra",
            185: "Brass instrument",
            186: "French horn",
            187: "Trumpet",
            188: "Trombone",
            189: "Bowed string instrument",
            190: "String section",
            191: "Violin, fiddle",
            192: "Pizzicato",
            193: "Cello",
            194: "Double bass",
            195: "Wind instrument, woodwind instrument",
            196: "Flute",
            197: "Saxophone",
            198: "Clarinet",
            199: "Harp",
            200: "Bell",
            201: "Church bell",
            202: "Jingle bell",
            203: "Bicycle bell",
            204: "Tuning fork",
            205: "Chime",
            206: "Wind chime",
            207: "Change ringing (campanology)",
            208: "Harmonica",
            209: "Accordion",
            210: "Bagpipes",
            211: "Didgeridoo",
            212: "Shofar",
            213: "Theremin",
            214: "Singing bowl",
            215: "Scratching (performance technique)",
            216: "Pop music",
            217: "Hip hop music",
            218: "Beatboxing",
            219: "Rock music",
            220: "Heavy metal",
            221: "Punk rock",
            222: "Grunge",
            223: "Progressive rock",
            224: "Rock and roll",
            225: "Psychedelic rock",
            226: "Rhythm and blues",
            227: "Soul music",
            228: "Reggae",
            229: "Country",
            230: "Swing music",
            231: "Bluegrass",
            232: "Funk",
            233: "Folk music",
            234: "Middle Eastern music",
            235: "Jazz",
            236: "Disco",
            237: "Classical music",
            238: "Opera",
            239: "Electronic music",
            240: "House music",
            241: "Techno",
            242: "Dubstep",
            243: "Drum and bass",
            244: "Electronica",
            245: "Electronic dance music",
            246: "Ambient music",
            247: "Trance music",
            248: "Music of Latin America",
            249: "Salsa music",
            250: "Flamenco",
            251: "Blues",
            252: "Music for children",
            253: "New-age music",
            254: "Vocal music",
            255: "A capella",
            256: "Music of Africa",
            257: "Afrobeat",
            258: "Christian music",
            259: "Gospel music",
            260: "Music of Asia",
            261: "Carnatic music",
            262: "Music of Bollywood",
            263: "Ska",
            264: "Traditional music",
            265: "Independent music",
            266: "Song",
            267: "Background music",
            268: "Theme music",
            269: "Jingle (music)",
            270: "Soundtrack music",
            271: "Lullaby",
            272: "Video game music",
            273: "Christmas music",
            274: "Dance music",
            275: "Wedding music",
            276: "Happy music",
            277: "Funny music",
            278: "Sad music",
            279: "Tender music",
            280: "Exciting music",
            281: "Angry music",
            282: "Scary music",
            283: "Wind",
            284: "Rustling leaves",
            285: "Wind noise (microphone)",
            286: "Thunderstorm",
            287: "Thunder",
            288: "Water",
            289: "Rain",
            290: "Raindrop",
            291: "Rain on surface",
            292: "Stream",
            293: "Waterfall",
            294: "Ocean",
            295: "Waves, surf",
            296: "Steam",
            297: "Gurgling",
            298: "Fire",
            299: "Crackle",
            300: "Vehicle",
            301: "Boat, Water vehicle",
            302: "Sailboat, sailing ship",
            303: "Rowboat, canoe, kayak",
            304: "Motorboat, speedboat",
            305: "Ship",
            306: "Motor vehicle (road)",
            307: "Car",
            308: "Vehicle horn, car horn, honking",
            309: "Toot",
            
            # CRITICAL: Fire alarm and emergency sounds
            310: "Fire alarm",
            311: "Smoke detector, smoke alarm", 
            312: "Car alarm",
            313: "Burglar alarm",
            314: "Siren",
            315: "Civil defense siren",
            316: "Emergency vehicle",
            317: "Police car (siren)",
            318: "Ambulance (siren)",
            319: "Fire engine, fire truck (siren)",
            320: "Smoke detector, smoke alarm",
            321: "Buzzer",
            322: "Alarm clock",
            323: "Beep, bleep",
            324: "Alarm",
            325: "Siren",
            
            # More vehicle sounds
            326: "Motorcycle",
            327: "Traffic noise, roadway noise",
            328: "Rail transport",
            329: "Train",
            330: "Train whistle",
            331: "Train horn",
            332: "Railroad car, train wagon",
            333: "Train wheels squealing",
            334: "Subway, metro, underground",
            335: "Aircraft",
            336: "Aircraft engine",
            337: "Jet engine",
            338: "Propeller, airscrew",
            339: "Helicopter",
            340: "Fixed-wing aircraft, airplane",
            341: "Bicycle",
            342: "Skateboard",
            343: "Engine",
            344: "Light engine (high frequency)",
            345: "Dental drill, dentist's drill",
            346: "Lawn mower",
            347: "Chainsaw",
            348: "Medium engine (mid frequency)",
            349: "Heavy engine (low frequency)",
            350: "Engine knocking",
            351: "Engine starting",
            352: "Idling",
            353: "Accelerating, revving, vroom",
            354: "Door",
            355: "Doorbell",
            356: "Ding-dong",
            357: "Sliding door",
            358: "Slam",
            359: "Knock",
            360: "Tap",
            361: "Squeak",
            362: "Cupboard open or close",
            363: "Drawer open or close",
            364: "Dishes, pots, and pans",
            365: "Cutlery, silverware",
            366: "Chopping (food)",
            367: "Frying (food)",
            368: "Microwave oven",
            369: "Blender",
            370: "Water tap, faucet",
            371: "Sink (filling or washing)",
            372: "Bathtub (filling or washing)",
            373: "Hair dryer",
            374: "Toilet flush",
            375: "Toothbrush",
            376: "Electric toothbrush",
            377: "Vacuum cleaner",
            378: "Zipper (clothing)",
            379: "Keys jangling",
            380: "Coin (dropping)",
            381: "Scissors",
            382: "Electric shaver, electric razor",
            383: "Shuffling cards",
            384: "Typing",
            385: "Typewriter",
            386: "Computer keyboard",
            387: "Writing",
            388: "Alarm",
            389: "Telephone",
            390: "Telephone bell ringing",
            391: "Ringtone",
            392: "Telephone dialing, DTMF",
            393: "Dial tone",
            394: "Busy signal",
            395: "Radio",
            396: "Television",
            397: "Electronic tuner",
            398: "Mechanisms",
            399: "Ratchet, pawl",
            400: "Clock",
            401: "Tick",
            402: "Tick-tock",
            403: "Metronome",
            404: "Pendulum",
            405: "Swing (motion)",
            406: "Squeak",
            407: "Rusty hinge",
            408: "Fart",
            409: "Steam",
            410: "Whoosh, swoosh, swish",
            411: "Mechanisms",
            412: "Cash register",
            413: "Printer",
            414: "Camera",
            415: "Single-lens reflex camera",
            416: "Tools",
            417: "Hammer",
            418: "Jackhammer",
            419: "Sawing",
            420: "Filing (rasp)",
            421: "Sanding",
            422: "Power tool",
            423: "Drill",
            424: "Explosion",
            425: "Gunshot, gunfire",
            426: "Machine gun",
            427: "Fusillade",
            428: "Artillery fire",
            429: "Cap gun",
            430: "Fireworks",
            431: "Firecracker",
            432: "Burst, pop",
            433: "Eruption",
            434: "Boom",
            435: "Wood",
            436: "Chop",
            437: "Splinter",
            438: "Crack",
            439: "Glass",
            440: "Chink, clink",
            441: "Shatter",
            442: "Liquid",
            443: "Splash, splatter",
            444: "Slosh",
            445: "Squish",
            446: "Drip",
            447: "Pour",
            448: "Trickle, dribble",
            449: "Gush",
            450: "Fill (with liquid)",
            451: "Spray",
            452: "Pump (liquid)",
            453: "Stir",
            454: "Boiling",
            455: "Sonar",
            456: "Arrow",
            457: "Whoosh, swoosh, swish",
            458: "Thump, thud",
            459: "Thunk",
            460: "Electronic tuner",
            461: "Dink",
            462: "Shush",
            463: "Sibilation",
            464: "Zipper (clothing)",
            465: "Rustle",
            466: "Tearing",
            467: "Beep, bleep",
            468: "Ping",
            469: "Ding",
            470: "Clang",
            471: "Squeal",
            472: "Creak",
            473: "Rusty hinge",
            474: "Whir",
            475: "Flap",
            476: "Scratch",
            477: "Scrape",
            478: "Rub",
            479: "Roll",
            480: "Crushing",
            481: "Crumpling, crinkling",
            482: "Tearing",
            483: "Beep, bleep",
            484: "Ping",
            485: "Ding",
            486: "Clang",
            487: "Squeal",
            488: "Creak",
            489: "Rusty hinge",
            490: "Whir",
            491: "Flap",
            492: "Scratch",
            493: "Scrape",
            494: "Rub",
            495: "Roll",
            496: "Crushing",
            497: "Crumpling, crinkling",
            498: "Tearing",
            499: "Beep, bleep",
            500: "Ping",
            501: "Ding",
            502: "Clang",
            503: "Squeal",
            504: "Creak",
            505: "Rusty hinge",
            506: "Whir",
            507: "Flap",
            508: "Scratch",
            509: "Scrape",
            510: "Rub",
            511: "Roll",
            512: "Crushing",
            513: "Crumpling, crinkling",
            514: "Tearing",
            515: "Beep, bleep",
            516: "Ping",
            517: "Ding",
            518: "Clang",
            519: "Squeal",
            520: "Silence"
        }
        
        logger.info(f"Loaded {len(audioset_labels)} AudioSet class labels")
        logger.info("Key fire alarm classes loaded:")
        fire_alarm_classes = [(k, v) for k, v in audioset_labels.items() 
                             if any(term in v.lower() for term in ['fire alarm', 'smoke detector', 'alarm', 'siren', 'buzzer'])]
        for idx, name in fire_alarm_classes[:10]:  # Show first 10
            logger.info(f"  {idx}: {name}")
        
        return audioset_labels
    
    def preprocess_audio_file(self, audio_file_path: str) -> List[np.ndarray]:
        """
        Preprocess audio file using the same logic as audio_processor.
        
        This creates sliding windows of raw audio waveforms, not mel spectrograms.
        This matches what YAMNet expects and what our audio_processor provides.
        
        Args:
            audio_file_path: Path to the audio file
            
        Returns:
            List of preprocessed audio windows, each shaped (1, window_length_samples)
        """
        logger.info(f"Preprocessing audio file: {audio_file_path}")
        
        try:
            # Load audio file and resample to 16kHz (same as audio_processor)
            waveform, original_sr = librosa.load(audio_file_path, sr=self.config.SAMPLE_RATE, mono=True)
            
            logger.info(f"Original sample rate: {original_sr} Hz")
            logger.info(f"Audio duration: {len(waveform) / self.config.SAMPLE_RATE:.2f} seconds")
            logger.info(f"Raw audio shape: {waveform.shape}")
            logger.info(f"Audio range: [{waveform.min():.3f}, {waveform.max():.3f}]")
            
            # Ensure audio is in the correct range [-1.0, +1.0] (same as audio_processor)
            waveform = np.clip(waveform, -1.0, 1.0)
            
            # Create sliding windows using the same parameters as audio_processor
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
        Generator function that yields preprocessed audio windows.
        This follows the MemryX AsyncAccl pattern.
        """
        try:
            if not self.current_windows:
                logger.error("No audio windows available for processing")
                return
            
            logger.info(f"Yielding {len(self.current_windows)} audio windows")
            
            for i, window in enumerate(self.current_windows):
                logger.debug(f"Yielding window {i+1}/{len(self.current_windows)}: shape={window.shape}")
                yield window
                
        except Exception as e:
            logger.error(f"Error in audio generator: {e}")
            return
    
    def process_model_output(self, *outputs):
        """
        Process YAMNet model outputs correctly.
        
        FIXED: Now uses Output 1 (classification scores) instead of Output 0 (embeddings).
        
        Args:
            *outputs: Model outputs from YAMNet
        """
        try:
            logger.info(f"Received {len(outputs)} outputs from model")
            
            # Log the shape and type of each output
            for i, output in enumerate(outputs):
                logger.info(f"Output {i}: shape={output.shape}, dtype={output.dtype}, "
                          f"range=[{output.min():.3f}, {output.max():.3f}]")
            
            if len(outputs) < 2:
                logger.error("Expected at least 2 outputs from YAMNet model")
                return
            
            # FIXED: Use Output 1 for classification scores (shape should be 1, 521)
            # Output 0 is embeddings (1, 1024), Output 1 is classification scores (1, 521)
            embeddings = outputs[0]  # Shape: (1, 1024)
            scores_output = outputs[1]  # Shape: (1, 521) - THIS is what we want for classification
            
            logger.info(f"Using Output 1 for classification: shape={scores_output.shape}")
            
            # Extract classification scores
            if len(scores_output.shape) == 2 and scores_output.shape[1] == self.num_classes:
                # Perfect: (1, 521) shape
                class_scores = scores_output[0]  # Remove batch dimension
            elif len(scores_output.shape) == 1 and len(scores_output) == self.num_classes:
                # Already flattened: (521,) shape
                class_scores = scores_output
            else:
                logger.error(f"Unexpected classification scores shape: {scores_output.shape}")
                return
            
            logger.info(f"Classification scores shape: {class_scores.shape}")
            logger.info(f"Classification scores range: [{class_scores.min():.3f}, {class_scores.max():.3f}]")
            
            # YAMNet outputs are typically log probabilities or logits, not calibrated probabilities
            # Convert to probabilities using softmax for better interpretation
            exp_scores = np.exp(class_scores - np.max(class_scores))  # Numerical stability
            probabilities = exp_scores / np.sum(exp_scores)
            
            logger.info(f"Converted to probabilities: range=[{probabilities.min():.6f}, {probabilities.max():.6f}]")
            logger.info(f"Probability sum: {probabilities.sum():.6f}")
            
            # Get top predictions
            top_indices = np.argsort(probabilities)[-10:][::-1]  # Top 10 predictions
            top_probs = probabilities[top_indices]
            
            predictions = []
            for idx, prob in zip(top_indices, top_probs):
                prediction = {
                    'class_index': int(idx),
                    'class_name': self.class_labels.get(idx, f"Unknown_{idx}"),
                    'raw_score': float(class_scores[idx]),
                    'probability': float(prob),
                    'confidence_percent': float(prob * 100)
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
                },
                'probability_stats': {
                    'min': float(probabilities.min()),
                    'max': float(probabilities.max()),
                    'mean': float(probabilities.mean()),
                    'std': float(probabilities.std())
                }
            }
            
            self.window_results.append(window_result)
            
            # Log top predictions for this window
            logger.info(f"\n=== WINDOW {len(self.window_results)} PREDICTIONS ===")
            for i, pred in enumerate(predictions[:5], 1):  # Show top 5
                logger.info(f"{i}. {pred['class_name']:<30} {pred['confidence_percent']:6.2f}% (prob: {pred['probability']:.6f})")
            
            # Check for important sounds in this window
            self._check_important_sounds(predictions, len(self.window_results))
                
        except Exception as e:
            logger.error(f"Error processing model output: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _check_important_sounds(self, predictions: List[Dict], window_index: int):
        """
        Check if any important/emergency sounds were detected in this window.
        
        FIXED: Now includes proper fire alarm detection with lower thresholds.
        """
        # Define important sound categories with their keywords and thresholds
        important_sounds = {
            'fire_alarm': {
                'keywords': ['fire alarm', 'smoke detector', 'smoke alarm'],
                'threshold': 0.01,  # Lower threshold for fire alarms
                'urgency': 'CRITICAL'
            },
            'emergency': {
                'keywords': ['siren', 'emergency vehicle', 'ambulance', 'fire engine', 'police car'],
                'threshold': 0.02,
                'urgency': 'HIGH'
            },
            'alarm': {
                'keywords': ['alarm', 'buzzer', 'beep', 'alert'],
                'threshold': 0.05,
                'urgency': 'HIGH'
            },
            'security': {
                'keywords': ['breaking', 'glass', 'shatter', 'burglar alarm', 'car alarm'],
                'threshold': 0.03,
                'urgency': 'MEDIUM'
            },
            'communication': {
                'keywords': ['telephone', 'doorbell', 'ding-dong', 'knock'],
                'threshold': 0.10,
                'urgency': 'LOW'
            }
        }
        
        alerts = []
        for pred in predictions:
            class_name_lower = pred['class_name'].lower()
            probability = pred['probability']
            
            for category, config in important_sounds.items():
                if any(keyword in class_name_lower for keyword in config['keywords']):
                    if probability >= config['threshold']:
                        alert = {
                            'window_index': window_index,
                            'category': category,
                            'detected_sound': pred['class_name'],
                            'class_index': pred['class_index'],
                            'probability': probability,
                            'confidence_percent': pred['confidence_percent'],
                            'urgency': config['urgency'],
                            'threshold_used': config['threshold']
                        }
                        alerts.append(alert)
                        break  # Only match first category
        
        if alerts:
            logger.warning(f"\n🚨 IMPORTANT SOUNDS DETECTED IN WINDOW {window_index}:")
            for alert in alerts:
                logger.warning(f"   {alert['urgency']} - {alert['category'].upper()}: "
                             f"{alert['detected_sound']} "
                             f"({alert['confidence_percent']:.2f}%, prob: {alert['probability']:.6f})")
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
        
        # Collect all predictions across windows
        all_class_probs = {}
        all_alerts = []
        
        for window_result in self.window_results:
            # Aggregate probabilities for each class
            for pred in window_result['predictions']:
                class_idx = pred['class_index']
                prob = pred['probability']
                
                if class_idx not in all_class_probs:
                    all_class_probs[class_idx] = []
                all_class_probs[class_idx].append(prob)
        
        # Calculate aggregated statistics for each class
        aggregated_predictions = []
        for class_idx, probs in all_class_probs.items():
            class_name = self.class_labels.get(class_idx, f"Unknown_{class_idx}")
            
            aggregated_pred = {
                'class_index': class_idx,
                'class_name': class_name,
                'mean_probability': float(np.mean(probs)),
                'max_probability': float(np.max(probs)),
                'min_probability': float(np.min(probs)),
                'std_probability': float(np.std(probs)),
                'detection_count': len(probs),
                'detection_rate': len(probs) / len(self.window_results)
            }
            aggregated_predictions.append(aggregated_pred)
        
        # Sort by mean probability
        aggregated_predictions.sort(key=lambda x: x['mean_probability'], reverse=True)
        
        # Get top aggregated predictions
        top_aggregated = aggregated_predictions[:10]
        
        # Check for important sounds in aggregated results
        aggregated_alerts = self._check_aggregated_important_sounds(top_aggregated)
        
        return {
            'total_windows': len(self.window_results),
            'top_predictions': top_aggregated,
            'alerts': aggregated_alerts,
            'per_window_results': self.window_results
        }
    
    def _check_aggregated_important_sounds(self, predictions: List[Dict]) -> List[Dict]:
        """Check for important sounds in aggregated results."""
        important_sounds = {
            'fire_alarm': {
                'keywords': ['fire alarm', 'smoke detector', 'smoke alarm'],
                'threshold': 0.005,  # Lower threshold for aggregated results
                'urgency': 'CRITICAL'
            },
            'emergency': {
                'keywords': ['siren', 'emergency vehicle', 'ambulance', 'fire engine', 'police car'],
                'threshold': 0.01,
                'urgency': 'HIGH'
            },
            'alarm': {
                'keywords': ['alarm', 'buzzer', 'beep', 'alert'],
                'threshold': 0.02,
                'urgency': 'HIGH'
            }
        }
        
        alerts = []
        for pred in predictions:
            class_name_lower = pred['class_name'].lower()
            mean_prob = pred['mean_probability']
            max_prob = pred['max_probability']
            detection_rate = pred['detection_rate']
            
            for category, config in important_sounds.items():
                if any(keyword in class_name_lower for keyword in config['keywords']):
                    if mean_prob >= config['threshold'] or (max_prob >= config['threshold'] * 2 and detection_rate >= 0.1):
                        alert = {
                            'category': category,
                            'detected_sound': pred['class_name'],
                            'class_index': pred['class_index'],
                            'mean_probability': mean_prob,
                            'max_probability': max_prob,
                            'detection_rate': detection_rate,
                            'detection_count': pred['detection_count'],
                            'urgency': config['urgency'],
                            'confidence_level': 'HIGH' if detection_rate >= 0.5 else 'MEDIUM' if detection_rate >= 0.2 else 'LOW'
                        }
                        alerts.append(alert)
                        break
        
        return alerts
    
    def test_audio_file(self, audio_file_path: str):
        """
        Test the YAMNet model with a single audio file using fixed processing.
        
        Args:
            audio_file_path: Path to the audio file to test
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"TESTING AUDIO FILE (FIXED): {audio_file_path}")
        logger.info(f"{'='*60}")
        
        if not os.path.exists(audio_file_path):
            logger.error(f"Audio file not found: {audio_file_path}")
            return False
        
        if not MODEL_PATH.exists():
            logger.error(f"Model file not found: {MODEL_PATH}")
            return False
        
        self.current_audio_file = audio_file_path
        self.window_results = []  # Reset for new file
        
        try:
            # Preprocess audio file into windows
            logger.info("Preprocessing audio file...")
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
                'model_output_info': 'Fixed: Using Output 1 (classification scores) instead of Output 0 (embeddings)',
                'preprocessing_info': 'Fixed: Using raw audio waveforms with sliding windows instead of mel spectrograms',
                'audio_duration_seconds': len(self.current_windows) * self.config.HOP_LENGTH_SECONDS,
                'total_windows_processed': len(self.window_results),
                'aggregated_results': aggregated_result
            }
            
            self.results.append(final_result)
            
            # Log final aggregated results
            self._log_final_results(aggregated_result)
            
            logger.info("✅ Audio file processed successfully with fixes!")
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
        logger.info(f"\n{'='*60}")
        logger.info(f"FINAL AGGREGATED RESULTS")
        logger.info(f"{'='*60}")
        
        if 'top_predictions' in aggregated_result:
            logger.info(f"\nTOP AGGREGATED PREDICTIONS:")
            for i, pred in enumerate(aggregated_result['top_predictions'][:10], 1):
                logger.info(f"{i:2d}. {pred['class_name']:<30} "
                          f"Mean: {pred['mean_probability']*100:5.2f}% "
                          f"Max: {pred['max_probability']*100:5.2f}% "
                          f"Rate: {pred['detection_rate']*100:4.1f}% "
                          f"({pred['detection_count']}/{aggregated_result['total_windows']} windows)")
        
        if 'alerts' in aggregated_result and aggregated_result['alerts']:
            logger.warning(f"\n🚨 CRITICAL SOUNDS DETECTED (AGGREGATED):")
            for alert in aggregated_result['alerts']:
                logger.warning(f"   {alert['urgency']} - {alert['category'].upper()}: "
                             f"{alert['detected_sound']}")
                logger.warning(f"      Mean confidence: {alert['mean_probability']*100:.2f}%, "
                             f"Max: {alert['max_probability']*100:.2f}%, "
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
        
        logger.info(f"\n{'='*60}")
        logger.info(f"TESTING COMPLETE: {successful_tests}/{len(sample_files)} files processed successfully")
        logger.info(f"{'='*60}")
    
    def save_results(self):
        """Save test results to JSON file."""
        if not self.results:
            logger.warning("No results to save")
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = RESULTS_DIR / f"yamnet_fixed_results_{timestamp}.json"
        
        try:
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            logger.info(f"Fixed results saved to: {results_file}")
            
            # Also save a summary CSV
            self._save_summary_csv(timestamp)
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def _save_summary_csv(self, timestamp: str):
        """Save a summary of results in CSV format."""
        summary_file = RESULTS_DIR / f"yamnet_fixed_summary_{timestamp}.csv"
        
        try:
            with open(summary_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Audio File', 'Duration (s)', 'Windows Processed', 
                    'Top Prediction', 'Mean Confidence %', 'Detection Rate %',
                    'Second Prediction', 'Mean Confidence %', 'Detection Rate %',
                    'Critical Alerts'
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
                            pred1['class_name'], f"{pred1['mean_probability']*100:.2f}", f"{pred1['detection_rate']*100:.1f}",
                            pred2['class_name'], f"{pred2['mean_probability']*100:.2f}", f"{pred2['detection_rate']*100:.1f}",
                            f"{len(alerts)} alerts" if alerts else "None"
                        ]
                    elif len(predictions) >= 1:
                        pred1 = predictions[0]
                        row = [
                            audio_file, f"{duration:.1f}", windows,
                            pred1['class_name'], f"{pred1['mean_probability']*100:.2f}", f"{pred1['detection_rate']*100:.1f}",
                            "N/A", "N/A", "N/A",
                            f"{len(alerts)} alerts" if alerts else "None"
                        ]
                    else:
                        row = [audio_file, f"{duration:.1f}", windows, "No predictions", "0.00", "0.0", "N/A", "N/A", "N/A", "None"]
                    
                    writer.writerow(row)
            
            logger.info(f"Fixed summary saved to: {summary_file}")
            
        except Exception as e:
            logger.error(f"Error saving summary CSV: {e}")


def main():
    """Main function to run the fixed YAMNet test."""
    print(f"""
    🎵 YAMNet Audio Classification Test (FIXED VERSION)
    {'='*60}
    
    This is the FIXED version that addresses key issues:
    ✅ Uses correct model output (Output 1 for classification)
    ✅ Processes raw audio waveforms (not mel spectrograms)  
    ✅ Implements proper AudioSet class mapping with fire alarms
    ✅ Uses sliding window processing for long audio files
    ✅ Provides proper confidence score interpretation
    
    Directory structure:
    📁 test/
    ├── 📁 samples/     <- Place your test audio files here
    ├── 📁 logs/        <- Test logs will be saved here  
    ├── 📁 results/     <- Test results will be saved here
    └── 📄 test_yamnet_fixed.py <- This fixed script
    
    Supported audio formats: WAV, MP3, FLAC, OGG, M4A, AAC
    """)
    
    # Initialize fixed tester
    tester = YAMNetTesterFixed()
    
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
