# Real-Time Audio Classification with MemryX MXA

A comprehensive audio classification system that uses Google's YAMNet model running on MemryX hardware accelerators to classify audio in real-time. The system can identify 521 different audio events from the AudioSet ontology and includes both batch processing and real-time streaming capabilities with web-based visualization.

## 🎯 Features

- **Real-time audio classification** using YAMNet model (521 audio classes)
- **Batch processing** for uploaded audio files (.wav format)
- **Live audio recording** and classification through web interface
- **Interactive web dashboard** with real-time charts and visualizations
- **Alert system** with Discord webhook integration for critical sounds
- **Hardware acceleration** using MemryX MXA chips
- **Multiple interfaces**: Web UI, command-line, and real-time streaming

## 🏗️ Architecture

The project uses a sophisticated pipeline architecture that's like a factory assembly line for audio processing:

```
Audio Input → Preprocessing → MemryX MXA → Postprocessing → Classification Results
     ↓              ↓              ↓             ↓                    ↓
[Microphone/   [Normalization  [Hardware     [Score        [Web Dashboard/
 File Upload]   & Windowing]   Acceleration]  Processing]   Alert System]
```

### Key Components

1. **AudioClassify Class** (`classify_audio.py`): Core classification engine
2. **Flask Web App** (`app.py`): File upload and batch processing interface  
3. **Real-time System** (`realtime_app.py`): Live audio streaming with WebSocket
4. **MemryX Integration**: Hardware-accelerated inference using AsyncAccl API

## 📋 Requirements

### Hardware
- MemryX MXA hardware (4-chip solution recommended)
- Microphone for real-time classification
- 8GB+ RAM for optimal performance

### Software
- Linux (Ubuntu 20.04+ recommended, ARM64 supported)
- Python 3.9, 3.10, 3.11, or 3.12
- MemryX SDK v2.0

## 🚀 Installation

### 1. Install MemryX SDK

```bash
# Create virtual environment
python3 -m venv ~/mx
source ~/mx/bin/activate
pip3 install --upgrade pip wheel

# Install MemryX SDK
pip3 install --extra-index-url https://developer.memryx.com/pip memryx
```

### 2. Install Runtime Dependencies

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install libhdf5-dev python3-dev cmake python3-venv build-essential

# Install MemryX drivers and runtime
sudo apt install memx-drivers memx-accl
```

### 3. Install Project Dependencies

```bash
# Clone and setup project
git clone <your-repository>
cd audio-classification
pip install -r requirements.txt
```

### 4. Download Model Assets

The project requires YAMNet model files and class mappings:
- `yamnet_class_map.csv` - Audio class definitions (521 classes)
- Pre-trained YAMNet models (preprocessing, main, postprocessing)
- Compiled DFP files for MemryX hardware

## 📁 Project Structure

```
├── src/python/
│   ├── app.py                    # Flask web interface for file uploads
│   ├── classify_audio.py         # Core audio classification engine
│   ├── realtime_app.py          # Real-time streaming with WebSocket
│   ├── realtime_classification.py # Command-line real-time version
│   └── templates/
│       ├── index.html           # File upload interface
│       └── index_realtime.html  # Real-time dashboard
├── assets/
│   └── yamnet_class_map.csv     # AudioSet class definitions
├── models/                      # YAMNet model files
└── requirements.txt             # Python dependencies
```

## 🎮 Usage

### Web Interface (Recommended)

#### Batch Processing
```bash
python src/python/app.py
# Open http://localhost:5000
# Upload .wav files or record audio directly
```

#### Real-time Classification
```bash
python src/python/realtime_app.py  
# Open http://localhost:5000
# View live audio classification with charts and alerts
```

### Command Line Interface
```bash
python src/python/realtime_classification.py
# Real-time classification with terminal output
```

## 🎵 Supported Audio Classes

The system can classify 521 different audio events including:

**Human Sounds**: Speech, laughter, crying, singing, coughing, sneezing
**Animal Sounds**: Dog bark, cat meow, bird chirping, cow moo, horse neigh  
**Musical Instruments**: Piano, guitar, drums, violin, trumpet, saxophone
**Mechanical Sounds**: Car engine, motorcycle, airplane, train, machinery
**Environmental**: Thunder, rain, wind, ocean waves, fire crackling
**Alert Sounds**: Sirens, alarms, doorbell, phone ringing

*See `assets/yamnet_class_map.csv` for the complete list*

## 🚨 Alert System

The real-time system includes intelligent audio monitoring with three priority levels:

- **🔴 Critical**: Fire alarms, smoke detectors, emergency sirens
- **🟠 High Priority**: Police/ambulance sirens, emergency vehicles  
- **🟡 Medium Priority**: Doorbell, phone ringing, car horns, baby crying

Alerts are sent via Discord webhook integration for immediate notification.

## ⚙️ Technical Details

### Model Architecture
- **Base Model**: Google YAMNet (trained on AudioSet)
- **Input**: 16kHz mono audio, 15,600 samples per frame
- **Output**: 521-class probability distribution
- **Processing**: Sliding window with 7,800 sample hop size

### Hardware Acceleration
The system leverages MemryX MXA chips through a three-stage pipeline:

1. **Preprocessing Model**: Audio normalization and feature extraction
2. **Main Model**: YAMNet inference on MXA hardware  
3. **Postprocessing Model**: Score computation and classification

This is like having a specialized audio processing factory where each stage is optimized for its specific task, resulting in real-time performance.

### Performance Metrics
- **Latency**: <50ms end-to-end processing time
- **Throughput**: 1500+ FPS on 4-chip MXA solution
- **Accuracy**: Based on AudioSet-trained YAMNet performance

## 🌐 Web Interface Features

### Batch Processing Interface
- Drag-and-drop file upload (.wav files)
- Direct microphone recording
- Real-time preview and processing
- Interactive pie charts showing classification confidence
- Detailed results with intermediate predictions

### Real-time Dashboard  
- Live audio classification with streaming updates
- Real-time confidence visualization
- Top classifications chart (live updating)
- Classification history tracking
- Performance metrics (FPS, frame count, runtime)
- Alert system with visual notifications

## 🔧 Configuration

### Audio Settings
```python
# In classify_audio.py
SAMPLE_RATE = 16000        # Required by YAMNet
FRAME_SIZE = 15600         # Samples per frame  
HOP_SIZE = 7800           # Frame overlap
```

### Alert Configuration
```python
# In realtime_app.py - Customize alert sounds
alert_sounds = {
    'critical': ['Fire alarm', 'Smoke detector'],
    'high': ['Police car (siren)', 'Ambulance (siren)'], 
    'medium': ['Doorbell', 'Baby cry']
}
```

### Discord Integration
Set your Discord webhook URL in `realtime_app.py`:
```python
discord_webhook_url = "YOUR_DISCORD_WEBHOOK_URL"
```

## 🐛 Troubleshooting

### Common Issues

**Audio Device Not Found**
```bash
# List available audio devices
python -c "import sounddevice as sd; print(sd.query_devices())"
```

**MemryX Hardware Not Detected**
```bash
# Verify MXA connection
mx_bench --hello
```

**Permission Errors**
```bash
# Add user to audio group
sudo usermod -a -G audio $USER
# Reboot required
```

**Model Loading Issues**
- Ensure all model files are in the correct `models/` directory
- Verify DFP files are compiled for your specific MXA hardware
- Check file permissions and paths

## 📊 Performance Optimization

### For Real-time Applications
- Use `AsyncAccl` with proper callback configuration
- Minimize audio buffer sizes for lower latency
- Enable hardware acceleration for preprocessing/postprocessing

### For Batch Processing  
- Process multiple files in parallel when possible
- Use larger batch sizes for improved throughput
- Consider `SyncAccl` for simpler offline processing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project builds upon Google's YAMNet model and MemryX SDK. Please refer to their respective licenses for usage terms.

## 🙏 Acknowledgments

- **Google Research** for the YAMNet model and AudioSet dataset
- **MemryX** for hardware acceleration SDK and documentation
- **AudioSet** for comprehensive audio event taxonomy

## 📞 Support

For technical support:
- MemryX Documentation: https://developer.memryx.com
- YAMNet Model: https://www.kaggle.com/models/google/yamnet
- Issues: Create an issue in this repository

---

*Built with ❤️ for real-time audio intelligence*
