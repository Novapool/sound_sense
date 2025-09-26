#!/usr/bin/env python3
"""
YAMNet Audio Classification Benchmark Script
============================================

This script benchmarks the YAMNet audio classification model using the MemryX
Benchmark API. It measures the Frames Per Second (FPS) performance when running
audio frames through the MXA hardware accelerator.

The benchmark follows the same pattern as the MobileNet example but is adapted
for audio classification using YAMNet which processes audio waveforms and 
predicts 521 audio event classes from the AudioSet ontology.
"""

import os
import sys
from pathlib import Path
from memryx import Benchmark
import numpy as np
import time

def verify_dfp_exists(dfp_path):
    """
    Verify that the YAMNet DFP file exists.
    
    Args:
        dfp_path (str): Path to the DFP file
        
    Returns:
        bool: True if file exists, False otherwise
    """
    if not os.path.exists(dfp_path):
        print(f"Error: DFP file not found at {dfp_path}")
        print("Please ensure you have the yamnet.dfp file in the models/ directory")
        return False
    return True

def get_model_info():
    """
    Display information about YAMNet model expectations.
    
    YAMNet typically expects:
    - Sample rate: 16 kHz
    - Window size: Variable (often processes ~1 second of audio)
    - Output: 521 class predictions for AudioSet categories
    """
    print("YAMNet Model Information:")
    print("=" * 40)
    print("- Model: Google YAMNet Audio Event Classifier")
    print("- Input: Audio waveform (16 kHz sample rate)")
    print("- Output: 521 audio event class predictions")
    print("- Categories: AudioSet ontology events")
    print("- Use case: Environmental sound detection")
    print()

def benchmark_yamnet(dfp_path, num_frames=1000, verbose=True):
    """
    Benchmark the YAMNet model using the MemryX Benchmark API.
    
    This function is like a performance test for your audio AI model - imagine 
    you're testing how fast a chef can prepare dishes. Instead of food, we're 
    feeding the model audio data and measuring how many "audio samples" it can 
    process per second.
    
    Args:
        dfp_path (str): Path to the compiled YAMNet DFP file
        num_frames (int): Number of frames to run for benchmarking (default: 1000)
        verbose (bool): Whether to print detailed information
        
    Returns:
        tuple: (outputs, latency, fps) - benchmark results
    """
    if verbose:
        print(f"Initializing benchmark with DFP: {dfp_path}")
        print(f"Running {num_frames} frames for performance measurement...")
        print()
    
    try:
        # Initialize the Benchmark with the YAMNet DFP
        # Think of this like setting up a test environment for your audio AI
        benchmark = Benchmark(dfp=dfp_path, verbose=1 if verbose else 0)
        
        # Run the benchmark - this is where the magic happens!
        # The benchmark feeds random audio-like data through the model
        # and measures how fast it can process it
        with benchmark as accl:
            start_time = time.time()
            
            # Run inference with random data (threading=True for best FPS)
            # This simulates feeding audio frames to your sound detection system
            outputs, latency, fps = accl.run(
                frames=num_frames, 
                threading=True  # Enable threading for maximum throughput
            )
            
            end_time = time.time()
            total_time = end_time - start_time
            
            if verbose:
                print("Benchmark Results:")
                print("=" * 40)
                print(f"Frames processed: {num_frames}")
                print(f"Total time: {total_time:.2f} seconds")
                print(f"Average FPS: {fps:.2f}")
                print(f"Average latency: {latency:.2f} ms" if latency else "Latency: N/A (threading enabled)")
                print()
                
                # Calculate some additional metrics
                if fps > 0:
                    time_per_frame = 1000 / fps  # milliseconds per frame
                    print(f"Time per frame: {time_per_frame:.2f} ms")
                    print(f"Theoretical real-time capability: {fps:.0f}x faster than real-time")
                    print("(assuming 1 frame = 1 second of audio)")
            
            return outputs, latency, fps
            
    except Exception as e:
        print(f"Error during benchmarking: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure MXA hardware is properly connected")
        print("2. Verify MemryX drivers are installed")
        print("3. Check that the DFP file is valid")
        print("4. Try running 'mx_bench --hello' to test hardware")
        return None, None, None

def run_latency_test(dfp_path):
    """
    Run a separate latency test with threading disabled.
    
    This test measures how long it takes to process a single audio frame,
    which is important for real-time applications like live sound monitoring.
    
    Args:
        dfp_path (str): Path to the compiled YAMNet DFP file
    """
    print("Running latency test (single frame, no threading)...")
    print("-" * 50)
    
    try:
        benchmark = Benchmark(dfp=dfp_path)
        
        with benchmark as accl:
            # Run single frame with threading disabled for accurate latency
            outputs, latency, fps = accl.run(
                frames=1, 
                threading=False  # Disable threading for latency measurement
            )
            
            if latency:
                print(f"Single frame latency: {latency:.2f} ms")
                print(f"This means your sound detection system could respond")
                print(f"to audio events in {latency:.2f} milliseconds!")
            else:
                print("Latency measurement not available")
                
    except Exception as e:
        print(f"Latency test failed: {e}")

def main():
    """
    Main function to run the YAMNet benchmark.
    
    This orchestrates the entire benchmarking process, like a conductor 
    leading an orchestra through a performance test.
    """
    print("YAMNet Audio Classification Benchmark")
    print("=" * 50)
    print()
    
    # Define the path to your YAMNet DFP file
    # Adjust this path based on where you've placed your yamnet.dfp file
    current_dir = Path(__file__).parent
    dfp_path = current_dir / "models" / "Audio_classification_YamNet_96_64_1_tflite.dfp"
    
    # Convert to string for compatibility
    dfp_path_str = str(dfp_path)
    
    # Check if DFP file exists
    if not verify_dfp_exists(dfp_path_str):
        sys.exit(1)
    
    # Display model information
    get_model_info()
    
    # Run the main benchmark test
    print("Starting FPS benchmark test...")
    print("=" * 40)
    outputs, latency, fps = benchmark_yamnet(dfp_path_str, num_frames=1000)
    
    if fps is None:
        print("Benchmark failed. Please check your setup.")
        sys.exit(1)
    
    print()
    
    # Run latency test
    run_latency_test(dfp_path_str)
    
    print()
    print("Benchmark completed successfully!")
    print("=" * 50)
    
    # Provide interpretation of results
    if fps:
        print("\nResults Interpretation:")
        print("-" * 30)
        print(f"Your YAMNet model achieved {fps:.0f} FPS on MXA hardware.")
        print("This means for real-time audio monitoring applications:")
        print(f"- Can process {fps:.0f} seconds of audio per second")
        print(f"- Suitable for {'real-time' if fps >= 10 else 'batch'} processing")
        print(f"- Performance: {'Excellent' if fps >= 100 else 'Good' if fps >= 50 else 'Adequate'}")

if __name__ == "__main__":
    main()