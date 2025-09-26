#!/usr/bin/env python3
"""
Simple YAMNet Benchmark Script
==============================

A minimal version of the YAMNet benchmark following the exact pattern
shown in the MemryX documentation examples.
"""

from memryx import Benchmark

def main():
    """Simple benchmark following the MobileNet example pattern."""
    
    # Path to your YAMNet DFP file
    dfp_path = "models/Audio_classification_YamNet_96_64_1_tflite.dfp"
    
    print("Running YAMNet benchmark...")
    
    # Initialize the Benchmark with the YAMNet DFP
    # This follows the exact same pattern as the MobileNet example
    benchmark = Benchmark(dfp=dfp_path, verbose=1)
    
    # Run the benchmark with 1000 frames (same as MobileNet example)
    with benchmark as accl:
        outputs, latency, fps = accl.run(frames=1000)
        print(f"FPS of YAMNet Accelerated on MXA: {fps:.2f}")

if __name__ == "__main__":
    main()