"""
Live audio inference - real-time adaptation and metrics display
Usage: python live_inference.py --checkpoint artifacts/natvox_model.pth
"""

import argparse
import torch
import numpy as np
import time
from pathlib import Path
from model import NATVOXAdapter
from live_audio import LiveAudioProcessor, LiveAudioPlayback, get_available_input_devices, get_available_output_devices


def print_device_info():
    """Print available audio devices"""
    print("\n" + "="*60)
    print("📢 AVAILABLE INPUT DEVICES:")
    print("="*60)
    input_devices = get_available_input_devices()
    for name, idx in input_devices.items():
        print(f"  [{idx}] {name}")
    
    print("\n" + "="*60)
    print("🔊 AVAILABLE OUTPUT DEVICES:")
    print("="*60)
    output_devices = get_available_output_devices()
    for name, idx in output_devices.items():
        print(f"  [{idx}] {name}")
    print("="*60 + "\n")


def live_inference(checkpoint_path, input_device=None, output_device=None, duration=None):
    """
    Run live inference session
    
    Args:
        checkpoint_path: Path to model checkpoint
        input_device: Input device ID (None = default)
        output_device: Output device ID (None = default)
        duration: Duration to run in seconds (None = infinite)
    """
    
    # Load model
    print(f"\n📦 Loading model from {checkpoint_path}...")
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    model = NATVOXAdapter()
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()
    print("✅ Model loaded successfully!")
    
    # Create processors
    processor = LiveAudioProcessor(model, sample_rate=16000, buffer_duration=2.0)
    playback = LiveAudioPlayback(sample_rate=16000)
    
    # Start capture
    print(f"🎤 Starting live audio capture...")
    processor.start_capture(device=input_device)
    
    try:
        print("\n" + "="*60)
        print("🟢 LIVE INFERENCE ACTIVE - Press Ctrl+C to stop")
        print("="*60 + "\n")
        
        start_time = time.time()
        last_print = time.time()
        print_interval = 1.0  # Print metrics every second
        
        while True:
            # Check if duration exceeded
            if duration and (time.time() - start_time) > duration:
                break
            
            # Get latest results
            result = processor.get_latest_results(timeout=0.5)
            
            if result:
                elapsed = time.time() - start_time
                print(
                    f"⏱️  {elapsed:6.1f}s | "
                    f"Chunk: {result['chunk']:4d} | "
                    f"RMS: {result['rms']:7.4f} | "
                    f"Similarity: {result['similarity']:6.4f}"
                )
            
            # Print summary periodically
            if (time.time() - last_print) > print_interval:
                metrics = processor.get_metrics_summary()
                if metrics['chunks_processed'] > 0:
                    print(
                        f"\n📊 SUMMARY | "
                        f"Chunks: {metrics['chunks_processed']:d} | "
                        f"Avg Similarity: {metrics['avg_similarity']:.4f} | "
                        f"Range: [{metrics['min_similarity']:.4f}, {metrics['max_similarity']:.4f}]\n"
                    )
                last_print = time.time()
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopping live inference...")
    
    finally:
        # Cleanup
        processor.stop_capture()
        playback.stop_playback()
        
        # Final summary
        final_metrics = processor.get_metrics_summary()
        print("\n" + "="*60)
        print("📈 FINAL METRICS SUMMARY")
        print("="*60)
        print(f"Total chunks processed: {final_metrics['chunks_processed']}")
        print(f"Average similarity: {final_metrics['avg_similarity']:.4f}")
        print(f"Similarity range: [{final_metrics['min_similarity']:.4f}, {final_metrics['max_similarity']:.4f}]")
        print(f"Std deviation: {final_metrics['std_similarity']:.4f}")
        print(f"Average input RMS: {final_metrics['avg_input_rms']:.4f}")
        print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Live audio inference with NATVOX-AI adapter',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default devices
  python live_inference.py --checkpoint artifacts/natvox_model.pth
  
  # Run with specific devices
  python live_inference.py --checkpoint artifacts/natvox_model.pth --input-device 0 --output-device 1
  
  # Run for 30 seconds
  python live_inference.py --checkpoint artifacts/natvox_model.pth --duration 30
  
  # List available devices
  python live_inference.py --list-devices
        """
    )
    
    parser.add_argument(
        '--checkpoint',
        default='natvox_model.pth',
        help='Path to model checkpoint (default: natvox_model.pth)'
    )
    parser.add_argument(
        '--input-device',
        type=int,
        default=None,
        help='Input device ID (see --list-devices)'
    )
    parser.add_argument(
        '--output-device',
        type=int,
        default=None,
        help='Output device ID (see --list-devices)'
    )
    parser.add_argument(
        '--duration',
        type=float,
        default=None,
        help='Duration to run in seconds (default: infinite until Ctrl+C)'
    )
    parser.add_argument(
        '--list-devices',
        action='store_true',
        help='List available audio devices and exit'
    )
    
    args = parser.parse_args()
    
    if args.list_devices:
        print_device_info()
    else:
        live_inference(
            checkpoint_path=args.checkpoint,
            input_device=args.input_device,
            output_device=args.output_device,
            duration=args.duration
        )
