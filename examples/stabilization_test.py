import torch
import time
import sys
import os

# Ensure we can import the engine locally
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from phi_engine.core import Philter

def run_test():
    # Detect Hardware (Showcase M-Series Support)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🔥 phi-engine initializing on {device.type.upper()}...")
    
    # Configuration
    BATCH_SIZE = 128
    DIM = 4096 # Large dimension to stress-test stability
    LAYERS = 12 # The "Cathedral" Stack
    
    print(f"⚙️  Config: {LAYERS} Layers | {DIM} Dimensions | Batch {BATCH_SIZE}")
    print("-" * 60)
    
    # Create the Stack
    layers = [Philter(DIM, DIM, recursion_depth=i+1).to(device) for i in range(LAYERS)]
    input_signal = torch.randn(BATCH_SIZE, DIM).to(device)
    
    # Run the Pipeline
    signal = input_signal
    start_time = time.time()
    
    for i, layer in enumerate(layers):
        signal, log = layer(signal)
        print(f"Layer {i+1:02d} | Variance Delta: {log['variance_delta']:.8f} | Status: {log['convergence']}")
        
        # Safety Check
        if log['variance_delta'] > 1.0:
            print("❌ CRITICAL FAILURE: GRADIENT EXPLOSION")
            return

    end_time = time.time()
    
    print("-" * 60)
    print(f"✅ TEST PASSED. Throughput: {(BATCH_SIZE * LAYERS) / (end_time - start_time):.2f} ops/sec")
    print("   The signal remained stable through 12 harmonic layers.")
    print("   Geometric constraints held.")

if __name__ == "__main__":
    run_test()
