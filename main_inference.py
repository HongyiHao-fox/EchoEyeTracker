import numpy as np
import torch
import sys
import os

# Ensure the project root is in sys.path
sys.path.append(os.getcwd())

from signal_processing.gaze_processor import GazeTrakProcessor
from model.gaze_trak_model import GazeTrakModel

def test_pipeline():
    print("Starting EchoEyeTracker Pipeline Integration Test...")

    # Configuration
    FS = 50000
    MICROPHONES = 8
    SPEAKER_BANDS = [(18000, 21000), (21500, 24500)]
    CHANNELS = MICROPHONES * len(SPEAKER_BANDS)
    CHIRP_LEN = 600
    BUFFER_SIZE = 26
    
    # 1. Initialize Processors (One for each channel)
    print(f"Initializing {CHANNELS} GazeTrakProcessors...")
    processors = []
    for _ in range(MICROPHONES):
        per_mic = []
        for f_start, f_end in SPEAKER_BANDS:
            per_mic.append(
                GazeTrakProcessor(
                    fs=FS,
                    chirp_len=CHIRP_LEN,
                    f_start=f_start,
                    f_end=f_end,
                    band_low=f_start,
                    band_high=f_end,
                    buffer_size=BUFFER_SIZE,
                )
            )
        processors.append(per_mic)

    # 2. Initialize Model
    print("Initializing GazeTrakModel...")
    model = GazeTrakModel(in_channels=CHANNELS, output_dim=2, pretrained=False)
    model.eval()

    # 3. Generate Dummy Audio Data
    # Simulate a few frames of audio
    # Shape: (frames * chirp_len, channels)
    num_frames = BUFFER_SIZE + 5 # Process enough to fill buffer
    total_samples = num_frames * CHIRP_LEN
    dummy_audio = np.random.uniform(-1, 1, (total_samples, MICROPHONES))
    print(f"Generated dummy audio with shape: {dummy_audio.shape}")

    # 4. Process Audio Frame by Frame
    print("Processing frames...")
    
    # We need to simulate the streaming process. 
    # For each "chirp window", we process all channels.
    
    # In a real scenario, we might get chunks of data. 
    # Here we slice exactly chirp_len for simplicity.
    
    last_tensor_batch = None
    
    for i in range(num_frames):
        start_idx = i * CHIRP_LEN
        end_idx = start_idx + CHIRP_LEN
        
        # Audio chunk for this frame: (600, 8)
        frame_chunk = dummy_audio[start_idx:end_idx, :]
        
        channel_outputs = []
        for mic_idx in range(MICROPHONES):
            single_ch_data = frame_chunk[:, mic_idx]
            for proc in processors[mic_idx]:
                processed_matrix = proc.process_frame(single_ch_data)
                channel_outputs.append(processed_matrix)
        
        # Check if all buffers are ready (non-None)
        if all(out is not None for out in channel_outputs):
            # Stack channels: List of (26, 70) -> (16, 26, 70)
            # channel_outputs is a list of numpy arrays
            input_tensor = np.stack(channel_outputs, axis=0)
            
            # Add batch dimension: (1, 8, 20, 100)
            input_tensor = torch.FloatTensor(input_tensor).unsqueeze(0)
            
            last_tensor_batch = input_tensor
            # print(f"Frame {i}: Pipeline Ready. Tensor shape: {input_tensor.shape}")
        else:
             print(f"Frame {i}: Buffers filling...")

    # 5. Inference
    if last_tensor_batch is not None:
        print("\nRunning Inference on last frame...")
        with torch.no_grad():
            output = model(last_tensor_batch)
        
        print(f"Input Tensor Shape: {last_tensor_batch.shape}")
        print(f"Model Output: {output}")
        print(f"Model Output Shape: {output.shape}")
        print("\n✅ Integration Test Passed!")
    else:
        print("\n❌ Integration Test Failed: Buffer never filled.")

if __name__ == "__main__":
    test_pipeline()
