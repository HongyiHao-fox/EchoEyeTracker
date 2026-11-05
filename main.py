import pygame
import sounddevice as sd
import numpy as np
import time
import h5py
import threading
import queue
import sys

# Parameters
SAMPLE_RATE = 50000  # Audio sample rate (must be > 2 × max chirp freq ~24.5kHz)
CHANNELS = 8         # Multi-channel audio for microphone array
BLOCK_SIZE = 1024    # Audio block size for stream
TARGET_FPS = 83.3    # Target frequency for coordinate recording
DURATION = 10        # Duration of recording in seconds
FILENAME = "recording.h5"
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
DOT_RADIUS = 20
DOT_COLOR = (255, 0, 0)
BG_COLOR = (0, 0, 0)

# Global variables for data collection
audio_queue = queue.Queue()
is_recording = False

def audio_callback(indata, frames, time_info, status):
    """Callback function for audio stream."""
    if status:
        print(status, file=sys.stderr)
    if is_recording:
        # Provide a timestamp for the start of this chunk if needed, 
        # but purely contiguous samples are better.
        try:
            audio_queue.put(indata.copy())
        except queue.Full:
            pass # Should unlikely happen with unlimited queue

def run_experiment():
    global is_recording

    # Initialize Pygame
    pygame.init()
    # Request VSync off to allow higher FPS if needed for the 83.3Hz target
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.DOUBLEBUF | pygame.HWSURFACE)
    pygame.display.set_caption("Visual Stimulus & Audio Recorder")
    clock = pygame.time.Clock()

    # Data containers
    recorded_coords = []  # List to store (timestamp, x, y)
    audio_data_list = []

    # Prepare audio stream
    # We use a stream with a callback to ensure no samples are dropped
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audio_callback)
    
    start_time = time.perf_counter()
    print("Starting recording...")
    
    with stream:
        # Give a moment for the stream to stabilize
        time.sleep(0.5) 
        
        is_recording = True
        
        running = True
        frame_count = 0
        
        # We calculate the interval in seconds
        interval = 1.0 / TARGET_FPS
        next_target_time = start_time

        # Movement parameters
        x, y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
        dx, dy = 3, 2

        while running:
            current_time = time.perf_counter()
            elapsed = current_time - start_time

            if elapsed >= DURATION:
                running = False
                break

            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Update dot position
            x += dx
            y += dy
            
            # Bounce off walls
            if x - DOT_RADIUS < 0 or x + DOT_RADIUS > SCREEN_WIDTH:
                dx = -dx
            if y - DOT_RADIUS < 0 or y + DOT_RADIUS > SCREEN_HEIGHT:
                dy = -dy

            # Drawing
            screen.fill(BG_COLOR)
            pygame.draw.circle(screen, DOT_COLOR, (int(x), int(y)), DOT_RADIUS)
            pygame.display.flip()

            # Recording at specific frequency
            # We check if it is time to record a sample
            # Note: This runs the visual loop as fast as possible (or vsync) 
            # but records coordinates at TARGET_FPS? 
            # OR we limit the loop itself. Limiting the loop is simpler for synchronization.
            # Let's try to stick to the loop timing.
            
            # Using clock.tick for rough timing, but custom logic for precise logging
            # actually clock.tick is good enough for most non-realtime-kernel OS needs.
            # However, 83.3 is float. clock.tick accepts float in newer pygame versions? Use tick_busy_loop for better precision.
            
            dt = clock.tick(TARGET_FPS) # This attempts to hold the loop at TARGET_FPS
            
            # Record accurate timestamp and position
            # Timestamp relative to start
            rec_time = time.perf_counter() - start_time
            recorded_coords.append([rec_time, x, y])

        is_recording = False

    print("Recording finished. Processing data...")
    pygame.quit()

    # Collect all audio from queue
    while not audio_queue.empty():
        audio_data_list.append(audio_queue.get())
    
    if not audio_data_list:
        print("No audio captured!")
        return

    # Concatenate audio chunks
    audio_full = np.concatenate(audio_data_list, axis=0)
    
    # Convert coordinates to numpy array
    coords_array = np.array(recorded_coords)
    
    print(f"Audio shape: {audio_full.shape}")
    print(f"Coords shape: {coords_array.shape}")
    print(f"Saving to {FILENAME}...")

    # Save to HDF5
    with h5py.File(FILENAME, 'w') as f:
        # Create dataset for audio
        dset_audio = f.create_dataset("audio", data=audio_full)
        dset_audio.attrs["samplerate"] = SAMPLE_RATE
        dset_audio.attrs["start_time_perf_counter"] = start_time # Raw system time if needed for cross-ref

        # Create dataset for screen coordinates
        # Columns: [Timestamp (sec), X, Y]
        dset_coords = f.create_dataset("screen_coordinates", data=coords_array)
        dset_coords.attrs["fps_target"] = TARGET_FPS
        dset_coords.attrs["columns"] = "timestamp_sec,x,y"

    print("Done.")

if __name__ == "__main__":
    run_experiment()
