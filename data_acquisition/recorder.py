import pygame
import sounddevice as sd
import numpy as np
import h5py
import time
import queue
import os
from typing import List

# Configuration
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
BG_COLOR = (0, 0, 0)
DOT_COLOR = (255, 0, 0)
DOT_RADIUS = 10
TARGET_FPS = 83.3
AUDIO_FS = 50000
CHANNELS = 8
BLOCK_SIZE = 1024
OUTPUT_DIR = "collected_data"
FILENAME = "eye_tracking_data"
GRID_COLS = 10
GRID_ROWS = 10
MIN_DWELL_SEC = 0.5
MAX_DWELL_SEC = 3.5
CALIBRATION_SECONDS = 15.0

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

class DataRecorder:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.audio_start_time = 0.0
        self.screen_data = [] # List of (timestamp, x, y)

    def audio_callback(self, indata, frames, time_info, status):
        """Callback for sounddevice InputStream."""
        if status:
            print(f"Audio status: {status}")
        if self.is_recording:
            # Copy data to avoid buffer overwrite issues
            self.audio_queue.put(indata.copy())

    def start_recording(self):
        self.is_recording = True
        self.audio_start_time = time.perf_counter()
        print("Recording started...")

    def stop_recording(self):
        self.is_recording = False
        print("Recording stopped.")

    def save_data(self, filename):
        print("Saving data...")
        
        # Process Audio Data
        audio_chunks = []
        while not self.audio_queue.empty():
            audio_chunks.append(self.audio_queue.get())
        
        if not audio_chunks:
            print("No audio data recorded.")
            audio_data = np.zeros((0, CHANNELS))
        else:
            audio_data = np.concatenate(audio_chunks, axis=0)

        # Process Screen Data
        screen_data_np = np.array(self.screen_data, dtype=np.float32)
        
        # Save to HDF5
        filepath = os.path.join(OUTPUT_DIR, f"{filename}_{int(time.time())}.h5")
        with h5py.File(filepath, 'w') as f:
            # Create datasets
            f.create_dataset('audio', data=audio_data, compression='gzip')
            f.create_dataset('screen_coordinates', data=screen_data_np, compression='gzip')
            
            # Metadata
            f.attrs['audio_fs'] = AUDIO_FS
            f.attrs['fps'] = TARGET_FPS
            f.attrs['start_time'] = self.audio_start_time
            
            # Description of columns for screen_data
            f.attrs['screen_data_columns'] = "relative_time_sec, x_px, y_px"

        print(f"Data saved to {filepath}")
        print(f"Audio shape: {audio_data.shape}, Screen data shape: {screen_data_np.shape}")


def _calibration_points() -> List[np.ndarray]:
    return [
        np.array([DOT_RADIUS, DOT_RADIUS], dtype=float),
        np.array([SCREEN_WIDTH - DOT_RADIUS, DOT_RADIUS], dtype=float),
        np.array([DOT_RADIUS, SCREEN_HEIGHT - DOT_RADIUS], dtype=float),
        np.array([SCREEN_WIDTH - DOT_RADIUS, SCREEN_HEIGHT - DOT_RADIUS], dtype=float),
        np.array([SCREEN_WIDTH / 2.0, SCREEN_HEIGHT / 2.0], dtype=float),
    ]


def _random_point_in_region(region_col: int, region_row: int) -> np.ndarray:
    cell_w = SCREEN_WIDTH / GRID_COLS
    cell_h = SCREEN_HEIGHT / GRID_ROWS
    left = region_col * cell_w + DOT_RADIUS
    right = min((region_col + 1) * cell_w - DOT_RADIUS, SCREEN_WIDTH - DOT_RADIUS)
    top = region_row * cell_h + DOT_RADIUS
    bottom = min((region_row + 1) * cell_h - DOT_RADIUS, SCREEN_HEIGHT - DOT_RADIUS)
    x = np.random.uniform(left, max(left, right))
    y = np.random.uniform(top, max(top, bottom))
    return np.array([x, y], dtype=float)


def _generate_session_targets() -> List[np.ndarray]:
    regions = [(c, r) for r in range(GRID_ROWS) for c in range(GRID_COLS)]
    np.random.shuffle(regions)
    return [_random_point_in_region(c, r) for c, r in regions]


def _generate_dwell_times(count: int) -> List[float]:
    return list(np.random.uniform(MIN_DWELL_SEC, MAX_DWELL_SEC, size=count))

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Eye Tracking Calibration Data Collection")
    clock = pygame.time.Clock()

    recorder = DataRecorder()

    dot_pos = np.array([SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2], dtype=float)
    phase = "idle"
    phase_start = 0.0
    calibration_pts = _calibration_points()
    session_targets: List[np.ndarray] = []
    session_dwells: List[float] = []
    session_idx = 0
    session_target_start = 0.0
    
    # Initialize Audio Stream
    stream = sd.InputStream(
        samplerate=AUDIO_FS,
        channels=CHANNELS,
        blocksize=BLOCK_SIZE,
        callback=recorder.audio_callback
    )

    running = True
    print("Press SPACE to start recording. Press ESC to quit.")
    
    font = pygame.font.SysFont(None, 24)
    
    with stream:
        while running:
            dt = clock.tick(TARGET_FPS) / 1000.0 # dt in seconds
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        if not recorder.is_recording:
                            recorder.start_recording()
                            phase = "calibration"
                            phase_start = time.perf_counter()
                            dot_pos = calibration_pts[0].copy()
                            session_targets = []
                            session_dwells = []
                            session_idx = 0
                        else:
                            recorder.stop_recording()
                            recorder.save_data(FILENAME)
                            # Reset for next recording if needed
                            recorder.screen_data = []
                            phase = "idle"

            now = time.perf_counter()

            if recorder.is_recording and phase == "calibration":
                elapsed = now - phase_start
                if elapsed >= CALIBRATION_SECONDS:
                    phase = "session"
                    session_targets = _generate_session_targets()
                    session_dwells = _generate_dwell_times(len(session_targets))
                    session_idx = 0
                    session_target_start = now
                    dot_pos = session_targets[session_idx].copy()
                else:
                    seg_duration = CALIBRATION_SECONDS / len(calibration_pts)
                    idx = min(int(elapsed / seg_duration), len(calibration_pts) - 1)
                    dot_pos = calibration_pts[idx].copy()

            elif recorder.is_recording and phase == "session":
                if session_idx < len(session_targets):
                    dwell_elapsed = now - session_target_start
                    if dwell_elapsed >= session_dwells[session_idx]:
                        session_idx += 1
                        session_target_start = now
                        if session_idx < len(session_targets):
                            dot_pos = session_targets[session_idx].copy()

            # Record Data
            if recorder.is_recording and recorder.audio_start_time > 0:
                # Time relative to audio start
                timestamp = time.perf_counter() - recorder.audio_start_time
                recorder.screen_data.append([timestamp, dot_pos[0], dot_pos[1]])

            # Drawing
            screen.fill(BG_COLOR)
            
            # Draw Dot
            pygame.draw.circle(screen, DOT_COLOR, dot_pos.astype(int), DOT_RADIUS)
            
            # Draw Status Text
            if recorder.is_recording:
                if phase == "calibration":
                    status_text = "CALIBRATION (15s)"
                else:
                    status_text = f"SESSION TARGET {session_idx + 1}/100"
            else:
                status_text = "Press SPACE to start"
            text_surf = font.render(status_text, True, (255, 255, 255))
            screen.blit(text_surf, (10, 10))

            pygame.display.flip()

    if recorder.is_recording or recorder.screen_data:
        recorder.save_data(FILENAME + "_autosave") # distinct from manual save
    pygame.quit()

if __name__ == "__main__":
    main()
