# src/training/config.py
class Config:
    SEQUENCE_LENGTH = 2  # Can be 2 or 3
    CAMERAS = [0, 1, 2, 3]  # Train on 4 cameras
    TEST_CAMERA = 4
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4