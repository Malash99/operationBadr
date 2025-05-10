# src/data_processing/pair_generator.py
import numpy as np

class FramePairGenerator:
    def __init__(self, sequence_length=2):
        self.sequence_length = sequence_length
        
    def create_sequences(self, image_paths):
        """Create N-consecutive frame sequences"""
        sequences = []
        for i in range(len(image_paths) - self.sequence_length + 1):
            seq = image_paths[i:i+self.sequence_length]
            sequences.append(seq)
        return sequences

    def load_image_pair(self, seq_paths):
        """Load and stack multiple frames"""
        frames = []
        for path in seq_paths:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            frames.append(img)
        return np.concatenate(frames, axis=2)  # Stack along channels