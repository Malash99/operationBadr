import os
import torch
from torch.utils.data import Dataset, DataLoader
from .pair_generator import FramePairGenerator

class OdometryDataset(Dataset):
    def __init__(self, data_dir, sequence_length=2):
        self.generator = FramePairGenerator(sequence_length)
        self.image_paths = self._load_image_paths(data_dir)
        self.poses = self._load_ground_truth(data_dir)
        self.sequences = self.generator.create_sequences(self.image_paths)
        
    def _load_image_paths(self, data_dir):
        return sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.png')])
        
    def _load_ground_truth(self, data_dir):
        # Load corresponding TUM file
        tum_path = os.path.join(data_dir, 'ground_truth.tum')
        return calculate_delta_poses(tum_to_poses(tum_path))
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        frames = self.generator.load_image_pair(sequence)
        target = self.poses[idx]
        return torch.tensor(frames).permute(2, 0, 1).float(), torch.tensor(target).float()