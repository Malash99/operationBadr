import argparse
import torch
from torch.utils.data import DataLoader
from ..data_processing.dataloader import OdometryDataset
from ..models.flownet_regressor import FlowNetRegressor

def train_model(data_dir, cameras, sequence_length, model_output):
    datasets = []
    for cam in cameras:
        cam_dir = os.path.join(data_dir, f'cam{cam}')
        datasets.append(OdometryDataset(cam_dir, sequence_length))
    
    full_dataset = torch.utils.data.ConcatDataset(datasets)
    loader = DataLoader(full_dataset, batch_size=8, shuffle=True)
    
    model = FlowNetRegressor(sequence_length=sequence_length)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()
    
    # Training loop
    for epoch in range(100):
        for inputs, targets in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        torch.save(model.state_dict(), model_output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--cameras', nargs='+', type=int, required=True)
    parser.add_argument('--sequence-length', type=int, default=2)
    parser.add_argument('--model-output', required=True)
    args = parser.parse_args()
    
    train_model(args.data_dir, args.cameras, args.sequence_length, args.model_output)