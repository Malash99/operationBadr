import argparse
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from ..models.flownet_regressor import FlowNetRegressor
from ..data_processing.tum_processor import tum_to_poses, calculate_delta_poses

def load_model(model_path, seq_length=2):
    model = FlowNetRegressor(sequence_length=seq_length)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def calculate_metrics(y_true, y_pred):
    trans_rmse = np.sqrt(mean_squared_error(y_true[:, :3], y_pred[:, :3]))
    rot_rmse = np.sqrt(mean_squared_error(y_true[:, 3:], y_pred[:, 3:]))
    return {
        'translation_rmse': trans_rmse,
        'rotation_rmse': rot_rmse,
        'overall_rmse': np.sqrt(mean_squared_error(y_true, y_pred))
    }

def evaluate_model(model, test_loader, device='cuda'):
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            preds = model(inputs).cpu().numpy()
            
            all_preds.append(preds)
            all_targets.append(targets.numpy())
    
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    return calculate_metrics(y_true, y_pred)

def visualize_trajectory(ground_truth, predictions):
    # Implement your visualization logic here
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--sequence-length', type=int, default=2)
    args = parser.parse_args()
    
    model = load_model(args.model_path, args.sequence_length)
    test_loader = create_test_loader(args.data_dir, args.sequence_length)
    
    metrics = evaluate_model(model, test_loader)
    print(f"Evaluation Metrics: {metrics}")
    
    # Generate trajectory visualization
    visualize_trajectory(test_loader.dataset.poses, predictions)