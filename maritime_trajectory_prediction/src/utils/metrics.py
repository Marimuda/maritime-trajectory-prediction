import torch
import numpy as np
import torchmetrics
from math import radians, sin, cos, sqrt, atan2

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    
    Args:
        lat1, lon1: Coordinates of first point
        lat2, lon2: Coordinates of second point
        
    Returns:
        Distance in kilometers
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    r = 6371  # Radius of earth in kilometers
    
    return r * c

class HaversineDistanceMetric(torchmetrics.Metric):
    """Torchmetrics implementation of Haversine distance"""
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        
        self.add_state("distances", default=[], dist_reduce_fx="cat")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        
    def update(self, preds, target):
        """
        Update state with predictions and targets
        
        Args:
            preds: Predicted coordinates (batch_size, 2) with [lat, lon]
            target: Target coordinates (batch_size, 2) with [lat, lon]
        """
        # Convert to numpy for calculation
        preds_np = preds.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        # Calculate distances
        batch_size = preds.shape[0]
        distances = []
        
        for i in range(batch_size):
            lat1, lon1 = preds_np[i]
            lat2, lon2 = target_np[i]
            
            dist = haversine_distance(lat1, lon1, lat2, lon2)
            distances.append(dist)
        
        # Update state
        self.distances.extend(distances)
        self.total += batch_size
    
    def compute(self):
        """Compute average distance"""
        return torch.tensor(np.mean(self.distances))

def rmse_error(pred, target):
    """
    Root Mean Squared Error between predicted and target trajectories
    
    Args:
        pred: Predicted trajectory (batch_size, seq_len, 2)
        target: Target trajectory (batch_size, seq_len, 2)
        
    Returns:
        RMSE error
    """
    return torch.sqrt(torch.mean((pred - target)**2))

def mae_error(pred, target):
    """
    Mean Absolute Error between predicted and target trajectories
    
    Args:
        pred: Predicted trajectory (batch_size, seq_len, 2)
        target: Target trajectory (batch_size, seq_len, 2)
        
    Returns:
        MAE error
    """
    return torch.mean(torch.abs(pred - target))

def rmse_haversine(pred_lats, pred_lons, target_lats, target_lons):
    """
    Root Mean Squared Haversine Error
    
    Args:
        pred_lats: Predicted latitudes (batch_size, seq_len)
        pred_lons: Predicted longitudes (batch_size, seq_len)
        target_lats: Target latitudes (batch_size, seq_len)
        target_lons: Target longitudes (batch_size, seq_len)
        
    Returns:
        RMSE Haversine error in kilometers
    """
    batch_size = pred_lats.shape[0]
    seq_len = pred_lats.shape[1]
    
    # Calculate Haversine distances for each point
    distances = []
    for b in range(batch_size):
        seq_distances = []
        for s in range(seq_len):
            dist = haversine_distance(
                pred_lats[b, s].item(), pred_lons[b, s].item(),
                target_lats[b, s].item(), target_lons[b, s].item()
            )
            seq_distances.append(dist)
        distances.append(seq_distances)
    
    # Convert to tensor
    distances = torch.tensor(distances)
    
    # Calculate RMSE
    return torch.sqrt(torch.mean(distances**2))
