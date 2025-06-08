import numpy as np
import pandas as pd
import torch
from sklearn.cluster import DBSCAN
from torch_geometric.data import Data
import networkx as nx

class GraphProcessor:
    """Base class for graph processing"""
    def __init__(self):
        pass
    
    def process(self, data):
        """Process data into graph representation"""
        raise NotImplementedError("Subclasses must implement this method")

class AISGraphProcessor(GraphProcessor):
    """Process AIS data into graph representation for GNN models"""
    def __init__(self, dp_epsilon=0.03, dbscan_eps=1.5, min_samples=100):
        """
        Initialize AIS graph processor
        
        Args:
            dp_epsilon: Douglas-Peucker epsilon value for trajectory compression
            dbscan_eps: DBSCAN epsilon value for waypoint clustering (in km)
            min_samples: Minimum samples for DBSCAN cluster
        """
        super().__init__()
        self.dp_epsilon = dp_epsilon
        self.dbscan_eps = dbscan_eps
        self.min_samples = min_samples
    
    def process(self, data):
        """
        Process AIS data into maritime graph representation
        
        Args:
            data: DataFrame containing AIS data
            
        Returns:
            Processed data with graph structure
        """
        # Clean data
        cleaned_data = self._clean_data(data)
        
        # Create trajectories
        trajectories = self._create_trajectories(cleaned_data)
        
        # Extract waypoints
        waypoints = self._extract_waypoints(trajectories)
        
        # Create graph
        graph = self._create_graph(waypoints, trajectories)
        
        return graph
    
    def _clean_data(self, data):
        """Clean AIS data (filter invalid points, etc.)"""
        # Filter out unrealistic speed values
        data = data[data['sog'] < 50]  # Max realistic speed
        
        # Filter out stationary vessels (moored/at anchor)
        data = data[data['sog'] > 0.5]  # Minimum speed threshold
        
        # Ensure sorted by timestamp
        data = data.sort_values(['mmsi', 'timestamp'])
        
        return data
    
    def _create_trajectories(self, data):
        """Create trajectories by segmenting data"""
        # Group by vessel ID
        grouped = data.groupby('mmsi')
        
        trajectories = []
        
        for mmsi, group in grouped:
            # Sort by timestamp
            group = group.sort_values('timestamp')
            
            # Calculate time differences
            time_diffs = group['timestamp'].diff().fillna(pd.Timedelta(seconds=0))
            
            # Segment when time gap is too large (e.g., 30 minutes)
            segment_indices = np.where(time_diffs > pd.Timedelta(minutes=30))[0]
            
            if len(segment_indices) == 0:
                # Single trajectory
                if len(group) >= 6:  # Minimum length
                    trajectories.append(group)
            else:
                # Multiple trajectories
                start_idx = 0
                for idx in segment_indices:
                    segment = group.iloc[start_idx:idx]
                    if len(segment) >= 6:  # Minimum length
                        trajectories.append(segment)
                    start_idx = idx
                
                # Last segment
                segment = group.iloc[start_idx:]
                if len(segment) >= 6:  # Minimum length
                    trajectories.append(segment)
        
        return trajectories
    
    def _extract_waypoints(self, trajectories):
        """Extract waypoints using Douglas-Peucker algorithm and DBSCAN clustering"""
        all_waypoints = []
        
        for traj in trajectories:
            # Extract coordinates
            coords = np.column_stack([traj['lat'].values, traj['lon'].values])
            
            # Apply Douglas-Peucker for trajectory simplification
            simplified_coords = self._douglas_peucker(coords, self.dp_epsilon)
            
            # Add to waypoints
            all_waypoints.extend(simplified_coords)
        
        # Convert to array
        all_waypoints = np.array(all_waypoints)
        
        # Apply DBSCAN to cluster waypoints
        clustering = DBSCAN(eps=self.dbscan_eps, min_samples=self.min_samples).fit(all_waypoints)
        
        # Extract cluster centers
        unique_labels = np.unique(clustering.labels_)
        waypoints = []
        
        for label in unique_labels:
            if label == -1:
                # Skip noise points
                continue
                
            # Get points in this cluster
            mask = clustering.labels_ == label
            cluster_points = all_waypoints[mask]
            
            # Calculate centroid
            centroid = cluster_points.mean(axis=0)
            waypoints.append(centroid)
        
        return np.array(waypoints)
    
    def _create_graph(self, waypoints, trajectories):
        """Create graph structure from waypoints and trajectories"""
        # Create graph
        G = nx.Graph()
        
        # Add nodes (waypoints)
        for i, waypoint in enumerate(waypoints):
            G.add_node(i, pos=(waypoint[0], waypoint[1]))
        
        # Add edges based on trajectory segments
        for traj in trajectories:
            # Map trajectory points to nearest waypoints
            traj_coords = np.column_stack([traj['lat'].values, traj['lon'].values])
            waypoint_indices = []
            
            for point in traj_coords:
                # Find nearest waypoint
                distances = np.sqrt(np.sum((waypoints - point)**2, axis=1))
                nearest_idx = np.argmin(distances)
                waypoint_indices.append(nearest_idx)
            
            # Create edges between consecutive waypoints
            for i in range(len(waypoint_indices) - 1):
                source = waypoint_indices[i]
                target = waypoint_indices[i + 1]
                
                if source != target:  # Avoid self-loops
                    if not G.has_edge(source, target):
                        G.add_edge(source, target, weight=1)
                    else:
                        # Increment edge weight
                        G[source][target]['weight'] += 1
        
        # Convert to PyG format
        return self._networkx_to_pyg(G, waypoints)
    
    def _networkx_to_pyg(self, G, waypoints):
        """Convert NetworkX graph to PyTorch Geometric format"""
        # Get edges
        edge_index = []
        edge_attr = []
        
        for source, target, data in G.edges(data=True):
            edge_index.append([source, target])
            edge_index.append([target, source])  # Make undirected
            
            edge_attr.append([data['weight']])
            edge_attr.append([data['weight']])  # Duplicate for undirected
        
        # Convert to tensor
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        # Node features (waypoint coordinates)
        x = torch.tensor(waypoints, dtype=torch.float)
        
        # Create PyG data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        return data
    
    def _douglas_peucker(self, points, epsilon):
        """
        Douglas-Peucker algorithm for trajectory simplification
        
        Args:
            points: Array of points (lat, lon)
            epsilon: Simplification threshold
            
        Returns:
            Simplified trajectory
        """
        if len(points) <= 2:
            return points.tolist()
        
        # Find point with max distance
        dmax = 0
        index = 0
        for i in range(1, len(points) - 1):
            d = self._point_line_distance(points[i], points[0], points[-1])
            if d > dmax:
                index = i
                dmax = d
        
        # If max distance > epsilon, recursively simplify
        if dmax > epsilon:
            # Recursive call
            rec_results1 = self._douglas_peucker(points[:index + 1], epsilon)
            rec_results2 = self._douglas_peucker(points[index:], epsilon)
            
            # Combine results
            return rec_results1[:-1] + rec_results2
        else:
            return [points[0].tolist(), points[-1].tolist()]
    
    def _point_line_distance(self, point, line_start, line_end):
        """Calculate perpendicular distance from point to line"""
        if np.all(line_start == line_end):
            return np.sqrt(np.sum((point - line_start)**2))
        
        # Calculate distance
        num = np.abs(
            (line_end[1] - line_start[1]) * point[0] - 
            (line_end[0] - line_start[0]) * point[1] + 
            line_end[0] * line_start[1] - 
            line_end[1] * line_start[0]
        )
        
        den = np.sqrt(
            (line_end[1] - line_start[1])**2 + 
            (line_end[0] - line_start[0])**2
        )
        
        return num / den
