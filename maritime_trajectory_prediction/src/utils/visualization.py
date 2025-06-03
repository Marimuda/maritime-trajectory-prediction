import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
import torch

def plot_trajectory(trajectory, title=None, figsize=(10, 6)):
    """
    Plot a single vessel trajectory
    
    Args:
        trajectory: DataFrame with lat, lon columns
        title: Optional title for the plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    plt.plot(trajectory['lon'], trajectory['lat'], 'b-', linewidth=2)
    plt.plot(trajectory['lon'].iloc[0], trajectory['lat'].iloc[0], 'go', markersize=10, label='Start')
    plt.plot(trajectory['lon'].iloc[-1], trajectory['lat'].iloc[-1], 'ro', markersize=10, label='End')
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    if title:
        plt.title(title)
    else:
        plt.title('Vessel Trajectory')
    
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    return plt.gca()

def plot_multiple_trajectories(trajectories, title=None, figsize=(12, 8), cmap=None):
    """
    Plot multiple vessel trajectories
    
    Args:
        trajectories: List of DataFrames with lat, lon columns
        title: Optional title for the plot
        figsize: Figure size
        cmap: Optional colormap for trajectories
    """
    plt.figure(figsize=figsize)
    
    if cmap is None:
        cmap = plt.cm.jet
    
    colors = cmap(np.linspace(0, 1, len(trajectories)))
    
    for i, traj in enumerate(trajectories):
        plt.plot(traj['lon'], traj['lat'], '-', color=colors[i], linewidth=1, alpha=0.7)
        plt.plot(traj['lon'].iloc[0], traj['lat'].iloc[0], 'o', color=colors[i], markersize=5)
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    if title:
        plt.title(title)
    else:
        plt.title(f'Multiple Vessel Trajectories (n={len(trajectories)})')
    
    plt.grid(True)
    plt.tight_layout()
    
    return plt.gca()

def create_interactive_map(trajectories, zoom_start=10, tiles='OpenStreetMap'):
    """
    Create an interactive map with vessel trajectories
    
    Args:
        trajectories: List of DataFrames with lat, lon columns
        zoom_start: Initial zoom level
        tiles: Map tile type
        
    Returns:
        Folium map object
    """
    # Calculate center point
    all_lats = []
    all_lons = []
    
    for traj in trajectories:
        all_lats.extend(traj['lat'].tolist())
        all_lons.extend(traj['lon'].tolist())
    
    center_lat = np.mean(all_lats)
    center_lon = np.mean(all_lons)
    
    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, tiles=tiles)
    
    # Add trajectories
    colors = plt.cm.jet(np.linspace(0, 1, len(trajectories)))
    colors = [(int(r*255), int(g*255), int(b*255)) for r, g, b, _ in colors]
    
    for i, traj in enumerate(trajectories):
        points = list(zip(traj['lat'].tolist(), traj['lon'].tolist()))
        folium.PolyLine(
            points,
            color=f'#{colors[i][0]:02x}{colors[i][1]:02x}{colors[i][2]:02x}',
            weight=3,
            opacity=0.7
        ).add_to(m)
        
        # Add start/end markers
        folium.Marker(
            [traj['lat'].iloc[0], traj['lon'].iloc[0]],
            icon=folium.Icon(color='green', icon='play', prefix='fa')
        ).add_to(m)
        
        folium.Marker(
            [traj['lat'].iloc[-1], traj['lon'].iloc[-1]],
            icon=folium.Icon(color='red', icon='stop', prefix='fa')
        ).add_to(m)
    
    return m

def plot_density_map(trajectories, bins=100, figsize=(12, 10), cmap='viridis'):
    """
    Create a 2D histogram showing vessel density
    
    Args:
        trajectories: List of DataFrames with lat, lon columns
        bins: Number of bins for histogram
        figsize: Figure size
        cmap: Colormap for density visualization
        
    Returns:
        Figure and axes objects
    """
    # Extract all coordinates
    all_lats = []
    all_lons = []
    
    for traj in trajectories:
        all_lats.extend(traj['lat'].tolist())
        all_lons.extend(traj['lon'].tolist())
    
    # Create histogram
    fig, ax = plt.subplots(figsize=figsize)
    h, xedges, yedges, im = ax.hist2d(all_lons, all_lats, bins=bins, cmap=cmap)
    
    plt.colorbar(im, ax=ax, label='Count')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Vessel Density Map')
    ax.grid(True)
    
    return fig, ax

def plot_prediction_vs_ground_truth(input_traj, pred_traj, gt_traj, figsize=(12, 8)):
    """
    Plot predicted trajectory vs ground truth
    
    Args:
        input_traj: Input trajectory data points
        pred_traj: Predicted trajectory points
        gt_traj: Ground truth trajectory points
        figsize: Figure size
        
    Returns:
        Figure and axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot input trajectory
    ax.plot(
        input_traj[:, 1], input_traj[:, 0],
        'b-', linewidth=2, label='Input Trajectory'
    )
    
    # Plot ground truth
    ax.plot(
        gt_traj[:, 1], gt_traj[:, 0],
        'g-', linewidth=2, label='Ground Truth'
    )
    
    # Plot prediction
    ax.plot(
        pred_traj[:, 1], pred_traj[:, 0],
        'r-', linewidth=2, label='Prediction'
    )
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Trajectory Prediction')
    ax.grid(True)
    ax.legend()
    
    return fig, ax

def plot_maritime_graph(graph, positions=None, figsize=(12, 10), node_size=50, edge_width=1.0):
    """
    Visualize a maritime graph
    
    Args:
        graph: NetworkX graph object
        positions: Optional node positions
        figsize: Figure size
        node_size: Size of nodes
        edge_width: Width of edges
        
    Returns:
        Figure and axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get positions if not provided
    if positions is None:
        if nx.get_node_attributes(graph, 'pos'):
            positions = nx.get_node_attributes(graph, 'pos')
        else:
            positions = nx.spring_layout(graph)
    
    # Draw nodes
    nx.draw_networkx_nodes(
        graph, positions,
        node_size=node_size,
        node_color='blue',
        alpha=0.7,
        ax=ax
    )
    
    # Draw edges with weights as width
    edge_widths = []
    if nx.get_edge_attributes(graph, 'weight'):
        weights = nx.get_edge_attributes(graph, 'weight')
        max_weight = max(weights.values())
        for u, v in graph.edges():
            width = edge_width * weights.get((u, v), 1) / max_weight
            edge_widths.append(width)
    else:
        edge_widths = [edge_width] * len(graph.edges())
    
    nx.draw_networkx_edges(
        graph, positions,
        width=edge_widths,
        alpha=0.5,
        edge_color='gray',
        ax=ax
    )
    
    ax.set_title('Maritime Traffic Graph')
    ax.axis('off')
    
    return fig, ax

def visualize_attention_weights(attention_weights, input_sequence, figsize=(10, 8)):
    """
    Visualize attention weights from transformer model
    
    Args:
        attention_weights: Attention weight matrix
        input_sequence: Input sequence tokens/timestamps
        figsize: Figure size
        
    Returns:
        Figure and axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert to numpy if tensor
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    # Create heatmap
    sns.heatmap(
        attention_weights,
        cmap='viridis',
        ax=ax,
        xticklabels=input_sequence,
        yticklabels=input_sequence
    )
    
    ax.set_title('Attention Weights')
    ax.set_xlabel('Input Sequence')
    ax.set_ylabel('Output Sequence')
    
    plt.tight_layout()
    
    return fig, ax

def plot_prediction_uncertainty(trajectories, ground_truth=None, figsize=(12, 8)):
    """
    Visualize prediction uncertainty with multiple sampled trajectories
    
    Args:
        trajectories: List of predicted trajectories
        ground_truth: Optional ground truth trajectory
        figsize: Figure size
        
    Returns:
        Figure and axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert trajectories to numpy if tensor
    if isinstance(trajectories[0], torch.Tensor):
        trajectories = [t.detach().cpu().numpy() for t in trajectories]
    
    # Plot all predicted trajectories with transparency
    for traj in trajectories:
        ax.plot(traj[:, 1], traj[:, 0], 'r-', alpha=0.1)
    
    # Calculate and plot mean trajectory
    mean_traj = np.mean(np.array(trajectories), axis=0)
    ax.plot(mean_traj[:, 1], mean_traj[:, 0], 'r-', linewidth=2, label='Mean Prediction')
    
    # Plot ground truth if provided
    if ground_truth is not None:
        if isinstance(ground_truth, torch.Tensor):
            ground_truth = ground_truth.detach().cpu().numpy()
        ax.plot(ground_truth[:, 1], ground_truth[:, 0], 'g-', linewidth=2, label='Ground Truth')
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Prediction Uncertainty ({len(trajectories)} samples)')
    ax.grid(True)
    ax.legend()
    
    return fig, ax

def create_animation(trajectories, figsize=(10, 8), interval=200):
    """
    Create an animation of vessel movements
    
    Args:
        trajectories: List of DataFrames with lat, lon columns
        figsize: Figure size
        interval: Interval between frames in milliseconds
        
    Returns:
        Animation object
    """
    from matplotlib.animation import FuncAnimation
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate bounds
    all_lats = []
    all_lons = []
    
    for traj in trajectories:
        all_lats.extend(traj['lat'].tolist())
        all_lons.extend(traj['lon'].tolist())
    
    min_lat, max_lat = min(all_lats), max(all_lats)
    min_lon, max_lon = min(all_lons), max(all_lons)
    
    # Add some padding
    lat_padding = (max_lat - min_lat) * 0.1
    lon_padding = (max_lon - min_lon) * 0.1
    
    ax.set_xlim(min_lon - lon_padding, max_lon + lon_padding)
    ax.set_ylim(min_lat - lat_padding, max_lat + lat_padding)
    
    # Setup plot
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Vessel Movements')
    ax.grid(True)
    
    # Create line objects
    lines = []
    points = []
    
    for _ in trajectories:
        line, = ax.plot([], [], '-')
        point, = ax.plot([], [], 'o')
        lines.append(line)
        points.append(point)
    
    def init():
        for line, point in zip(lines, points):
            line.set_data([], [])
            point.set_data([], [])
        return lines + points
    
    def animate(frame):
        for i, (line, point, traj) in enumerate(zip(lines, points, trajectories)):
            if frame < len(traj):
                # Plot trajectory up to current frame
                line.set_data(traj['lon'].iloc[:frame], traj['lat'].iloc[:frame])
                # Plot current position
                point.set_data([traj['lon'].iloc[frame-1]], [traj['lat'].iloc[frame-1]])
            else:
                # Keep the full trajectory if we've passed its length
                line.set_data(traj['lon'], traj['lat'])
                point.set_data([traj['lon'].iloc[-1]], [traj['lat'].iloc[-1]])
        
        return lines + points
    
    # Find max length
    max_length = max(len(traj) for traj in trajectories)
    
    # Create animation
    anim = FuncAnimation(
        fig, animate, init_func=init,
        frames=max_length, interval=interval, blit=True
    )
    
    return anim
