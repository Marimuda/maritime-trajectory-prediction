"""
Visualization utilities for maritime trajectory data.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
import folium
from folium import plugins
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class TrajectoryVisualizer:
    """
    Visualization utilities for maritime trajectory data and predictions.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        """
        Initialize visualizer.
        
        Args:
            style: Matplotlib style to use
        """
        plt.style.use(style)
        self.colors = plt.cm.Set1(np.linspace(0, 1, 10))
    
    def plot_trajectory(self, 
                       trajectory: pd.DataFrame,
                       title: str = "Vessel Trajectory",
                       save_path: Optional[str] = None,
                       figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot a single vessel trajectory.
        
        Args:
            trajectory: DataFrame with lat, lon columns
            title: Plot title
            save_path: Optional path to save the plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot trajectory on map
        ax1.plot(trajectory['lon'], trajectory['lat'], 'b-', alpha=0.7, linewidth=2)
        ax1.scatter(trajectory['lon'].iloc[0], trajectory['lat'].iloc[0], 
                   c='green', s=100, marker='o', label='Start', zorder=5)
        ax1.scatter(trajectory['lon'].iloc[-1], trajectory['lat'].iloc[-1], 
                   c='red', s=100, marker='s', label='End', zorder=5)
        
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.set_title(f'{title} - Geographic View')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot speed over time if available
        if 'timestamp' in trajectory.columns and 'sog' in trajectory.columns:
            ax2.plot(trajectory['timestamp'], trajectory['sog'], 'r-', linewidth=2)
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Speed (knots)')
            ax2.set_title(f'{title} - Speed Profile')
            ax2.grid(True, alpha=0.3)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        else:
            # Plot distance vs point index
            if 'distance_km' in trajectory.columns:
                ax2.plot(range(len(trajectory)), trajectory['distance_km'], 'g-', linewidth=2)
                ax2.set_xlabel('Point Index')
                ax2.set_ylabel('Distance from Previous Point (km)')
                ax2.set_title(f'{title} - Distance Profile')
                ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_multiple_trajectories(self,
                                 trajectories: List[pd.DataFrame],
                                 labels: Optional[List[str]] = None,
                                 title: str = "Multiple Trajectories",
                                 save_path: Optional[str] = None,
                                 figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot multiple trajectories on the same map.
        
        Args:
            trajectories: List of trajectory DataFrames
            labels: Optional labels for each trajectory
            title: Plot title
            save_path: Optional path to save the plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if labels is None:
            labels = [f'Trajectory {i+1}' for i in range(len(trajectories))]
        
        for i, (traj, label) in enumerate(zip(trajectories, labels)):
            color = self.colors[i % len(self.colors)]
            
            ax.plot(traj['lon'], traj['lat'], color=color, alpha=0.7, 
                   linewidth=2, label=label)
            
            # Mark start and end points
            ax.scatter(traj['lon'].iloc[0], traj['lat'].iloc[0], 
                      c=color, s=80, marker='o', alpha=0.8)
            ax.scatter(traj['lon'].iloc[-1], traj['lat'].iloc[-1], 
                      c=color, s=80, marker='s', alpha=0.8)
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_prediction_comparison(self,
                                 true_trajectory: pd.DataFrame,
                                 predicted_trajectory: pd.DataFrame,
                                 input_sequence: Optional[pd.DataFrame] = None,
                                 title: str = "Trajectory Prediction",
                                 save_path: Optional[str] = None,
                                 figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot comparison between true and predicted trajectories.
        
        Args:
            true_trajectory: True trajectory DataFrame
            predicted_trajectory: Predicted trajectory DataFrame
            input_sequence: Optional input sequence used for prediction
            title: Plot title
            save_path: Optional path to save the plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot input sequence if provided
        if input_sequence is not None:
            ax.plot(input_sequence['lon'], input_sequence['lat'], 
                   'k-', linewidth=3, alpha=0.8, label='Input Sequence')
            ax.scatter(input_sequence['lon'].iloc[-1], input_sequence['lat'].iloc[-1],
                      c='black', s=100, marker='o', label='Prediction Start')
        
        # Plot true trajectory
        ax.plot(true_trajectory['lon'], true_trajectory['lat'], 
               'g-', linewidth=2, alpha=0.8, label='True Trajectory')
        ax.scatter(true_trajectory['lon'].iloc[0], true_trajectory['lat'].iloc[0],
                  c='green', s=80, marker='o')
        ax.scatter(true_trajectory['lon'].iloc[-1], true_trajectory['lat'].iloc[-1],
                  c='green', s=80, marker='s')
        
        # Plot predicted trajectory
        ax.plot(predicted_trajectory['lon'], predicted_trajectory['lat'], 
               'r--', linewidth=2, alpha=0.8, label='Predicted Trajectory')
        ax.scatter(predicted_trajectory['lon'].iloc[0], predicted_trajectory['lat'].iloc[0],
                  c='red', s=80, marker='o')
        ax.scatter(predicted_trajectory['lon'].iloc[-1], predicted_trajectory['lat'].iloc[-1],
                  c='red', s=80, marker='s')
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_interactive_map(self,
                             trajectories: List[pd.DataFrame],
                             labels: Optional[List[str]] = None,
                             center_lat: Optional[float] = None,
                             center_lon: Optional[float] = None,
                             zoom_start: int = 10) -> folium.Map:
        """
        Create an interactive map with trajectories.
        
        Args:
            trajectories: List of trajectory DataFrames
            labels: Optional labels for each trajectory
            center_lat: Map center latitude (auto-calculated if None)
            center_lon: Map center longitude (auto-calculated if None)
            zoom_start: Initial zoom level
            
        Returns:
            Folium map object
        """
        if labels is None:
            labels = [f'Trajectory {i+1}' for i in range(len(trajectories))]
        
        # Calculate center if not provided
        if center_lat is None or center_lon is None:
            all_lats = []
            all_lons = []
            for traj in trajectories:
                all_lats.extend(traj['lat'].tolist())
                all_lons.extend(traj['lon'].tolist())
            
            center_lat = np.mean(all_lats)
            center_lon = np.mean(all_lons)
        
        # Create map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)
        
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 
                 'lightred', 'beige', 'darkblue', 'darkgreen']
        
        for i, (traj, label) in enumerate(zip(trajectories, labels)):
            color = colors[i % len(colors)]
            
            # Add trajectory line
            coordinates = [[row['lat'], row['lon']] for _, row in traj.iterrows()]
            folium.PolyLine(
                coordinates,
                color=color,
                weight=3,
                opacity=0.8,
                popup=label
            ).add_to(m)
            
            # Add start marker
            folium.Marker(
                [traj['lat'].iloc[0], traj['lon'].iloc[0]],
                popup=f'{label} - Start',
                icon=folium.Icon(color=color, icon='play')
            ).add_to(m)
            
            # Add end marker
            folium.Marker(
                [traj['lat'].iloc[-1], traj['lon'].iloc[-1]],
                popup=f'{label} - End',
                icon=folium.Icon(color=color, icon='stop')
            ).add_to(m)
        
        return m
    
    def plot_metrics_dashboard(self,
                             metrics_history: Dict[str, List[float]],
                             title: str = "Training Metrics",
                             save_path: Optional[str] = None,
                             figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot a dashboard of training metrics.
        
        Args:
            metrics_history: Dictionary with metric names and their values over time
            title: Dashboard title
            save_path: Optional path to save the plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        n_metrics = len(metrics_history)
        n_cols = 2
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (metric_name, values) in enumerate(metrics_history.items()):
            row = i // n_cols
            col = i % n_cols
            
            ax = axes[row, col]
            ax.plot(values, linewidth=2)
            ax.set_title(metric_name)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_metrics, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

