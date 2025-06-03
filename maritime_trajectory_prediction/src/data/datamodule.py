import torch
import pytorch_lightning as pl
from torch_geometric.data import Data, Batch

class AISFuserDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.processor = AISGraphProcessor(
            dp_epsilon=0.03,
            dbscan_eps=1.5,  # 1.5km in degrees
            min_samples=100
        )

    def prepare_data(self):
        # Implement maritime graph generation
        raw_data = load_ais_data(self.config.data_path)
        self.processed = self.processor.process(raw_data)

    def _create_pyg_graph(self, trajectory):
        # Convert trajectory to PyG Data object
        edge_index = create_spatial_edges(trajectory.waypoints)
        return Data(
            x=trajectory.features,
            edge_index=edge_index,
            weather=trajectory.weather,
            y=trajectory.labels
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            [self._create_pyg_graph(t) for t in self.processed.train],
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=Batch.from_data_list
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            [self._create_pyg_graph(t) for t in self.processed.val],
            batch_size=self.config.batch_size,
            collate_fn=Batch.from_data_list
        )
