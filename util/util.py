import numpy as np
from torch_geometric.data import Data, Batch
import torch
from torch.utils.data import Dataset
import pickle


class MyDataset(Dataset):
    """
    Custom dataset for handling graph data.
    """
    def __init__(self, data_path=None, data=None):
        """
        Initialize the dataset either from a file or pre-loaded data.

        Args:
            data_path (str, optional): Path to the data file.
            data (list, optional): Pre-loaded list of data samples.
        """
        if data is None:
            if data_path is None:
                data_path = '../data/data.pt'
            with open(data_path, 'rb') as f:
                self.data = pickle.load(f)
        else:
            self.data = data

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, index):
        """Retrieve a sample by index."""
        return self.data[index]

    def get_subset(self, indices):
        """
        Get a subset of the dataset based on given indices.

        Args:
            indices (list): List of indices to select.

        Returns:
            MyDataset: Subset dataset containing selected samples.
        """
        subset_data = [self.data[i] for i in indices]
        return MyDataset(data=subset_data)


def collate_fn(batch):
    """
    Collate function for batching graph samples.

    Args:
        batch (list): List of samples, each being a tuple (graph1, graph2, cell, label).

    Returns:
        tuple: Batched graphs and labels.
    """
    graphs1, graphs2, labels = [], [], []
    
    for graph1, graph2, cell, label in batch:
        graph1.cell = cell  # Attach cell feature to first graph
        graphs1.append(graph1)
        graphs2.append(graph2)
        labels.append(label)

    return Batch.from_data_list(graphs1), Batch.from_data_list(graphs2), torch.tensor(labels)


def save_metrics(metrics, filename):
    """
    Save metrics to a CSV file.

    Args:
        metrics (list): List of metric values.
        filename (str): Path to the output file.
    """
    with open(filename, 'a') as f:
        f.write(','.join(map(str, metrics)) + '\n')


# Example usage
if __name__ == '__main__':
    # Load dataset
    data_path = '../data/data.pt'
    with open(data_path, 'rb') as f:
        raw_data = pickle.load(f)
        
    full_dataset = MyDataset(data=raw_data)
    demo_subset = full_dataset.get_subset(range(15, 30))
    
    # Prepare lists for batching
    graphs2_list = [item[1] for item in demo_subset]
    
    print("Second graph list sample:", graphs2_list)
    batched_graphs2 = Batch.from_data_list(graphs2_list)
