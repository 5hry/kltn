import contextlib
import os
from collections import namedtuple
from torch.utils.data import Dataset
from PIL import Image
from torchvision import datasets, transforms
import numpy as np

default_dataset_roots = dict(
    TabularToImg20='./data/img_iot',
    TabularToImg17='./data/img_iot',
    TabularToImg18='./data/img_iot',
    TabularToImg23='./data/img_iot'
)


dataset_normalization = dict(
    TabularToImg20=((0.5,), (0.5,)),
    TabularToImg17=((0.5,), (0.5,)),
    TabularToImg18=((0.5,), (0.5,)),
    TabularToImg23=((0.5,), (0.5,))
)


dataset_labels = dict(
    TabularToImg20=list(range(5)), #IOTID20
    TabularToImg17=list(range(9)), #2017
    TabularToImg18=list(range(8)), #2018
    TabularToImg23=list(range(14)) #2023
)

# (nc, real_size, num_classes)
DatasetStats = namedtuple('DatasetStats', ' '.join(['nc', 'real_size', 'num_classes']))

dataset_stats = dict(
    TabularToImg20=DatasetStats(1, 9, 5), #IOTID20
    TabularToImg17=DatasetStats(1, 9, 9), #2017
    TabularToImg18=DatasetStats(1, 9, 9),
    TabularToImg23=DatasetStats(1, 7, 14) #2018
)

assert(set(default_dataset_roots.keys()) == set(dataset_normalization.keys()) ==
       set(dataset_labels.keys()) == set(dataset_stats.keys()))


def get_info(state):
    name = state.dataset  # argparse dataset fmt ensures that this is lowercase and doesn't contrain hyphen
    assert name in dataset_stats, 'Unsupported dataset: {}'.format(state.dataset)
    nc, input_size, num_classes = dataset_stats[name]
    normalization = dataset_normalization[name]
    root = state.dataset_root
    if root is None:
        root = default_dataset_roots[name]
    labels = dataset_labels[name]
    return name, root, nc, input_size, num_classes, normalization, labels


@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        yield
import numpy as np
import torch
from torch.utils.data import Dataset

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

class TabularToImg(Dataset):
    def __init__(self, file_path, input_size=9, transform=None):
        """
        Args:
            file_path (str): Path to the .npz file containing 'inputs' and 'labels'.
            input_size (int): Size of the square image (e.g., 9 for 9x9 images).
            transform (callable, optional): Optional transform to be applied on an image.
        """
        # Load the .npz file with allow_pickle=True
        data = np.load(file_path, allow_pickle=True)
        
        # Ensure the file contains required keys
        if 'inputs' not in data or 'labels' not in data:
            raise KeyError("The .npz file must contain 'inputs' and 'labels' keys.")

        self.inputs = data['inputs']
        self.labels = data['labels']
        self.input_size = input_size
        self.transform = transform

        # Encode string labels into integers using LabelEncoder
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Get the sample and reshape into a 2D image
        img = self.inputs[idx].reshape(self.input_size, self.input_size).astype(np.float32)
        label = self.labels[idx]

        # Convert to tensor and add channel dimension
        img = torch.tensor(img).unsqueeze(0)  # Shape becomes [1, input_size, input_size]
        label = torch.tensor(label).long()  # Ensure labels are in the correct type

        # Apply transformations if provided
        if self.transform:
            img = self.transform(img)

        return img, label



def get_dataset(state, phase):
    assert phase in ('train', 'test'), 'Unsupported phase: %s' % phase
    name, root, nc, input_size, num_classes, normalization, _ = get_info(state)
    real_size = dataset_stats[name].real_size

    if name == 'TabularToImg20':
        # Construct the path to the .npz file
        npz_file = os.path.join(root, f"{phase}_2020.npz")
        
        # Apply normalization as part of transforms
        transform_list = [
            transforms.Normalize(*normalization),  # Normalize inputs to match dataset stats
        ]
        
        # Return the TabularToImg dataset instance
        return TabularToImg(file_path=npz_file, input_size=input_size, transform=transforms.Compose(transform_list))

    elif name == 'TabularToImg17':
        # Construct the path to the .npz file
        npz_file = os.path.join(root, f"{phase}_2017.npz")
        
        # Apply normalization as part of transforms
        transform_list = [
            transforms.Normalize(*normalization),  # Normalize inputs to match dataset stats
        ]
        
        # Return the TabularToImg dataset instance
        return TabularToImg(file_path=npz_file, input_size=input_size, transform=transforms.Compose(transform_list))

    elif name == 'TabularToImg18':
        # Construct the path to the .npz file
        npz_file = os.path.join(root, f"{phase}_2018.npz")
        
        # Apply normalization as part of transforms
        transform_list = [
            transforms.Normalize(*normalization),  # Normalize inputs to match dataset stats
        ]
        
        # Return the TabularToImg dataset instance
        return TabularToImg(file_path=npz_file, input_size=input_size, transform=transforms.Compose(transform_list))

    elif name == 'TabularToImg23':
        # Construct the path to the .npz file
        npz_file = os.path.join(root, f"{phase}_2023.npz")
        
        # Apply normalization as part of transforms
        transform_list = [
            transforms.Normalize(*normalization),  # Normalize inputs to match dataset stats
        ]
        
        # Return the TabularToImg dataset instance
        return TabularToImg(file_path=npz_file, input_size=input_size, transform=transforms.Compose(transform_list))

    else:
        raise ValueError('Unsupported dataset: %s' % state.dataset)
