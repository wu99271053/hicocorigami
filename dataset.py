import os
import torch
from torch.utils.data import Dataset
import subprocess
import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import os
import torch
from torch.utils.data import Dataset

class ChromosomeDataset(Dataset):
    def __init__(self, data_dir, window, length, val_chr,feature,mode='train'):
        self.data_dir = data_dir
        self.window = window
        self.length = length
        self.val_chr = val_chr
        self.mode = mode  # 'train' or 'val'
        self.feature=feature
        self.files = []

        # Load file paths for training and validation
        for chr_num in range(1, 17):  # Assuming 16 chromosomes
            feature_file = f"{chr_num}_{window}_{length}_feature_matrix.pt"
            contact_file = f"{chr_num}_{window}_{length}_contact_matrix.pt"
            feature_path = os.path.join(data_dir, feature_file)
            contact_path = os.path.join(data_dir, contact_file)

            if os.path.exists(feature_path) and os.path.exists(contact_path):
                if chr_num == val_chr and mode == 'val':
                    self.files.append((feature_path, contact_path))
                elif chr_num != val_chr and mode == 'train':
                    self.files.append((feature_path, contact_path))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        feature_path, contact_path = self.files[idx]
        feature_matrix = torch.load(feature_path)
        contact_matrix = torch.load(contact_path)
        return feature_matrix, contact_matrix



