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
import os
import torch
from torch.utils.data import Dataset

class ChromosomeDataset(Dataset):
    def __init__(self, data_dir, window, length, val_chr,feature,itype,mode='train'):
        self.data_dir = data_dir
        self.window = window
        self.length = length
        self.feature=feature
        self.itype=itype
        self.val_chr = val_chr
        self.mode = mode  # 'train' or 'val'
        self.feature_matrices = []
        self.contact_matrices = []

        # Load and combine data for training
        if self.mode == 'train':
            for chr_num in range(1, 17):  # Assuming 16 chromosomes
                if chr_num == self.val_chr:
                    continue  # Skip the validation chromosome
                
                feature_file = f"{chr_num}_{window}_{length}_{feature}_{itype}_feature_matrix.pt"
                contact_file = f"{chr_num}_{window}_{length}_{feature}_{itype}_contact_matrix.pt"
                feature_path = os.path.join(self.data_dir, feature_file)
                contact_path = os.path.join(self.data_dir, contact_file)

                if os.path.exists(feature_path) and os.path.exists(contact_path):
                    feature_matrix = torch.load(feature_path)
                    contact_matrix = torch.load(contact_path)
                    self.feature_matrices.append(feature_matrix)
                    self.contact_matrices.append(contact_matrix)
            
            # Concatenate all training feature and contact matrices
            self.feature_matrices = torch.cat(self.feature_matrices, dim=0)
            self.contact_matrices = torch.cat(self.contact_matrices, dim=0)
            self.contact_matrices=self.contact_matrices.view(-1, window, window)
            if feature=='DNA':
                self.feature_matrices=self.feature_matrices.view(-1, 4, length*window)
            else:
                self.feature_matrices=self.feature_matrices.view(-1, 30, length*window)


        elif self.mode == 'val':
            feature_file = f"{val_chr}_{window}_{length}_{feature}_{itype}_feature_matrix.pt"
            contact_file = f"{val_chr}_{window}_{length}_{feature}_{itype}_contact_matrix.pt"
            feature_path = os.path.join(self.data_dir, feature_file)
            contact_path = os.path.join(self.data_dir, contact_file)
            self.feature_matrices = torch.load(feature_path)
            self.contact_matrices = torch.load(contact_path)
            self.contact_matrices=self.contact_matrices.view(-1, window, window)
            if feature=='DNA':
                self.feature_matrices=self.feature_matrices.view(-1, 4, window*length)
            else:
                self.feature_matrices=self.feature_matrices.view(-1, 30, length*window)




    def __len__(self):
        if self.mode == 'train':
            return self.feature_matrices.size(0)

    def __getitem__(self, idx):
            return self.feature_matrices[idx], self.contact_matrices[idx]



# data_dir = '../../Desktop/processed'
# window = 16
# length = 128
# val_chr = 1
# batch_size=256
# feature='DNA'
# itype='Outward'
# train_dataset = ChromosomeDataset(data_dir, window, length,val_chr,feature=feature,itype=itype,mode='train')
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
# sampled=next(iter(train_loader))
# print(sampled[1].shape,sampled[0].shape)
