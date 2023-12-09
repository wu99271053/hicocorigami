import os
import torch
from torch.utils.data import Dataset
import os
import torch


import os
import torch
from torch.utils.data import Dataset

class ChromosomeDataset(Dataset):
    def __init__(self, data_dir, window, length, chr, itype):
        self.data_dir = data_dir
        self.window = window
        self.length = length
        self.itype = itype
        self.data = torch.empty((0,))

        # Ensure chr is a list even if it's a single value
        if not isinstance(chr, list):
            chr = [chr]

        # Load data for each chromosome and store in a list
        # data_list = []
        # for i in chr:
        #     data_file_name = f"{i}_{window}_{length}_{itype}_data.pt"
        #     data_path = os.path.join(self.data_dir, data_file_name)
        #     data = torch.load(data_path)
        #     data_list.extend(data)

        # # Combine data from all chromosomes if there are multiple, else use single chromosome data
        # if len(data_list) > 1:
        #     # Assuming data can be concatenated along the first dimension
        #     self.data = torch.cat(data_list, dim=0)
        # else:
        #     self.data = data_list[0]
        for i in chr:
            data_file_name = f"{i}_{window}_{length}_{itype}_data.pt"
            data_path = os.path.join(self.data_dir, data_file_name)
            data = torch.load(data_path)

            # Concatenate the new data to the existing tensor
            self.data = torch.cat((self.data, data), dim=0)
        
    def __len__(self):
            return len(self.data[0])

    def __getitem__(self, idx):
            return self.data[0][idx], self.data[1][idx]


    # Implement __len__ and __getitem__ as per your requirement

# Example usage

def split_chromosomes(input_chr):
    all_chromosomes = list(range(1, 17))
    return ([input_chr], [chr for chr in all_chromosomes if chr != input_chr])

val_chr,train_chr=split_chromosomes(1)

train_dataset = ChromosomeDataset(data_dir='processed', window=256, length=128, chr=[1,3], itype='Outward')

val_dataset = ChromosomeDataset(data_dir='processed', window=256, length=128, chr=[2], itype='Outward')

train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=256,shuffle=True,drop_last=True)
val_loader=torch.utils.data.DataLoader(val_dataset,batch_size=256,shuffle=False,drop_last=True)
sampled_train_x,sampled_train_y=next(iter(train_loader))
sampled_val_x,sampled_val_y=next(iter(val_loader))

#print(sampled_train_x,sampled_train_y,sampled_val_y,sampled_val_y)
print(sampled_train_x.shape,sampled_train_y.shape,sampled_val_x.shape,sampled_val_y.shape)



# data_dir = '../../My Drive/corigamidata'
# window = 16
# length = 128
# val_chr = 1
# batch_size=256
# feature='DNA'
# itype='Outward'
# train_dataset = ChromosomeDataset(data_dir, window, length,val_chr,feature=feature,itype=itype,mode='val')
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
# sampled=next(iter(train_loader))
# print(sampled[1].shape,sampled[0].shape)
# # data=torch.load('../../My Drive/corigamidata/1_16_128_DNA_Outward_contact_matrix.pt')
# # print(data.shape)

# feature_matrix = torch.load('../../My Drive/jokedata/feature_matrix.pt')
# contact_matrix = torch.load('../../My Drive/jokedata/contact_matrix.pt')

# # Step 2: Create a custom dataset
# class MyDataset(Dataset):
#     def __init__(self, features, labels):
#         self.features = features
#         self.labels = labels

#     def __len__(self):
#         return len(self.features)

#     def __getitem__(self, idx):
#         return self.features[idx], self.labels[idx]

# dataset = MyDataset(feature_matrix, contact_matrix)

# # Step 3: Split the data
# # Define the proportion for the training set (e.g., 80%)
# train_size = int(0.8 * len(dataset))
# val_size = len(dataset) - train_size

# # Randomly split the dataset into training and validation sets
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# # Step 4: Create DataLoaders
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,drop_last=True)
# val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False,drop_last=True)

# sampled=next(iter(train_loader))
# print(sampled[1].shape,sampled[0].shape)