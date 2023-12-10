import os
import torch
from torch.utils.data import Dataset
import os
import torch
import hicomodel
from tqdm import tqdm
import numpy as np


class ChromosomeDataset(Dataset):
    def __init__(self, data_dir, window, length, chr, itype):
        self.data_dir = data_dir
        self.window = window
        self.length = length
        self.itype = itype

        self.x = []
        self.y = []

        # Ensure chr is a list even if it's a single value
        if not isinstance(chr, list):
            chr = [chr]
        for i in chr:
            contact_file_name = f"{i}_{window}_{length}_{itype}_contact.pt"
            feature_file_name  = f"{i}_{window}_{length}_{itype}_feature.pt"
            feature_data_path = os.path.join(self.data_dir, feature_file_name)
            contact_data_path = os.path.join(self.data_dir, contact_file_name)
            contact = torch.load(contact_data_path)
            feature = torch.load(feature_data_path)

            # Concatenate the new data to the existing tensor
            self.x.extend(feature)
            self.y.extend(contact)
        
    def __len__(self):
            return len(self.x)

    def __getitem__(self, idx):
            return self.x[idx], self.y[idx]

def split_chromosomes(input_chr):
    all_chromosomes = list(range(1, 17))
    return ([input_chr], [chr for chr in all_chromosomes if chr != input_chr])

# val_chr,train_chr=split_chromosomes(1)

# train_dataset = ChromosomeDataset(data_dir='processed', window=128, length=128, chr=train_chr, itype='Outward')

# val_dataset = ChromosomeDataset(data_dir='../../Desktop/', window=128, length=128, chr=1, itype='Outward')
# val_loader=torch.utils.data.DataLoader(val_dataset,batch_size=1,shuffle=False,drop_last=True)
# model = newmodel.ConvTransModel(True,128)
# untrain_model = newmodel.ConvTransModel(True, 128)

# checkpoint=torch.load('../../Desktop/models/epoch=20-step=36540.ckpt',map_location=torch.device('cpu'))
# model_weights = checkpoint['state_dict']
# model_weights = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('model.')}
# untrain_weights = {k.replace('untrain.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('untrain.')}
#     # Edit keys
# model.load_state_dict(model_weights)
# untrain_model.load_state_dict(untrain_weights)

# model.eval()
# output=[]
# target=[]
# untrain_model.eval()
# untrain_output=[]
# with torch.no_grad():
#      for i in tqdm(val_loader):
#         inputs, targets = i
#         inputs = inputs.transpose(1, 2).contiguous()
#         inputs, targets = inputs.float(), targets.float()
#         outputs = model(inputs)
#         untrain_outputs = untrain_model(inputs)
#         output.append(outputs.cpu().view(-1).numpy())
#         target.append(targets.cpu().view(-1).numpy())
#         untrain_output.append(untrain_outputs.cpu().view(-1).numpy())

# np.savetxt('computed_outputs.csv', np.concatenate(output), delimiter=",")
# np.savetxt('targets.csv', np.concatenate(target), delimiter=",")
# np.savetxt('untrain_outputs.csv', np.concatenate(untrain_output), delimiter=",")






# train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=64,shuffle=True,drop_last=True)
# val_loader=torch.utils.data.DataLoader(val_dataset,batch_size=64,shuffle=False,drop_last=True)
# sampled_train_x,sampled_train_y=next(iter(train_loader))
# sampled_val_x,sampled_val_y=next(iter(val_loader))

# #print(sampled_train_x,sampled_train_y,sampled_val_y,sampled_val_y)
# print(sampled_train_x.shape,sampled_train_y.shape,sampled_val_x.shape,sampled_val_y.shape)



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