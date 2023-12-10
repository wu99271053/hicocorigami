import torch
import torch.optim as optim
import torch.nn as nn
import hicomodel
from dataset import ChromosomeDataset
import copy
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
import os

# Assuming your model is defined as 'model'
# Assuming your data loaders are defined as 'train_loader' and 'val_loader'

# Define the loss function and optimizer
import torch
from torch.utils.data import DataLoader, Dataset

# Load your data
# feature_matrix = torch.load('/content/drive/My Drive/jokedata/feature_matrix.pt')
# contact_matrix = torch.load('/content/drive/My Drive/jokedata/contact_matrix.pt')

# # Define your custom dataset class
# class MyDataset(Dataset):
#     def __init__(self, features, labels):
#         self.features = features
#         self.labels = labels

#     def __len__(self):
#         return len(self.features)

#     def __getitem__(self, idx):
#         return self.features[idx], self.labels[idx]

# # Calculate the sizes for training and validation sets
# val_size = int(0.2 * len(feature_matrix))
# train_size = len(feature_matrix) - val_size

# # Split the dataset into training and validation sets
# train_features = feature_matrix[val_size:]
# train_labels = contact_matrix[val_size:]
# val_features = feature_matrix[:val_size]
# val_labels = contact_matrix[:val_size]

# # Create dataset instances
# train_dataset = MyDataset(train_features, train_labels)
# val_dataset = MyDataset(val_features, val_labels)

# # Create DataLoaders
# batch_size = 64
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

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

val_chr,train_chr=split_chromosomes(1)

train_dataset = ChromosomeDataset(data_dir='processed', window=128, length=128, chr=train_chr, itype='Outward')

val_dataset = ChromosomeDataset(data_dir='processed', window=128, length=128, chr=val_chr, itype='Outward')
batch_size=8
train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,drop_last=True)
val_loader=torch.utils.data.DataLoader(val_dataset,batch_size=batch_size,shuffle=False,drop_last=True)
# Step 3: Split the data
# Define the proportion for the training set (e.g., 80%)
# train_size = int(0.8 * len(dataset))
# val_size = len(dataset) - train_size

# # # Randomly split the dataset into training and validation sets
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
# batch_size=64
# # Step 4: Create DataLoaders
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,drop_last=True)
# val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False,drop_last=True)
# Calculate the sizes for training and validation sets




#data_dir = '/content/drive/My Drive/corigami_hmdna'
# data_dir = '/user4/kyoto1/ericwyx/corigami_hmdna'
# window = 16
# length = 128
# val_chr = 16
# feature='hmDNA'
# itype='Outward'
# batch_size=64

# train_dataset = ChromosomeDataset(data_dir, window, length,val_chr,feature=feature,itype=itype,mode='train')
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True,num_workers=8)
# val_dataset=ChromosomeDataset(data_dir, window, length,val_chr,feature=feature,itype=itype,mode='val')
# val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,drop_last=True,num_workers=8)



model=hicomodel.ConvTransModel(True,128)
untrain=copy.deepcopy(model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
untrain.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Number of training epochs
num_epochs = 100
best_val_loss = float('inf')
best_model_path = 'best_model.pth'

for epoch in range(num_epochs):
    # Training
    model.train()
    total_train_loss = 0
    for batch in train_loader:
        # Get data to cuda if possible
        inputs, targets = batch
        inputs = inputs.transpose(1, 2).contiguous()
        if torch.cuda.is_available():
            inputs, targets = inputs.float().cuda(), targets.float().cuda()

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # Validation
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.transpose(1, 2).contiguous()

            if torch.cuda.is_available():
                inputs, targets = inputs.float().cuda(), targets.float().cuda()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_val_loss += loss.item()


    avg_val_loss = total_val_loss / len(val_loader)
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')


model.eval()
untrain.eval()
all_outputs = []
all_targets = []
all_untrain = []
with torch.no_grad():
    for batch in val_loader:
        inputs, targets = batch
        inputs = inputs.transpose(1, 2).contiguous()

        if torch.cuda.is_available():
            inputs = inputs.float().cuda()

        outputs = model(inputs)
        untrain_outputs = untrain(inputs)
        all_outputs.append(outputs.cpu().view(-1).numpy())
        all_targets.append(targets.view(-1).numpy())
        all_untrain.append(untrain_outputs.cpu().view(-1).numpy())

np.savetxt("computed_outputs_15.csv", np.concatenate(all_outputs), delimiter=",")
np.savetxt("ground_truths_15.csv", np.concatenate(all_targets), delimiter=",")
np.savetxt("untrained_outputs_15.csv", np.concatenate(all_untrain), delimiter=",")


# computed_outputs = np.loadtxt("computed_outputs.csv", delimiter=",")
# ground_truths = np.loadtxt("ground_truths.csv", delimiter=",")
# untrained_outputs = np.loadtxt("untrained_outputs.csv", delimiter=",")

# prediction = computed_outputs[0].reshape(16, 16)
# truth=ground_truths[0].reshape(16, 16)
# untrained=untrained_outputs[0].reshape(16, 16)

# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# # Plot the first heatmap
# cax1 = ax1.imshow(prediction, cmap='hot', interpolation='nearest')
# fig.colorbar(cax1, ax=ax1)
# ax1.set_title('predicted')

# # Plot the second heatmap
# cax2 = ax2.imshow(truth, cmap='hot', interpolation='nearest')
# fig.colorbar(cax2, ax=ax2)
# ax2.set_title('truth')

# # Plot the third heatmap
# cax3 = ax3.imshow(untrained, cmap='hot', interpolation='nearest')
# fig.colorbar(cax3, ax=ax3)
# ax3.set_title('untrained')

# # Display the plot
# plt.show()
# #gergegeg