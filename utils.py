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

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    window=128
    length=128
    val_chr=2
    itpye='Outward'
    rootpath=f'/content/drive/MyDrive/checkpoint_{val_chr}'
    data_dir=rootpath
    checkpointpath=f'{rootpath}/models/{val_chr}.ckpt'


        
    val_dataset = ChromosomeDataset(data_dir=data_dir, window=window, length=length, chr=val_chr, itype=itpye)
    val_loader=torch.utils.data.DataLoader(val_dataset,batch_size=1,shuffle=False,drop_last=True)
    model = hicomodel.ConvTransModel(True,window)
    untrain_model = hicomodel.ConvTransModel(True, window)

    checkpoint=torch.load(checkpointpath,map_location=torch.device(device))

    model_weights = checkpoint['state_dict']
    model_weights = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('model.')}
    untrain_weights = {k.replace('untrain.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('untrain.')}
        # Edit keys

    model.load_state_dict(model_weights)
    untrain_model.load_state_dict(untrain_weights)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    untrain_model.eval()
    model.to(device)
    untrain_model.to(device)
    outputlist=[]
    targetlist=[]
    untrain_outputlist=[]
    with torch.no_grad():
        for i in tqdm(val_loader):
            inputs, targets = i
            inputs = inputs.transpose(1, 2).contiguous()
            inputs, targets = inputs.float(), targets.float()
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            untrain_outputs = untrain_model(inputs)
            outputlist.append(outputs.cpu().view(-1).numpy())
            targetlist.append(targets.cpu().view(-1).numpy())
            untrain_outputlist.append(untrain_outputs.cpu().view(-1).numpy())

    np.savetxt(f'{data_dir}/computed_outputs.csv', np.concatenate(outputlist), delimiter=",")
    np.savetxt(f'{data_dir}/targets.csv', np.concatenate(targetlist), delimiter=",")
    np.savetxt(f'{data_dir}/untrain_outputs.csv', np.concatenate(untrain_outputlist), delimiter=",")

    outputlist=np.concatenate(outputlist)
    targetlist=np.concatenate(targetlist)
    untrain_outputlist=np.concatenate(untrain_outputlist)

    prediction = outputlist.reshape(-1,128, 128)[1]
    truth=targetlist.reshape(-1,128, 128)[1]
    untrained=untrain_outputlist.reshape(-1,128, 128)[1]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Plot the first heatmap
    cax1 = ax1.imshow(prediction, cmap='cool', interpolation='nearest')
    fig.colorbar(cax1, ax=ax1)
    ax1.set_title('predicted')

    # Plot the second heatmap
    cax2 = ax2.imshow(truth, cmap='cool', interpolation='nearest')
    fig.colorbar(cax2, ax=ax2)
    ax2.set_title('truth')

    # Plot the third heatmap
    cax3 = ax3.imshow(untrained, cmap='cool', interpolation='nearest')
    fig.colorbar(cax3, ax=ax3)
    ax3.set_title('untrained')

    # Display the plot
    plt.show()
