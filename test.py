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
    import argparse

    parser = argparse.ArgumentParser(description='C.Origami testing Module.')

    parser.add_argument('--window', dest='window', default=64,
                            type=int,
                            help='Random seed for training')
    
    parser.add_argument('--length', dest='length', default=128,
                            type=int,
                            help='Random seed for training')
    
    parser.add_argument('--val_chr', dest='val_chr', default=1,
                            type=int,
                            help='Random seed for training')

    parser.add_argument('--itype', dest='itpe', default='Outward',
                            help='Path to the model checkpoint')

    # Data directories
    parser.add_argument('--data_root', dest='dataset_data_root', default='/content/drive/MyDrive/corigamidata',
                            help='Root path of training data', required=True)
 
    

    args = parser.parse_args()

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_dir=f'{args.data_root}/result'
    if not os.path.exists(save_dir):
    # Create the directory if it does not exist
        os.makedirs(save_dir)
    checkpointpath=f'{args.data_root}/checkpoint_{args.val_char}/models/{args.val_chr}.ckpt'


        
    val_dataset = ChromosomeDataset(data_dir=f'{args.data_root}/processed/', window=args.window, length=args.length, chr=args.val_chr, itype=args.itype)
    val_loader=torch.utils.data.DataLoader(val_dataset,batch_size=1,shuffle=False,drop_last=True)
    model = hicomodel.ConvTransModel(True,args.window)
    untrain_model = hicomodel.ConvTransModel(True, args.window)

    checkpoint = torch.load(checkpointpath, map_location=device)
    model_weights = checkpoint['state_dict']

    # Edit keys
    for key in list(model_weights):
        model_weights[key.replace('model.', '')] = model_weights.pop(key)
    model.load_state_dict(model_weights)

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

    np.savetxt(f'{save_dir}/computed_outputs.csv', np.concatenate(outputlist), delimiter=",")
    np.savetxt(f'{save_dir}/targets.csv', np.concatenate(targetlist), delimiter=",")
    np.savetxt(f'{save_dir}/untrain_outputs.csv', np.concatenate(untrain_outputlist), delimiter=",")

    outputlist=np.concatenate(outputlist)
    targetlist=np.concatenate(targetlist)
    untrain_outputlist=np.concatenate(untrain_outputlist)
    outputlist_reshaped = outputlist.reshape(-1, args.window, args.window)
    for i in range(len(outputlist_reshaped)):
        prediction = outputlist.reshape(-1, args.window, args.window)[i]
        truth = targetlist.reshape(-1, args.window, args.window)[i]
        untrained = untrain_outputlist.reshape(-1, args.window, args.window)[i]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        # Plot the first heatmap
        cax1 = ax1.imshow(prediction, cmap='cool', interpolation='nearest')
        fig.colorbar(cax1, ax=ax1)
        ax1.set_title('Predicted')

        # Plot the second heatmap
        cax2 = ax2.imshow(truth, cmap='cool', interpolation='nearest')
        fig.colorbar(cax2, ax=ax2)
        ax2.set_title('Truth')

        # Plot the third heatmap
        cax3 = ax3.imshow(untrained, cmap='cool', interpolation='nearest')
        fig.colorbar(cax3, ax=ax3)
        ax3.set_title('Untrained')

        # Save the plot to the specified directory
        plt.savefig(os.path.join(save_dir, f'heatmap_{i}.png'))
        plt.close(fig)  # Close the figure to free memory

    # prediction = outputlist.reshape(-1,128, 128)[1]
    # truth=targetlist.reshape(-1,128, 128)[1]
    # untrained=untrain_outputlist.reshape(-1,128, 128)[1]

    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # # Plot the first heatmap
    # cax1 = ax1.imshow(prediction, cmap='cool', interpolation='nearest')
    # fig.colorbar(cax1, ax=ax1)
    # ax1.set_title('predicted')

    # # Plot the second heatmap
    # cax2 = ax2.imshow(truth, cmap='cool', interpolation='nearest')
    # fig.colorbar(cax2, ax=ax2)
    # ax2.set_title('truth')

    # # Plot the third heatmap
    # cax3 = ax3.imshow(untrained, cmap='cool', interpolation='nearest')
    # fig.colorbar(cax3, ax=ax3)
    # ax3.set_title('untrained')

    # # Display the plot
    # plt.show()
