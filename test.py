import os
import torch
from torch.utils.data import Dataset
import os
import torch
import hicomodel
from tqdm import tqdm
import numpy as np
import scipy.stats as stats
import pandas as pd


def get_colormap(itype):
    colormaps = {
        "Inward": "cool",
        "Outward": "winter",
        "Tandem+": "autumn",
        "Tandem-": "summer",
        "Combined": "Wistia"
    }

    return colormaps.get(itype, "cool") 

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

    parser.add_argument('--itype', default='Outward',
                            help='Path to the model checkpoint')

    # Data directories
    parser.add_argument('--data_root', default='/content/drive/MyDrive/corigamidata',
                            help='Root path of training data', required=True)
    
    parser.add_argument('--gaussian',action='store_true',
                        help='processed data and saved checkpoint')
    

 

    args = parser.parse_args()

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(args)
    if args.gaussian:
        save_dir=f'{args.itype}/gaussian/{args.window}/checkpoint_{args.val_chr}/result_1'
        if not os.path.exists(save_dir):
        # Create the directory if it does not exist
            os.makedirs(save_dir)
        checkpointpath=f'{args.itype}/gaussian/{args.window}/checkpoint_{args.val_chr}/models/{args.val_chr}.ckpt'
        data_dir=f'{args.data_root}/gaussian/'
    else:
        save_dir=f'{args.itype}/notransform/{args.window}/checkpoint_{args.val_chr}/result_1'
        if not os.path.exists(save_dir):
        # Create the directory if it does not exist
            os.makedirs(save_dir)
        checkpointpath=f'{args.itype}/notransform/{args.window}/checkpoint_{args.val_chr}/models/{args.val_chr}.ckpt'
        data_dir=f'{args.data_root}/notransform/'




        
    val_dataset = ChromosomeDataset(data_dir=data_dir, window=args.window, length=args.length, chr=args.val_chr, itype=args.itype)
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
    colormap = get_colormap(args.itype)
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

    computed_outputs = np.round(np.concatenate(outputlist), 2)
    targets = np.round(np.concatenate(targetlist), 2)
    untrain_outputs = np.round(np.concatenate(untrain_outputlist), 2)
    combined_data = np.column_stack((computed_outputs, targets, untrain_outputs))

    
    os.makedirs(f'{save_dir}/csv',exist_ok=True)
    os.makedirs(f'{save_dir}/plots',exist_ok=True)

    np.savetxt(f'{save_dir}/csv/combined_data.csv', combined_data, delimiter=",")

