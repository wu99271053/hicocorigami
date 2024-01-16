import os
import torch
from torch.utils.data import Dataset
import os
import torch
import hicomodel
from tqdm import tqdm
import numpy as np
import torch 
from Bio import SeqIO
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import norm
import argparse
from skimage.transform import resize
import os 


class inferenceDataset(Dataset):
    def __init__(self, data_dir, window, length, chr, itype,timestep):
        self.data_dir = data_dir
        self.window = window
        self.length = length
        self.itype = itype
        self.timestep=timestep

        self.x = []
        self.y = []

        # Ensure chr is a list even if it's a single value
        if not isinstance(chr, list):
            chr = [chr]
        for i in chr:
            feature_file_name  = f"{i}_{window}_{length}_{timestep}_feature.pt"
            feature_data_path = os.path.join(self.data_dir, feature_file_name)
            feature = torch.load(feature_data_path)

            # Concatenate the new data to the existing tensor
            self.x.extend(feature)
        
    def __len__(self):
            return len(self.x)

    def __getitem__(self, idx):
            return self.x[idx]

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import argparse
    parser = argparse.ArgumentParser(description='C.Origami_like Training Module.')
    parser.add_argument('--window',type=int,required=True,
                        help='size of heatmap')
    parser.add_argument('--length',default=128,type=int,
                        help='length of Nucleosomal DNA')
    parser.add_argument('--itype',required=True,
                        help='interaction type')
    parser.add_argument('--data_dir',required=True,
                        help='processed data and saved checkpoint')
    parser.add_argument('--val_chr', dest='val_chr', default=1,
                            type=int,
                            help='Random seed for training')
    parser.add_argument('--timestep', default=0,
                            type=int,
                            help='Random seed for training')
    
    parser.add_argument('--gaussian',action='store_true',
                        help='processed data and saved checkpoint')
    
    args = parser.parse_args()



    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.gaussian:
        save_dir=f'{args.itype}/gaussian/{args.window}/checkpoint_{args.val_chr}/inference_result'
        if not os.path.exists(save_dir):
        # Create the directory if it does not exist
            os.makedirs(save_dir)
        checkpointpath=f'{args.itype}/gaussian/{args.window}/checkpoint_{args.val_chr}/models/{args.val_chr}.ckpt'
    else:
        save_dir=f'{args.itype}/notransform/{args.window}/checkpoint_{args.val_chr}/inference_result'
        if not os.path.exists(save_dir):
        # Create the directory if it does not exist
            os.makedirs(save_dir)
        checkpointpath=f'{args.itype}/notransform/{args.window}/checkpoint_{args.val_chr}/models/{args.val_chr}.ckpt'
        
    infer_dataset = inferenceDataset(data_dir=args.data_dir, window=args.window, length=args.length, chr=args.val_chr, itype=args.itype,timestep=args.timestep)
    infer_loader=torch.utils.data.DataLoader(infer_dataset,batch_size=1,shuffle=False,drop_last=True)

    model = hicomodel.ConvTransModel(True,args.window)

    checkpoint = torch.load(checkpointpath, map_location=device)
    model_weights = checkpoint['state_dict']

    # Edit keys
    for key in list(model_weights):
        model_weights[key.replace('model.', '')] = model_weights.pop(key)
    model.load_state_dict(model_weights)

    model.eval()
    model.to(device)
    outputlist=[]
    with torch.no_grad():
        for i in tqdm(infer_loader):
            inputs = i
            inputs = inputs.transpose(1, 2).contiguous()
            inputs = inputs.float()
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputlist.append(outputs.cpu().view(-1).numpy())

    np.savetxt(f'{save_dir}/prediction_{args.timestep}.csv', np.round(np.concatenate(outputlist),2), delimiter=",")