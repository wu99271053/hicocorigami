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
            feature_file_name  = f"{i}_{window}_{length}_{itype}_{timestep}_feature.pt"
            feature_data_path = os.path.join(self.data_dir, feature_file_name)
            feature = torch.load(feature_data_path)

            # Concatenate the new data to the existing tensor
            self.x.extend(feature)
        
    def __len__(self):
            return len(self.x)

    def __getitem__(self, idx):
            return self.x[idx]

def preprocessing(data_dir=None,raw_dir=None,length=None,timestpe=0):
    nucleosome=f'{raw_dir}/GSE61888_nucs_normed.csv'
    fasta=f'{raw_dir}/saccer.fna'

    def one_hot_encoding(seq):
        # Create a dictionary of nucleotides
        nucleotides = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
        # Initialize an empty list to store the one-hot encoding
        one_hot_encoding = []
        # Iterate through the sequence and encode each nucleotide
        for nucleotide in seq:
            one_hot_encoding.append(nucleotides[nucleotide])
        # Convert the list of lists to a numpy array
        return np.transpose(np.array(one_hot_encoding))
        
    def expand_value(value, length, std_dev=20):
        """Expand a single value into a Gaussian distribution."""
        std_dev = 1 + 1 * np.log1p(abs(np.power(2, value)- 1))
    
    # Create a Gaussian distribution
        gaussian = norm.pdf(np.arange(length), loc=length // 2, scale=std_dev)
        gaussian /= gaussian.sum()  # Normalize the distribution
    
    # Spread the value across the Gaussian
        return np.round(np.power(2, value) * gaussian,2)

    def transform_dataframe(df, length):
        """Transform the DataFrame into a 3D array with the Gaussian expansion."""
        num_of_nucleo, num_of_histone = df.shape
        expanded_array = np.zeros((num_of_nucleo, num_of_histone, length))

        for i in range(num_of_nucleo):
            for j in range(num_of_histone):
                expanded_array[i, j, :] = expand_value(df.iloc[i, j], length)

        return expanded_array
        
    records = list(SeqIO.parse(fasta, "fasta"))
    node=pd.read_csv(nucleosome,header=None,usecols=[0,1,2], dtype=np.int32,skiprows=1)
    dna_matrix = np.zeros((node.shape[0],4,length), dtype=np.int8)

    for index,rows in node.iterrows():
        id,chr,bp=rows

        seq=records[chr-1].seq[bp-1-length//2:bp-1+length//2]
        seq=seq.upper()
        matrix=one_hot_encoding(seq)
        dna_matrix[id-1]=matrix
    
    df= pd.read_csv(nucleosome)
    nodedata_df = df.sort_values(by='nucleosome ID')

    # Extract the desired histone modification columns and append them to the dataframe
    usedhm_df = nodedata_df[[f'H2AK5ac_{timestpe}',f'H2AS129ph_{timestpe}',f'H3K14ac_{timestpe}',f'H3K18ac_{timestpe}',
                                    f'H3K23ac_{timestpe}',	f'H3K27ac_{timestpe}',f'H3K36me_{timestpe}',f'H3K36me2_{timestpe}',
                                    f'H3K36me3_{timestpe}',f'H3K4ac_{timestpe}',f'H3K4me_{timestpe}',f'H3K4me2_{timestpe}',f'H3K4me3_{timestpe}',
                                    f'H3K56ac_{timestpe}',f'H3K79me_{timestpe}',f'H3K79me3_{timestpe}',f'H3K9ac_{timestpe}',f'H3S10ph_{timestpe}',f'H4K12ac_{timestpe}',
                                    f'H4K16ac_{timestpe}',f'H4K20me_{timestpe}',f'H4K5ac_{timestpe}',f'H4K8ac_{timestpe}',f'H4R3me_{timestpe}',f'H4R3me2s_{timestpe}',f'Htz1_{timestpe}']]
        
    feature_matrix=transform_dataframe(usedhm_df, length)
    data_matrix=np.concatenate((dna_matrix,feature_matrix),axis=1)
    np.save(f'{data_dir}/{length}_{timestpe}.npy',data_matrix)
    
    return data_matrix


def chromosome_dataset(length,data_dir,itype,data_matrix,window_size,chromosome=None,timestep=0):

    data_matrix=data_matrix
    window_size=window_size


    def get_chromosome_range(chromosome=chromosome):
        all_ranges = [
            (1, 1274), (1275, 5761), (5762, 7477), (7478, 15987),
            (15988, 19141), (19142, 20638), (20639, 26651),
            (26652, 29757), (29758, 32184), (32185, 36330),
            (36331, 40029), (40030, 45654), (45655, 50769),
            (50770, 55112), (55113, 61130), (66131, 66360)
        ]
        
        range = all_ranges[chromosome - 1]

        
        return range
    
    def generate_matrix(chr,window_size=window_size):
        chrom_range = get_chromosome_range(chr)
        lower_bound, upper_bound = chrom_range
        max_id = upper_bound - window_size

        # subset_df = df[(df['id1'] >= lower_bound) & (df['id1'] <= upper_bound) &
        #            (df['id2'] >= lower_bound) & (df['id2'] <= upper_bound)]
        # value_dict = {(row['id1'], row['id2']): row['value'] for index, row in subset_df.iterrows()}

        
        feature_matrices = []
        for start_id in tqdm(range(lower_bound,max_id+1)):

            feature_matrix_resize = data_matrix[start_id - 1 : start_id - 1 + window_size].reshape(30, -1).astype(np.float16)
            #fake_window_size=60

            feature_matrices.append(feature_matrix_resize)

        torch.save(feature_matrices, f'{data_dir}/processed/{chr}_{window_size}_{length}_{itype}_{timestep}_feature.pt')


    
    for i in tqdm(range(1,17)):
        generate_matrix(i)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import argparse
    parser = argparse.ArgumentParser(description='C.Origami_like Training Module.')
    parser.add_argument('--raw_dir',required=True,
                        help='Path to the raw data')
    parser.add_argument('--window',type=int,required=True,
                        help='size of heatmap')
    parser.add_argument('--length',default=128,type=int,
                        help='length of Nucleosomal DNA')
    parser.add_argument('--i_type',required=True,
                        help='interaction type')
    parser.add_argument('--data_dir',required=True,
                        help='processed data and saved checkpoint')
    parser.add_argument('--model_dir',required=True,
                        help='processed data and saved checkpoint')
    parser.add_argument('--timestep',default=4,type=int,
                        help='length of Nucleosomal DNA')
    parser.add_argument('--val_chr', dest='val_chr', default=1,
                            type=int,
                            help='Random seed for training')
    
    
    args = parser.parse_args()

    # data_matrix=np.load(f'{args.data_dir}/{args.length}.npy')
    # df=pd.read_csv(f'{args.raw_dir}/histone_modification.csv', header=None, usecols=[0, 1],skiprows=1)
    # selected_id = df[df.iloc[:, 1] == -1].iloc[:, 0].tolist()

    if not os.path.exists(args.data_dir):
    # Create the directory if it does not exist
        os.makedirs(args.data_dir)

    data_matrix=preprocessing(data_dir=args.data_dir,raw_dir=args.raw_dir,length=args.length,timestpe=args.timestep)

    chromosome_dataset(args.length,data_dir=args.data_dir,itype=args.i_type,data_matrix=data_matrix,window_size=args.window,timestep=args.timestep)

    
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_dir=f'{args.data_dir}/result/{args.val_chr}'
    if not os.path.exists(save_dir):
    # Create the directory if it does not exist
        os.makedirs(save_dir)
    checkpointpath=f'{args.itype}/{args.window}/{args.data_dir}/checkpoint_{args.val_chr}/models/{args.val_chr}.ckpt'



        
    infer_dataset = inferenceDataset(data_dir=f'{args.data_dir}/processed', window=args.window, length=args.length, chr=args.val_chr, itype=args.itype,timestep=args.timestep)
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

    np.savetxt(f'{save_dir}/computed_outputs.csv', np.concatenate(outputlist), delimiter=",")

    outputlist=np.concatenate(outputlist)
    outputlist_reshaped = outputlist.reshape(-1, args.window, args.window)
    for i in range(len(outputlist_reshaped)):
        prediction = outputlist.reshape(-1, args.window, args.window)[i]

        fig, ax1 = plt.subplots(1, 1, figsize=(18, 6))

        # Plot the first heatmap
        cax1 = ax1.imshow(prediction, cmap='cool', interpolation='nearest')
        fig.colorbar(cax1, ax=ax1)
        ax1.set_title('Predicted')


        # Save the plot to the specified directory
        plt.savefig(os.path.join(save_dir, f'heatmap_{i}.png'))
        plt.close(fig)  # Close the figure to free memory



