import torch 
from Bio import SeqIO
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import norm
import argparse
from skimage.transform import resize
import os 





def cleaningup(data_dir=None,raw_dir=None,window=None,i_type=None):
    nucleosome=f'{raw_dir}/histone_modification.csv'
    interaction=f'{raw_dir}/{i_type}_WT_G1.csv'

    def removing_repeated():
        df=pd.read_csv(nucleosome, header=None, usecols=[0, 1],skiprows=1)
        selected_ids = df[df.iloc[:, 1] == -1].iloc[:, 0].tolist()
        new_df = pd.read_csv(interaction, header=None, dtype=np.int32)
        new_df[0] += 1
        new_df[1] += 1
        filtered_df = new_df[~(new_df.iloc[:, 0].isin(selected_ids) | new_df.iloc[:, 1].isin(selected_ids))]

    # Save the filtered DataFrame to a new CSV
        return filtered_df,selected_ids
    
    def removing_duplicated(df):
        df['key'] = df.apply(lambda row: tuple(sorted([row[0], row[1]])), axis=1)

    # Sort by 'key' and keep the first occurrence
        df[[0, 1]] = df.apply(lambda row: sorted([row[0], row[1]]), axis=1, result_type='expand')
        df_sorted = df.sort_values(by='key').drop_duplicates(subset='key', keep='first')
        df_droped=df_sorted.drop(columns='key')
        df_filtered = df_droped[df_droped.iloc[:, 1] - df_droped.iloc[:, 0] <= window]
        df_filtered.to_csv(f'{data_dir}/{i_type}_{window}.csv',index=False,header=None)

        #return df_droped
    

    # def filling_zeros(dataframe,window=window):
    #     df = dataframe.iloc[:, :2]
        
    #     # Create a set of tuples from the DataFrame for existing edges
    #     existing_edges = set(tuple(x) for x in df.to_numpy())
        
    #     missing_edges = []
    #     for src in tqdm(np.unique(df[0].values)):  # Note: This was changed to df[0]
    #         # Create the desired set of tuples based on window size
    #         desired_edges = {(src, src+i) for i in range(0, window+1)}
            
    #         # Identify the missing edges
    #         current_missing_edges = desired_edges - existing_edges
            
    #         for edge in current_missing_edges:
    #             #missing_edges.append([edge[0], edge[1], 0, 0, 0, 0])
    #             missing_edges.append([edge[0], edge[1], 0])#only one interaction

    #     missing_df = pd.DataFrame(missing_edges)
        
    #     # Combine original and missing data, then sort and save
    #     new_df = dataframe
    #     combined_data = pd.concat([new_df, missing_df], axis=0)
    #     combined_data = combined_data.sort_values(by=[0, 1])
    #     filtered_df = combined_data[combined_data.iloc[:, 1] - combined_data.iloc[:, 0] <= window]

    #     filtered_df.to_csv(f'{data_dir}/{i_type}_{window}.csv',index=False,header=None)
    
    df1,selected_id=removing_repeated()
    #df2=removing_duplicated(df1)
    removing_duplicated(df1)
    #filling_zeros(df2,window=window)
    return selected_id




def preprocessing(data_dir=None,raw_dir=None,length=None):
    nucleosome=f'{raw_dir}/histone_modification.csv'
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
    node=pd.read_csv(nucleosome,header=None,usecols=[0,2,3], dtype=np.int32,skiprows=1)
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
    usedhm_df = nodedata_df[['H2AK5ac','H2AS129ph','H3K14ac','H3K18ac',
                                    'H3K23ac',	'H3K27ac','H3K36me','H3K36me2',
                                    'H3K36me3','H3K4ac','H3K4me','H3K4me2','H3K4me3',
                                    'H3K56ac','H3K79me','H3K79me3','H3K9ac','H3S10ph','H4K12ac',
                                    'H4K16ac','H4K20me','H4K5ac','H4K8ac','H4R3me','H4R3me2s','Htz1']]
        
    feature_matrix=transform_dataframe(usedhm_df, length)
    data_matrix=np.concatenate((dna_matrix,feature_matrix),axis=1)
    np.save(f'{data_dir}/{length}.npy',data_matrix)
    
    return data_matrix


def chromosome_dataset(length,data_dir,itype,data_matrix,selected_id,window_size,chromosome=None):
    df=pd.read_csv(f'{data_dir}/{itype}_{window_size}.csv',header=None,dtype=np.int32)
    df.columns = ['id1', 'id2', 'value']
    value_dict = {(row['id1'], row['id2']): row['value'] for index, row in df.iterrows()}

    data_matrix=data_matrix
    selected_id=selected_id
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
        contact_matrices = []
        for start_id in tqdm(range(lower_bound,max_id+1)):
            if start_id in selected_id:
                continue

            feature_matrix_resize = data_matrix[start_id - 1 : start_id - 1 + window_size].reshape(30, -1)
            fake_window_size=200


            contact_matrix = np.zeros((fake_window_size, fake_window_size))

            for i in range(fake_window_size):
                for j in range(i,fake_window_size):
                    id1 = start_id + i
                    id2 = start_id + j

                    # Fetch value from df if it exists
                    value = value_dict.get((id1, id2), 0)
                    #value = subset_df.loc[(subset_df['id1'] == id1) & (subset_df['id2'] == id2), 'value']
                    contact_matrix[i, j] = contact_matrix[j, i]=value
            
            contact_matrix_resize=resize(contact_matrix,(window_size,window_size),anti_aliasing=True)
            
            contact_matrices.append(contact_matrix_resize)
            feature_matrices.append(feature_matrix_resize)

        torch.save(feature_matrices, f'{data_dir}/{chr}_{window_size}_{length}_{itype}_feature.pt')
        torch.save(contact_matrices, f'{data_dir}/{chr}_{window_size}_{length}_{itype}_contact.pt')


    
    for i in tqdm(range(1,17)):
        generate_matrix(i)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='C.Origami_like Training Module.')
    parser.add_argument('--raw_dir', default='../../Desktop/raw',
                        help='Path to the raw data')
    parser.add_argument('--window', default=256,type=int,
                        help='size of heatmap')
    parser.add_argument('--length',default=128,type=int,
                        help='length of Nucleosomal DNA')
    parser.add_argument('--i_type',default='Outward',
                        help='interaction type')
    parser.add_argument('--data_dir',default='../../Desktop/processed',
                        help='processed data and saved checkpoint')
    
    args = parser.parse_args()

    if os.path.exists(f'{args.data_dir}/{args.i_type}_{args.length}.csv') and os.path.exists(f'{args.data_dir}/{args.length}.csv'):
        data_matrix=np.load(f'{args.data_dir}/{args.length}.npy')
        df=pd.read_csv(f'{args.raw_dir}/histone_modification.csv', header=None, usecols=[0, 1],skiprows=1)
        selected_ids = df[df.iloc[:, 1] == -1].iloc[:, 0].tolist()
        selected_id=cleaningup(data_dir=args.data_dir,raw_dir=args.raw_dir,window=args.window,i_type=args.i_type)
        data_matrix=preprocessing(data_dir=args.data_dir,raw_dir=args.raw_dir,length=args.length)
    
    else: 
        selected_id=cleaningup(data_dir=args.data_dir,raw_dir=args.raw_dir,window=args.window,i_type=args.i_type)
        data_matrix=preprocessing(data_dir=args.data_dir,raw_dir=args.raw_dir,length=args.length)

    chromosome_dataset(args.length,data_dir=args.data_dir,itype=args.i_type,data_matrix=data_matrix,selected_id=selected_id,window_size=args.window)