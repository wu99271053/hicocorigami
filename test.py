import pyBigWig
import pandas as pd
import numpy as np
import torch 
from skimage.transform import resize
import matplotlib.pyplot as plt


# # df= pd.read_csv('../../Desktop/raw/histone_modification.csv')
# # nodedata_df = df.sort_values(by='nucleosome ID')

# #         # Extract the desired histone modification columns and append them to the dataframe
# # usedhm_df = nodedata_df[['H2AK5ac','H2AS129ph','H3K14ac','H3K18ac',
# #                                      'H3K23ac',	'H3K27ac','H3K36me','H3K36me2',
# #                                      'H3K36me3','H3K4ac','H3K4me','H3K4me2','H3K4me3',
# #                                      'H3K56ac','H3K79me','H3K79me3','H3K9ac','H3S10ph','H4K12ac',
# #                                      'H4K16ac','H4K20me','H4K5ac','H4K8ac','H4R3me','H4R3me2s','Htz1']]


# # print(usedhm_df.sha
# # 
# # df=pd.read_csv(f'../../My Drive/corigamidata/Outward_16.csv',header=None,dtype=np.int32)
# # df.columns = ['id1', 'id2', 'value']
# # lower_bound = 1
# # upper_bound = 1274

# # subset_df = df[(df['id1'] >= lower_bound) & (df['id1'] <= upper_bound) &
# #                    (df['id2'] >= lower_bound) & (df['id2'] <= upper_bound)]
# # print(df.shape,subset_df.shape)

# # oridata=torch.load('../../My Drive/jokedata/contact_matrix.pt')
# # print(oridata[1])
# # seconddata=torch.load('../../My Drive/corigamidata/1_16_128_DNA_Outward_contact_matrix.pt')
# thirddata=torch.load('../../My Drive/corigami_hmdna/1_16_128_hmDNA_Outward_feature_matrix.pt')
# # print(seconddata.reshape(-1,16,16)[1])
# print(thirddata[0][28])
# # print(len(oridata),seconddata.shape,thirddata.shape)

# htz1=pyBigWig.open('../../Downloads/GSM1516554_htz1_tp1_0.bw')
# input=pyBigWig.open('../../Downloads/GSM1516603_input3.2_tp1_0.bw')

# print(htz1.chroms('chrI'))
# print(htz1.stats('chrI', 617-50, 617+50))
# print(input.stats('chrI',617-50,617+50))
class Feature():

    def __init__(self, **kwargs):
        self.load(**kwargs)
    
    def load(self):
        raise Exception('Not implemented')

    def get(self):
        raise Exception('Not implemented')

    def __len__(self):
        raise Exception('Not implemented')

class HiCFeature(Feature):

    def load(self, path = None):
        self.hic = self.load_hic(path)

    def get(self, start, window = 2097152, res = 10000):
        start_bin = int(start / res)
        range_bin = int(window / res)
        end_bin = start_bin + range_bin
        hic_mat = self.diag_to_mat(self.hic, start_bin, end_bin)
        return hic_mat

    def load_hic(self, path):
        print(f'Reading Hi-C: {path}')
        return dict(np.load(path))

    def diag_to_mat(self, ori_load, start, end):
        '''
        Only accessing 256 x 256 region max, two loops are okay
        '''
        square_len = end - start
        diag_load = {}
        for diag_i in range(square_len):
            diag_load[str(diag_i)] = ori_load[str(diag_i)][start : start + square_len - diag_i]
            diag_load[str(-diag_i)] = ori_load[str(-diag_i)][start : start + square_len - diag_i]
        start -= start
        end -= start

        diag_region = []
        for diag_i in range(square_len):
            diag_line = []
            for line_i in range(-1 * diag_i, -1 * diag_i + square_len):
                if line_i < 0:
                    diag_line.append(diag_load[str(line_i)][start + line_i + diag_i])
                else:
                    diag_line.append(diag_load[str(line_i)][start + diag_i])
            diag_region.append(diag_line)
        diag_region = np.array(diag_region).reshape(square_len, square_len)
        return diag_region

    def __len__(self):
        return len(self.hic['0'])

# data=np.load('../../Desktop/hic_matrix/chr1.npz')
# #print(data.files)
# print(data['0'][2000:3000])

# data=HiCFeature(path='../../Desktop/hic_matrix/chr1.npz')
# mat=data.get(208*10000)
# mat_resize = resize(mat, (256,256), anti_aliasing=True)
import pandas as pd
import numpy as np

# Read the CSV file
csv_file = '../../Desktop/raw/Outward_WT_G1.csv'  # Update with your CSV file path
df = pd.read_csv(csv_file, header=None)  # Assuming no header in the CSV
df.columns = ['id1', 'id2', 'value']  # Rename columns

# Define window size
window_size = 256  # Update this with your desired window size

# Create a zero-filled numpy array
mat = np.zeros((window_size, window_size))

# Iterate through the DataFrame and fill the heatmap
for index, row in df.iterrows():
    i, j = int(row['id1']), int(row['id2'])
    if i < window_size and j < window_size:
        mat[i, j] = row['value']

# Your heatmap is now filled and ready for use
mat_resize=resize(mat,(256,256),anti_aliasing=True)

plt.subplot(1, 2, 1)
plt.imshow(mat, cmap='cool', interpolation='nearest')
plt.title('Original Heatmap (209x209)')
plt.colorbar()

# Resized heatmap
plt.subplot(1, 2, 2)
plt.imshow(mat_resize, cmap='cool', interpolation='nearest')
plt.title('Resized Heatmap (256x256)')
plt.colorbar()

plt.tight_layout()
plt.show()