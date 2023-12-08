import pyBigWig
import pandas as pd
import numpy as np
import torch 


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

htz1=pyBigWig.open('../../Downloads/GSM1516554_htz1_tp1_0.bw')
input=pyBigWig.open('../../Downloads/GSM1516603_input3.2_tp1_0.bw')

print(htz1.chroms('chrI'))
print(htz1.stats('chrI', 617-50, 617+50))
print(input.stats('chrI',617-50,617+50))
