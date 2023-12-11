import numpy as np
from skimage.transform import resize


data=np.load('../../Desktop/hic_matrix/chr1.npz')
print(data.files)

print(data[data.files[0]])