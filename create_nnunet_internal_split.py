import os
from sklearn.model_selection import GroupKFold
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import save_json

sourcepath='/home/klinfys/Desktop/nnUNet_raw/Dataset222_T2star/imagesTr'
destinationpath='/home/klinfys/Desktop/nnUNet_preprocessed/Dataset222_T2star'

files = [f[:-12] for f in os.listdir(sourcepath) if f.endswith('_0000.nii.gz')]
groups = np.array([f[:2] for f in files])
files = np.array(files)
group_kfold = GroupKFold(n_splits=5)

split = []
for i, (train_index, test_index) in enumerate(group_kfold.split(files, groups=groups)):
    print(i, len(files[train_index]), len(files[test_index]), len(np.unique(groups[train_index])), len(np.unique(groups[test_index])), np.unique(groups[test_index]))
    split.append({
        'train': files[train_index].tolist(), 
        'val': files[test_index].tolist()
    })

os.makedirs(destinationpath, exist_ok=True)


destination_file_path = os.path.join(destinationpath, 'splits_final.json')

save_json(split, destination_file_path)

