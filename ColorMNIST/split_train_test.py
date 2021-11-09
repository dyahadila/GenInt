import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split

colormnist_dir = "/nobackup/dyah_roopa/VAE_ColorMNIST_upsampled/color_MNIST_1/r_0.25"
label_file = "targets.npy"

in_dist = {'file': [], 'label': []}
ood = {'file': [], 'label': []}

images = [f_ for f_ in os.listdir(colormnist_dir) if '.png' in f_]
for i, file_ in enumerate(images):
    label = int(file_.split('.')[0][-1])
    if label == 0 or label == 1:
        in_dist['file'].append(os.path.join(colormnist_dir, file_))
        in_dist['label'].append(label)
    elif label == 5 or label == 6 or label == 7 or label == 8 or label == 9:
        ood['file'].append(os.path.join(colormnist_dir, file_))
        ood['label'].append(-1)
in_dist = pd.DataFrame(in_dist)
in_dist.to_csv(os.path.join(colormnist_dir, 'in_dist.csv'))
ood = pd.DataFrame(ood)
ood.to_csv(os.path.join(colormnist_dir, 'ood.csv'))

train_id, test_id = train_test_split(in_dist, test_size=0.2, random_state=42)
train_id.to_csv(os.path.join(colormnist_dir, 'train.csv'))


ood_test = ood.sample(n=test_id.shape[0])
test_files = pd.concat([test_id, ood_test])
test_files.to_csv(os.path.join(colormnist_dir, 'test.csv'))