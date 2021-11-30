import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import constants

celebA_dir = f"/nobackup/dyah_roopa/temp/Spurious_OOD/datasets/celebA"

train = {'file': [], 'label': []}
indist_test = {'file': [], 'label': []}
ood = {'file': [], 'label': []}

df_data = pd.read_csv(os.path.join(celebA_dir, "celebA_split.csv"))

images_train = [f_ for f_ in os.listdir(os.path.join(celebA_dir, "train_sq")) if '.png' in f_]
images_indist_test = [f_ for f_ in os.listdir(os.path.join(celebA_dir, "indist_test_sq")) if '.png' in f_]
# images = images_train + images_indist_test
for i, file_ in enumerate(images_train):
    image_id = file_.split("_")[-1][:-4] + ".jpg"
    label = df_data.loc[df_data["image_id"] == image_id]["Gray_Hair"].iloc[0]
    train['file'].append(os.path.join(celebA_dir, "train_sq", file_))
    train['label'].append(label)

for i, file_ in enumerate(images_indist_test):
    image_id = file_.split("_")[-1][:-4] + ".jpg"
    label = df_data.loc[df_data["image_id"] == image_id]["Gray_Hair"].iloc[0]
    indist_test['file'].append(os.path.join(celebA_dir, "indist_test_sq", file_))
    indist_test['label'].append(label)
spurious_images = [f_ for f_ in os.listdir(os.path.join(celebA_dir, "spurious_ood_sq")) if '.png' in f_]
for i, file_ in enumerate(spurious_images):
    image_id = file_.split("_")[-1][:-4] + ".jpg"
    ood['file'].append(os.path.join(celebA_dir, "spurious_ood_sq", file_))
    ood['label'].append(-1)


train = pd.DataFrame(train)
indist_test = pd.DataFrame(indist_test)
ood = pd.DataFrame(ood)
train.to_csv(os.path.join(celebA_dir, 'train.csv'))


ood_test = ood.sample(n=indist_test.shape[0])
test_files = pd.concat([indist_test, ood_test])
test_files.to_csv(os.path.join(celebA_dir, 'test.csv'))