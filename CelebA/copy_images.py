import os
import shutil
import pandas as pd
import constants

r = constants.r
train_file = f"/nobackup/dyah_roopa/temp/Spurious_OOD/datasets/celebA/train.csv"
test_file = f"/nobackup/dyah_roopa/temp/Spurious_OOD/datasets/celebA/test.csv"

target_train = f"/nobackup/dyah_roopa/CelebA/train_{r}"
target_test = f"/nobackup/dyah_roopa/CelebA/test_{r}"

if not os.path.isdir(target_train):
    os.mkdir(target_train)

if not os.path.isdir(target_test):
    os.mkdir(target_test)

# df_data = pd.read_csv(os.path.join(celebA_dir, "celebA_split.csv"))

# for _, row in pd.read_csv(train_file).iterrows():
#     filename = row['file']
#     target_dir = os.path.join(target_train,str(row['label']))
#     # str(df_data.loc[df_data["image_id"] == image_id]["Gray_Hair"].iloc[0])
    
#     if not os.path.isdir(target_dir):
#         os.mkdir(target_dir)
#     shutil.copy(filename, os.path.join(target_dir, filename.split('/')[-1]))

for _, row in pd.read_csv(test_file).iterrows():
    filename = row['file']
    # digit = int(filename.split('.png',)[0][-1])
    label = int(row['label'])
    if label in [0, 1]:
        target_dir = os.path.join(target_test, "in_dist")
        if not os.path.isdir(target_dir):
            os.mkdir(target_dir)
        target_dir = os.path.join(target_dir, str(label))
    else:
        target_dir = os.path.join(target_test, "spurious_ood")
        if not os.path.isdir(target_dir):
            os.mkdir(target_dir)
        # target_dir = os.path.join(target_dir, str(digit))
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)
    shutil.copy(filename, os.path.join(target_dir, filename.split('/')[-1]))