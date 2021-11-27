import os
import shutil
import pandas as pd

r = 0.9
train_file = f"/nobackup/dyah_roopa/VAE_ColorMNIST_original/color_MNIST_1/r_{r}/train.csv"
test_file = f"/nobackup/dyah_roopa/VAE_ColorMNIST_original/color_MNIST_1/r_{r}/test.csv"

target_train = f"/nobackup/dyah_roopa/VAE_ColorMNIST_original/color_MNIST_1/train_{r}"
target_test = f"/nobackup/dyah_roopa/VAE_ColorMNIST_original/color_MNIST_1/test_{r}"

if not os.path.isdir(target_train):
    os.mkdir(target_train)

if not os.path.isdir(target_test):
    os.mkdir(target_test)


for _, row in pd.read_csv(train_file).iterrows():
    filename = row['file']
    target_dir = os.path.join(target_train, filename.split('.png',)[0][-1])
    
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)
    shutil.copy(filename, os.path.join(target_dir, filename.split('/')[-1]))

for _, row in pd.read_csv(test_file).iterrows():
    filename = row['file']
    digit = int(filename.split('.png',)[0][-1])
    if digit in [0, 1]:
        target_dir = os.path.join(target_test, "in_dist")
        if not os.path.isdir(target_dir):
            os.mkdir(target_dir)
        target_dir = os.path.join(target_dir, str(digit))
    else:
        target_dir = os.path.join(target_test, "ood")
        if not os.path.isdir(target_dir):
            os.mkdir(target_dir)
        target_dir = os.path.join(target_dir, str(digit))
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)
    shutil.copy(filename, os.path.join(target_dir, filename.split('/')[-1]))