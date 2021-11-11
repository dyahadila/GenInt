import os
import shutil
import pandas as pd

train_file = "/nobackup/dyah_roopa/VAE_ColorMNIST_original/color_MNIST_1/r_0.25/train.csv"
test_file = "/nobackup/dyah_roopa/VAE_ColorMNIST_original/color_MNIST_1/r_0.25/test.csv"

target_train = "/nobackup/dyah_roopa/VAE_ColorMNIST_original/color_MNIST_1/train_0.25"
target_test = "/nobackup/dyah_roopa/VAE_ColorMNIST_original/color_MNIST_1/test_0.25"

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
    target_dir = os.path.join(target_test, filename.split('.png',)[0][-1])
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)
    shutil.copy(filename, os.path.join(target_dir, filename.split('/')[-1]))