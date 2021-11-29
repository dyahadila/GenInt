import pandas as pd
import shutil
import os

colormnist_dir = "/nobackup/dyah_roopa/VAE_ColorMNIST_upsampled/color_MNIST_1/r_0.25"
sample_dir = "/nobackup/dyah_roopa/temp/Spurious_OOD/datasets/SAMPLE"
train_samples = pd.read_csv(os.path.join(colormnist_dir, "test.csv")).sample(n=100, random_state=1)
for _, samples in train_samples.iterrows():
    shutil.copy(samples['file'], os.path.join(sample_dir, samples['file'].split('/')[-1]))