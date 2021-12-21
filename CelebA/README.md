# CelebA
 

### Instructions

1. The data is already present in /nobackup/dyah_roopa/CelebA

2. Depending on the correlation factor chosen to run, replace the value in constants.py as r = x where x is the value.

3. Train a standard Conditional VAE and create a new Intervened VAE train data: ```python train_vae.py```

4. Evaluate a simple classifier using different dataset: ```python evaluate_resnet_classification.py```

Please change the directory path inside the code to your own environment.
