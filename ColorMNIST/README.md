# Colored MNIST

 

### Instructions

1. Generate Colored MNIST train data, test data, and confound test data: ```python generate_colored_mnist.py```

2. Depending on the correlation factor chosen to run, replace the value in constants.py as r = x where x is the value.

2. Train a standard Conditional VAE and create a new Intervened VAE train data: ```python train_vae.py```

3. Evaluate a simple classifier using different dataset: ```python evaluate_cnn_classification.py```

 

Please change the directory path inside the code to your own environment.
