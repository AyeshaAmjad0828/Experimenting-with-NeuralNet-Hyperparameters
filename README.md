# Experimenting-with-NeuralNet-Hyperparameters

 Worked on three experiments:
1. Using a vanilla NN to process MNIST data in MNIST-Experimenting_Hyperparameter2.py
   - A feedforward network with no backprop
   - Training model has L1/L2 regularization and gradient clipping
   - Experiments with different activation functions {nn.Sigmoid(), nn.ReLU(), nn.Tanh(), nn.SiLU()}
  
2. Using a pretrained VGG16 network to process CIFAR10 data in CIFAR10-Experimenting_Hyperparameters.py
   - A pretained VGG16 network without any additions
   - Training model has L1/L2 regularization and gradient clipping
   - Experiments with different optimizers
  
3. Using VGG network to process to CIFAR10 and MNIST data
   - Explicitely defined layers of a VGG network
   - Added batchnorm and drop out layers.
  
 Here is a link to colab with executed code: https://colab.research.google.com/drive/1GhxnEjGDYAxbqZMs1YGrQp3xr08wUQnb?usp=sharing
