# Handwritten Digit Recognition: Neural Networks


We will implement a neural netowrk to recognize handwritten digits (from 0 to 9). We will use a dataset that contains 5000 training examples of handwritten digits; this dataset is a subset of the MNIST handwritten digit dataset.

I will adapt my classifier from the fourth exercise from Andrew Ng’s Machine Learning Course on Coursera.

# Running the Project 

- Make sure you have MATLAB or Octave installed. 
- Clone the project to your local machine. 
- Run digitrecognition_nn.m. For a guided implementation, you can instead run the live script HandwrittenDigitRecognition_NeuralNetwork.mlx. 

# Project Details

I will implement a neural netowork to recognize handwritten digits (from 0 to 9). We will use a dataset that contains 5000 training examples of handwritten digits; this dataset is a subset of the MNIST handwritten digit dataset. First, we load in our data (mnistdata.mat).  Note that MATLAB indexing has no zero index; so the digit '0' is labeled as '10', while the numbers '1' to '9' are labeled as expected.

Our neural network will have 3 layers: an input layer, a hidden layer, and an output layer. Our inputs are pixel values from our MNIST digit images; our images are 20 x 20, which will give us 400 input layer units (plus one extra unit for our bias unit, which always outputs +1).

First we will implement feedforward propogation for our neural network, which will predict our neural network's parameter values. We implement our cost function and gradient for our neural network. For now, we set our regularization parameter lambda to 1. We will implement backpropagation to compute the gradient for our neural network cost function. When we train our neural networks, we should randomly initaize our parameters for symmetry breaking. 

I will set our neural network to run with 200 iterations; this can be changed as you see fit!

We will predict the class for which our neural network classifier outputs the highest probability, and return that class label. In our case, it will return which number we think is contained in the image!

Our training set accuracy for our neural network classifier is more than 99.25%!
