# Image classification with MLP from scratch

## Introduction
This project will build a fully connected perceptron with 0 to 2 hidden layers to classify mini ImageNet data set that have 50 classes. For the MLP, only numpy, pandas and time is used.

### Feature extraction
Input features are extracted by Hu Moments, Haralick Texture and 3-color Histogram.
Using the feature_extract.py to transform the images into csv files that will later be passed to the model
You will need txt files with first column as link to the image and second as its class value.
The txt files are named train.txt, val.txt, test.txt.
Spcify bin size, fix image size in the main function.

### Model
To build a model, make the class MLP with following parameters:
MLP(x_train, y_train, hidden_layers, learning_rate, minibatch_size, lambd, epoch, act_function)
* x_train: the feature of the image, read from the csv file from feature_extract.py
* y_train: read from the class value of txt file
* hidden_layers: list of nodes in each hidden layer, for example, [80, 60] will create 80 nodes for 1st layer and 60 nodes for 2nd
* learning_rate: learning rate of the gradient descent
* minibatch_size: mini-batch size for training
* lambd: lambda, the rate for Regularization
* epoch: number of time for training
* act_function: list of activation in each hidden layer, for example, ["sigmoid", "relu"] will apply sigmoid for 1st layer and relu for 2nd

#### training
Use the function ".train()" of the class MLP for training
The function will apply minibatch gradient descent and Back-propagation to update weights.

After training the model, you may use the function ".save_result(file_name)" of the class MLP for saving the result of training loss and accuracy. Specify a sub-file name for this function.

#### backpropagation
I have built the classes for the operations, including dot product, plus, sigmoid, relu, and softmax. All of the classes have function for forward and backward propagation. Local gadient can be calculated by the ".backward()" functions.

### parameter tuning
Parameter tuning is done by the validation dataset.
After creating the model (MLP class), Prepare lists of candidate parameter, the function "grid_search_perceptron" can take up to two list of parameters. It will automatically run the ".train()" function and will return the accuracy of the selected candidate parameters.
It also have a function to save the validation result into CSV (including parameters,accuracy, running time)

### performance evaluation
After the parameter tuning, select and build the model with the best parameters. Train the model and test with the testing dataset

Use the function ".evaluate()" to get the accuracy. It will run the ".predict()" function automatically, which will return list of top 1 predicition and list of top 5 predicition of the test feature. The function will then compare the prediction with the true class label.

## Technologies
The code is tested on Python 3.8

packages used in feature extraction:
* MinMaxScaler from sklearn.preprocessing
* mahotas for Haralick Texture
* cv2
* numpy
* pandas

packages used in MLP:
* numpy
* pandas
* time

## Sources
My codes take reference from below:
* Feature extraction: https://github.com/87surendra/Random-Forest-Image-Classification-using-Python/blob/master/Random-Forest-Image-Classification-using-Python.py

* perceptron: https://stackabuse.com/creating-a-neural-network-from-scratch-in-python-multi-class-classification/

* MLP: https://github.com/musikalkemist/DeepLearningForAudioWithPython/tree/master/8-%20Training%20a%20neural%20network:%20Implementing%20back%20propagation%20from%20scratch/code

* MLP: https://www.brilliantcode.net/1381/backpropagation-2-forward-pass-backward-pass/
