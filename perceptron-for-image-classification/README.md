# Image classification with linear classifier (Perceptron) from scratch

## Introduction
This project will build a fully connected perceptron with 1 hidden layer to classify an image data set that have 50 classes.

## Data
You will need 3 txt files, each with first column as link to the image and second as its class value.
The txt files are named train.txt, val.txt, test.txt.

### Feature extration
Input features are extracted by Hu Moments, Haralick Texture and 3-color Histogram.
Using the "feature_extract.py" to transform the images into csv files that will later be passed to the model.
Specify bin size, fix image size in the main function.

### Perceptron
Sigmoid activation is appied in the hidden layer and softmax is appied in the output.
Regularization is also included, user can specify whether to add it by the parameter "lambda".
To build a model, make the class perceptron with following parameters:
* x_train: the feature of the image, read from the csv file from feature_extract.py
* y_train: one hot label, read from the class value of txt file and turn into one hot label by the function
* num_node: number of hidden nodes
* learning_rate: learning rate of the gradient descent
* minibatch_size: mini-batch size for training
* lambd: lambda, the rate for Regularization
* epoch: number of time for training
The optimization is done by stochastic gradient descent and Back-propagation.

### parameter turning
Parameter turning is done by the validation dataset.
First create the class "perceptron" with required parameters. Put 0 for lambd if no regularization
Then prepare lists of candidate parameters, the function "grid_search_perceptron" can take up to two list of parameters.
It also have a function to save the validation result into CSV (including parameters,accuracies, running time)

### performance evaluation
The file "randomforest.py" and "XGboost" will provide performances using random forest classifier and XGboost from sklearn package
Both classifiers are turned with the same turning methods (grid search with validation data set)
It return a final score with the test data set

## Technologies
The code is tested on Python 3.8

packages used in feature extraction:
* MinMaxScaler from sklearn.preprocessing
* mahotas for Haralick Texture
* cv2
* numpy
* pandas

packages used for random forest classifier and XGboost:
* RandomForestClassifier from sklearn.ensemble
* XGBClassifier from xgboost.sklearn

## Sources
My codes take reference from below:
* Feature extraction: https://github.com/87surendra/Random-Forest-Image-Classification-using-Python/blob/master/Random-Forest-Image-Classification-using-Python.py

* XGboost hyper-parameter tuning: https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/

* random forest hyper-parameter tuning: https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

* perceptron: https://stackabuse.com/creating-a-neural-network-from-scratch-in-python-multi-class-classification/
