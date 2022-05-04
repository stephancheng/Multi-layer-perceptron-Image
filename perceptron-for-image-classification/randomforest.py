# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 23:43:31 2021

@author: Stephan
"""

import numpy as np
import pandas as pd
from time import time
from sklearn.ensemble import RandomForestClassifier
#---------------------------------------------------------------------#
'''
# functions for loading feature set and class label
'''
def readdata(path):
    # read the txt file with image path and class label
    # 1st column is link to image, 2nd is class value
    df = pd.read_csv(path, sep= ' ', header=None, names=['img_link', 'class'])
    return df

def readfeature(path):
    # features are saved in csv format, read the csv file
    df = pd.read_csv(path, header=0, index_col = 0)
    return df

def import_feature():
    # read the txt file with image path and class label
    train_ref, val_ref = readdata("train.txt"), readdata("val.txt")
    # Features are already extracted, load it into panda dataframe
    x_train, x_val = readfeature('x_train.csv'), readfeature('x_val.csv')
    print("finish loaded training data")
    print("finish loaded validation data")
    y_train, y_val = train_ref['class'], val_ref['class']
    return x_train, x_val, y_train, y_val

#---------------------------------------------------------------------#
'''
# functions for doing parameter search for random forest
'''
class grid_search_rf():
    
    def __init__(self,max_features_list, n_estimators_list, x_train, y_train, x_val, y_val):
        self.max_features_list = max_features_list
        self.n_estimators_list = n_estimators_list
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.result = self.do_search()
        
    def do_search(self):
        result = pd.DataFrame(columns = ["max_features", "n_estimators", "accuracy", "elapsed"])
        for num_features in self.max_features_list:
            for num_tree in self.n_estimators_list:
                start = time() # record start time
                rf = RandomForestClassifier(max_features= num_features, n_estimators= num_tree)
                rf.fit(self.x_train, self.y_train)
                elapsed = time()-start # record end time
                random_accuracy = evaluate(rf, self.x_val, self.y_val)            
                cur_com = pd.DataFrame([[num_features,num_tree,random_accuracy, elapsed]], columns=["max_features", "n_estimators", "accuracy", "elapsed"])
                result = result.append(cur_com, ignore_index = True)
                del rf #free memory
        return result
    
    def save_csv(self):
        self.result.to_csv('rf_result.csv')

#---------------------------------------------------------------------#
'''
function to do model evaluation on accuracy
'''
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    accuracy = np.count_nonzero(predictions == test_labels)/test_labels.shape[0]*100
    return accuracy

#---------------------------------------------------------------------#
'''
# functions for doing evalution on test data
'''
def load_testdata():
    test_ref = readdata("test.txt")
    x_test = readfeature('x_test.csv')
    y_test = test_ref['class']
    print("finish loaded test data")
    return x_test, y_test

def testresult(num_features_best, num_tree_best, x_train, y_train):
    x_test, y_test = load_testdata()
    
    if num_features_best is None:
        rf_best = RandomForestClassifier()
    else:
        rf_best = RandomForestClassifier(max_features= num_features_best, n_estimators= num_tree_best)
    rf_best.fit(x_train, y_train)
    print("finish training")
    acc = evaluate(rf_best, x_test, y_test)
    print("Test result for {0} max_features and {1} n_estimators:{2} %".format(num_features_best,num_tree_best, acc))
    del rf_best # free memory 
    
#---------------------------------------------------------------------#

def main():    
    np.random.seed(42)    
    
    # ------------------quick run random forest without tuning---------#
    # Import features
    x_train, x_val, y_train, y_val = import_feature()
    
    # create and train base model
    base_model = RandomForestClassifier()
    base_model.fit(x_train, y_train)
    # Evaluate with validation data set
    base_accuracy = evaluate(base_model, x_val, y_val)
    print("Accuracy on validation data without tuning:", base_accuracy)
    del base_accuracy, base_model    
    # 20% accuracy on validation data without tuning
    #-------------------Validation test for parameter turining---------#
    # Create calidate list
    max_features_list= ['auto', 'sqrt', 'log2']
    n_estimators_list= np.arange(600, 1001, 200) 

    random_forest = grid_search_rf(max_features_list, n_estimators_list, x_train, y_train, x_val, y_val)
    print("Validation result:")
    print(random_forest.result)    
    # The best combinations are [auto, 1000] and [log 2, 1000]
    #-------------------Final evaluation for test data ----------------#
    
    test result without tuning
    testresult(None, None, x_train, y_train)
    # test result for auto and 1000    
    num_features_best, num_tree_best = "auto", 1000
    testresult(num_features_best, num_tree_best, x_train, y_train)
    # test result for log2 and 1000
    num_features_best, num_tree_best = "log2", 1000    
    testresult(num_features_best, num_tree_best, x_train, y_train)
    

    # 18.2 % without tuning
    # 17.1 with auto 1000
    # 19.3 with log2 1000
#-----------------------------------------------------------------#

if __name__ == "__main__":
    main()