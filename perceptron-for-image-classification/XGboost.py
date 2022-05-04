# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 00:21:53 2021

@author: Stephan
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 23:43:31 2021

@author: Stephan
"""

import numpy as np
import pandas as pd
from xgboost.sklearn import XGBClassifier
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
function to do model evaluation on accuracy
'''
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    accuracy = np.count_nonzero(predictions == test_labels)/test_labels.shape[0]*100
    return accuracy
#---------------------------------------------------------------------#

class grid_search_XG():
    def __init__(self, par1_list,par1_name, par2_list, par2_name, xg1, x_train, y_train, x_val, y_val):
            self.model = xg1
            self.par1_list = par1_list
            self.par1_name = par1_name
            self.par2_name = par2_name
            self.par2_list = par2_list
            self.x_train = x_train
            self.y_train = y_train
            self.x_val = x_val
            self.y_val = y_val
            self.result = self.do_search()
    
    def do_search(self):
        if self.par2_list is None:
            result = pd.DataFrame(columns = [self.par1_name, "accuracy"])
            for par_1 in self.par1_list:
                alg = self.model.set_params(gamma = par_1)
                alg.fit(self.x_train, self.y_train)
                random_accuracy = evaluate(alg, self.x_val, self.y_val)
                cur_com = pd.DataFrame([[par_1, random_accuracy]], columns=[self.par1_name,  "accuracy"])
                result = result.append(cur_com, ignore_index = True)
                del alg #free memory
            
        else:
            result = pd.DataFrame(columns = [self.par1_name, self.par2_name, "accuracy"])
            for par_1 in self.par1_list:
                for par_2 in self.par2_list:
                    alg = self.model.set_params(learning_rate=par_1, n_estimators = par_2)
                    alg.fit(self.x_train, self.y_train)
                    random_accuracy = evaluate(alg, self.x_val, self.y_val)
                    cur_com = pd.DataFrame([[par_1, par_2,random_accuracy]], columns=[self.par1_name, self.par2_name, "accuracy"])
                    result = result.append(cur_com, ignore_index = True)
                    del alg #free memory    
        return result
    
        def save_csv(self):
            file_name = "XG_result_"+self.par1_name+self.par2_name+".csv"
            self.result.to_csv(file_name)
            
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

def testresult(model, x_train, y_train):
    x_test, y_test = load_testdata()
    rf_best = model.fit(x_train, y_train)
    print("finish training")
    acc = evaluate(rf_best, x_test, y_test)
    print("Test result", acc)
    del rf_best # free memory    
#---------------------------------------------------------------------#

def main():    
    np.random.seed(42)    
    # Import features
    x_train, x_val, y_train, y_val = import_feature()
    
    # ------------------quick run random forest without tuning---------#  
    '''
    base_model = XGBClassifier()
    base_model.fit(x_train, y_train)
    base_accuracy = evaluate(base_model, x_val, y_val)
    print("Accuracy on validation data without tuning:", base_accuracy)
    del base_accuracy, base_model
    # 17% accuracy without training
    
    #-------------------Validation test for parameter turining---------#
    
    xgb1 = XGBClassifier(
     learning_rate =0.1,
     n_estimators=40,
     max_depth=9,
     min_child_weight=3,
     gamma=0,
     subsample=0.8,
     colsample_bytree=0.8,
     objective= 'multi:softmax',
     nthread=4,
     seed=27,
     use_label_encoder=False)
    
    # Create calidate list
    '''
    '''
    # First test n_estimator and max_depth
    # Need to change line 69/80 by hand
    n_estimators_list= np.arange(40, 150, 20) 
    max_depth = range(5,12,2)
    '''
    '''
    # Next test min_child_weight and gamma
    # Need to change line 69/80 by hand
    min_child_weight = range(1,6,2)
    gamma = [i/10.0 for i in range(0,5)]
    '''
    '''
    # lastly test learning rate and n_estimators    
    learning_rate= [0.1, 0.01, 0.001]
    n_estimators= [100]
    XGtest = grid_search_XG(learning_rate,"learning_rate", n_estimators, "n_estimators", xgb1, x_train, y_train, x_val, y_val)
    print("Validation result:")
    print(XGtest.result)
    '''
    #-------------------Final evaluation for test data ----------------#
    # test untune
    base_model = XGBClassifier()
    base_accuracy =  testresult(base_model, x_train, y_train)
    print("Accuracy on testing data without tuning:", base_accuracy)
    del base_accuracy, base_model
    
    '''
    # test tuned
    xgb1 = XGBClassifier(
     learning_rate =0.01,
     n_estimators=100,
     max_depth=11,
     min_child_weight=5,
     gamma=0,
     subsample=0.8,
     colsample_bytree=0.8,
     objective= 'multi:softmax',
     nthread=4,
     seed=27,
     use_label_encoder=False) 
    
    testresult(xgb1, x_train, y_train)
    '''
#-----------------------------------------------------------------#

if __name__ == "__main__":
    main()