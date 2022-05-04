# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 08:16:52 2021

@author: Stephan
"""
import numpy as np
import pandas as pd
from time import time
#---------------------------------------------------------------------#
'''
# load the feature set and class label
'''
def readdata(path):
    # read the txt file with image path and class label
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
do model evaluation on accuracy
given the model trained, predict the top 1 and top 5 classes
compare with test label
'''
def evaluate(model, test_features, test_labels):
    # predictions_top: list of class with top prob
    predictions_top, predictions_top5 = model.predict(test_features)
    # fix error if DataFrame shape not match
    if predictions_top.shape != test_labels.shape: 
        predictions_top = pd.DataFrame(predictions_top)
    # percentage of top 1 accuracy, count number of match with test label
    accuracy_top1 = np.count_nonzero(predictions_top == test_labels)/test_labels.shape[0]*100
    # Predictions_top: each row have top 5 accuracy
    # Loop over each row and compare
    count = 0
    for i in range(test_labels.shape[0]):
        if test_labels[i] in predictions_top5[i]: count = count + 1
    accuracy_top5 = count/test_labels.shape[0]*100
    return accuracy_top1, accuracy_top5
#---------------------------------------------------------------------#
'''
turn the class label(y value) into one hot vector for softmax
'''
def turn_one_hot(y_label):
    num_class = y_label.nunique() #number of class value in data set
    if type(num_class) is not int: num_class = num_class[0] # fix error for differnet data type
    one_hot_labels = np.zeros((y_label.shape[0], num_class)) # create np array of number of data * number of class
    for i in range(y_label.shape[0]):
        one_hot_labels[i, y_label.iloc[i]] = 1
    return pd.DataFrame(one_hot_labels)

#---------------------------------------------------------------------#
'''
# load test data
'''
def load_testdata():
    test_ref = readdata("test.txt")
    x_test = readfeature('x_test.csv')
    y_test = test_ref['class']
    print("finish loaded test data")
    return x_test, y_test
#---------------------------------------------------------------------#
'''
forward and backward operations
'''
class dot_matrix():
    def __init__(self,x, w):
        self.x = x
        self.w = w
        
    def forward(self):
        self.s0 = self.x.dot(self.w)
        return self.s0
    
    def backward_dx(self):
        return self.w.T
    
    def backward_dw(self):
        return self.x.T
    
class plus_oper():        
    def forward(self,a, b):
        return a+b
    
    def backward (self):
        dev = int(1)
        return dev
    
class softmax():
    def __init__(self,A):
        self.A = A
        
    def forward(self):
        try: expA = np.exp(self.A)
        except TypeError:
            expA =  np.exp(self.A.astype(np.float64))
        try: 
            expA = expA.to_numpy()
        except: AttributeError
        self.softmax_prob = expA / expA.sum(axis=1, keepdims = True)   
        return self.softmax_prob
    
    def softmax_loss(self, y_true):
        self.y_true = y_true
        loss = - y_true * np.log(self.forward())
        try: 
            total_loss = loss.sum().sum() # for pandas
        except AttributeError:
            total_loss = loss.sum() # for np array
        return total_loss
    
    def backward(self):
        return self.softmax_prob - self.y_true

class sigmoid():
    def forward(self, x):
        try:
            self.sig = 1 / (1 + np.exp(-x))
            
        except TypeError: 
            x = x.astype(np.float64)
            self.sig = 1 / (1 + np.exp(-x))
        return self.sig
    
    def backward(self):        
        return self.sig * (1 - self.sig)

class relu():
    def forward(self, x):
        return np.maximum(x, 0)        
        
    def backward(self, x):
        if x >=0: return 1
        else: return 0
#---------------------------------------------------------------------#
class perceptron():
    def __init__(self,x_train, y_train, num_node, learning_rate, minibatch_size, lambd, epoch):
        self.x_train = x_train # [n, m]
        self.y_train = y_train
        self.minibatch_size = minibatch_size
        self.num_node = num_node
        self.learning_rate = learning_rate
        self.num_feature = x_train.shape[1] # number of feature in data set
        self.num_row = x_train.shape[0] # number of data in data set
        self.num_class = y_train.shape[1]#number of class value in data set
        self.w1 = np.random.rand(self.num_feature, self.num_node) # [m, node]
        self.w2 = np.random.rand(self.num_node, self.num_class) # [node, c]
        self.b1 = np.random.randn(self.num_node)
        self.b2 = np.random.randn(self.num_class)
        self.lambd = lambd
        self.loss_flow = pd.DataFrame(columns = ["loss", "epoch"])
        self.epoch = epoch
                  
    def run_one_data(self, x, y, epoch):
        # forward prob
        # ------------------------- layer 1
        x_dot_weight = dot_matrix(x, self.w1) # [m,node].[n,m]
        s0 = x_dot_weight.forward()# [n,node] (x dot w)
        wx_plusb = plus_oper()
        s1 = wx_plusb.forward(s0, self.b1) # [n,node]
        sig_s1 = sigmoid()
        s2 = sig_s1.forward(s1) # [n,node]
        # ------------------------- layer 2
        layer1_dot_weight = dot_matrix(s2, self.w2) # [n,node].[node, class]
        s3 = layer1_dot_weight.forward() # [node, class]
        wx2_plusb2 = plus_oper()
        s4 = wx2_plusb2.forward(s3, self.b2) # [node, class]
        # ------------------------- 
        # soft max output
        softM = softmax(s4)
        loss = softM.softmax_loss(y)
        # ------------------------- 
        # back prob layer 2
        d_loss_ds4 = softM.backward() # [node, class]
        d_loss_ds3 = wx2_plusb2.backward() * d_loss_ds4 # [node, class]
        d_loss_dw2 = np.dot(layer1_dot_weight.backward_dw(), d_loss_ds3) # [node, class]
        d_loss_db2 = d_loss_ds3 # [node, class]
        # --------------------------- layer 1
        d_loss_ds2 = np.dot(d_loss_ds3, layer1_dot_weight.backward_dx())  # [n, node]    
        d_loss_ds1 = sig_s1.backward() * d_loss_ds2 # [n, node]
        d_loss_ds0 = wx_plusb.backward() * d_loss_ds1 # [n, node]
        d_loss_db1 = d_loss_ds0 # [n, node]
        d_loss_dw1 = np.dot(x_dot_weight.backward_dw(), d_loss_ds0) # [m,c]
        # Update Weights ================
        
        self.w1 = self.w1 -  self.learning_rate * (d_loss_dw1 + self.lambd * (self.w1)) 
        self.b1 = self.b1 - self.learning_rate * d_loss_db1.sum(axis=0) 
        self.w2 = self.w2 - self.learning_rate * (d_loss_dw2 + self.lambd * (self.w2)) 
        self.b2 = self.b2 - self.learning_rate * d_loss_db2.sum(axis=0)
        self.loss_flow = self.loss_flow.append(pd.DataFrame({"loss":[loss], "epoch":[epoch]}), ignore_index = True)
        return loss
        
    def create_minibatch(self):
        x_train_mini, y_train_mini = [], []
        # loop over every batch size of daa and add to the train_mini list
        for i in range(self.minibatch_size,self.num_row + self.minibatch_size, self.minibatch_size):
            x_train_mini.append(self.x_train.iloc[i-self.minibatch_size:i])
            y_train_mini.append(self.y_train.iloc[i-self.minibatch_size:i])
        x_train_mini, y_train_mini = np.array(x_train_mini, dtype=object), np.array(y_train_mini, dtype=object)
        return x_train_mini, y_train_mini
    
    def train(self):                          
        # Create minibatch for training
        x_train_mini, y_train_mini = self.create_minibatch()
        
        for epoch in range(self.epoch):
            loss = 0
            for i in range(x_train_mini.shape[0]):
                loss = loss + self.run_one_data(x_train_mini[i],y_train_mini[i], epoch)     
        self.loss_flow.to_csv("training_result.csv")
        
    def predict(self, x_test):
  
        # forward prob
        # ------------------------- layer 1
        x_dot_weight = dot_matrix(x_test, self.w1) # [m,node].[n,m]
        s0 = x_dot_weight.forward()# [n,node] (x dot w)
        wx_plusb = plus_oper()
        s1 = wx_plusb.forward(s0, self.b1) # [n,node]
        sig_s1 = sigmoid()
        s2 = sig_s1.forward(s1) # [n,node]
        # ------------------------- layer 2
        layer1_dot_weight = dot_matrix(s2, self.w2) # [n,node].[node, class]
        s3 = layer1_dot_weight.forward()
        wx2_plusb2 = plus_oper()
        s4 = wx2_plusb2.forward(s3, self.b2) # [n,node]
        # ------------------------- 
        # soft max output
        softM = softmax(s4)
        softmax_score = softM.forward()
        result_top1 = np.argmax(softmax_score, axis=1) 
        result_top5 = np.argpartition(softmax_score, -5, axis= 1)[:,-5:]
        return np.array(result_top1), np.array(result_top5)
    
    def set_params (self, parm_name, parm_value):
        # for changing parameters
        if parm_name == 'num_node':
            self.num_node = parm_value
        elif parm_name == 'learning_rate':
            self.learning_rate = parm_value
        elif parm_name == 'minibatch_size':
            self.minibatch_size = parm_value
        elif parm_name == 'lambd':
            self.lambd = parm_value
        elif parm_name == 'epoch':
            self.epoch = parm_value
        else: 
            print("error: please check parm_name, only support -num_node, learning_rate, minibatch_size, lambd, epoch")
            import sys
            sys.exit()
            
    def reset(self):        
        # for reseting after each validation
        self.w1 = np.random.rand(self.num_feature, self.num_node) # [m, node]
        self.w2 = np.random.rand(self.num_node, self.num_class) # [node, c]
        self.b1 = np.random.randn(self.num_node)
        self.b2 = np.random.randn(self.num_class)
        self.loss_flow = pd.DataFrame(columns = ["loss"])
#---------------------------------------------------------------------# 
class grid_search_perceptron():
    def __init__(self, par1_list,par1_name, par2_list, par2_name, model, x_val, y_val):
            self.model = model
            self.par1_list = par1_list
            self.par1_name = par1_name
            self.par2_name = par2_name
            self.par2_list = par2_list
            self.x_val = x_val
            self.y_val = y_val
            self.result = self.do_search()
    
    def execute(self):
        start = time() # record start time        
        self.model.train()
        elapsed = time()-start # record end time
        accuracy_top, accuracy_top5 = evaluate(self.model, self.x_val, self.y_val)
        self.model.reset() # reset weighting
        return accuracy_top, accuracy_top5, elapsed
    
    def do_search(self):
        if self.par2_list is None:
            result = pd.DataFrame(columns = [self.par1_name, "accuracy_top", "accuracy_top5", "elapse"])
            # Validate each parameter in the validation candidate list
            for par_1 in self.par1_list:
                self.model.set_params(self.par1_name, par_1) # set parameters
                accuracy_top, accuracy_top5, elapsed = self.execute() # train and get accuracy            
                cur_com = pd.DataFrame([[par_1, accuracy_top, accuracy_top5, elapsed]], columns=[self.par1_name,  "accuracy_top", "accuracy_top5", "elapse"])
                print('Model Performance')
                print(cur_com)
                result = result.append(cur_com, ignore_index = True)
        else:
            result = pd.DataFrame(columns = [self.par1_name, self.par2_name, "accuracy_top", "accuracy_top5", "elapse"])
            for par_1 in self.par1_list:
                for par_2 in self.par2_list:
                    self.model.set_params(self.par1_name, par_1)
                    self.model.set_params(self.par2_name, par_2)
                    accuracy_top, accuracy_top5, elapsed = self.execute()    
                    cur_com = pd.DataFrame([[par_1, par_2,accuracy_top, accuracy_top5,elapsed ]], columns=[self.par1_name, self.par2_name, "accuracy_top", "accuracy_top5", "elapse"])
                    print('Model Performance')
                    print(cur_com)
                    result = result.append(cur_com, ignore_index = True) 
        return result  
    
    def save_csv(self):
        if self.par2_list is None: 
            file_name = "MLP_result_"+self.par1_name+".csv"
        else: file_name = "MLP_result_"+self.par1_name+self.par2_name+".csv"        
        self.result.to_csv(file_name)     
        
#---------------------------------------------------------------------#
def main():    
    
    np.random.seed(42)    
    
    '''     
    # To check the correctness of the MLP, I test it with other datasets
    '''    
    '''
    from sklearn import datasets
    from sklearn.model_selection import train_test_split    
    '''

    '''
    # ------------------------test dataset 1-----------------
    feature_set, labels = datasets.make_moons(1000, noise=0.20)
    # train accuracy 95.7%
    # test accuracy = 98%
    '''
    '''
    # ------------------------test dataset 2-----------------
    # loading the iris dataset 
    iris = datasets.load_iris() 
    
    # X -> features, y -> label 
    feature_set, labels = iris.data, iris.target
    # train accuracy=  97%
    # test accuracy = 100%
    '''
    '''
    # ----------------------- to split data-----------------
    # dividing X, y into train and test data 
    x_train, x_val, y_train, y_val = train_test_split(feature_set, labels, random_state = 42) 
    x_train, x_val, y_train, y_val = pd.DataFrame(x_train), pd.DataFrame(x_val),pd.DataFrame(y_train), pd.DataFrame(y_val)
    y_train_onehot = turn_one_hot(y_train)
    '''    
    # ------------------------------ to train and evaluate
    '''
    model.train()
    result = evaluate(model, x_train, y_train)
    print(result)
    result = evaluate(model, x_val, y_val)
    print(result)   
    '''
    
    # -----------------------image data set---------------------------------
    # import feature and class labels
    x_train, x_val, y_train, y_val = import_feature()
    y_train_onehot = turn_one_hot(y_train)    
    y_train_onehot = pd.DataFrame(y_train_onehot)
    
    # build model
    model = perceptron(x_train, y_train_onehot, 80, 0.001, 250, 0, 100)
    # para: num_node, learning_rate, minibatch_size, lambd, epoch
    
    # Cadidate list for validation
    learning_rate= [0.001, 0.0001, 0.00001]
    epoch = [1000, 5000, 10000]
    '''
    minibatch_size = [30, 60, 120, 250]
    num_node = [20, 40, 60, 80]
    '''
    # test 2 parameters @ a time
    MLPval = grid_search_perceptron(learning_rate,"learning_rate", epoch, "epoch", model, x_val, y_val)
    print(MLPval.result)
    MLPval.save_csv()   
       
    # -----------------------Final test---------------------------------
    
    x_test, y_test = load_testdata()
    lambd = [0, 0.00001]
    model = perceptron(x_train, y_train_onehot, 80, 0.001, 250, 0, 1000)
    MLPtest = grid_search_perceptron(lambd, "lambd",None, None, model, x_test, y_test)
    MLPtest.save_csv()
    
if __name__ == "__main__":
    main()