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
    '''
    # read the txt file with image path and class label
    # Parameter: path to the image location
    '''
    df = pd.read_csv(path, sep= ' ', header=None, names=['img_link', 'class'])
    return df

def readfeature(path):
    '''
    # features are saved in csv format, read the csv file
    # Parameter: path to the image location/ file name
    '''
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
    '''   
    Parameters
    ----------
    model : TYPE
        Pretrained model.
    test_features : TYPE 
    test_labels : TYPE np array
        list of vetor labels, size = number of test feature.

    Returns
    -------
    accuracy_top1 : TYPE float
        Percentage of top1 accuracy.
    accuracy_top5 : TYPE float
        Percentage of top5 accuracy.

    '''
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
        left = test_labels.iloc[i]
        right = predictions_top5[i]
        if test_labels.iloc[i] in predictions_top5[i]: 
            count = count + 1
    accuracy_top5 = count/test_labels.shape[0]*100
    return accuracy_top1, accuracy_top5
#---------------------------------------------------------------------#
'''
turn the class label(y value) into one hot vector for softmax
'''
def turn_one_hot(y_label):
    '''
    Parameters
    ----------
    y_label: pandas vector dataframe

    Returns
    -------
    one_hot_labels : TYPE pandas dataframe [number of data X class]
        one hot vector of the class label.
    '''
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
        # return softmax probability of each sample of each class
        try: expA = np.exp(self.A)
        except TypeError:
            expA =  np.exp(self.A.astype(np.float64))
        try: 
            expA = expA.to_numpy()
        except: AttributeError
        self.softmax_prob = expA / expA.sum(axis=1, keepdims = True)   
        return self.softmax_prob
    
    def softmax_loss(self, y_true):
        '''
        Can run directly without running forward
        Parameter: one hot vector of class label
        return total softmax loss
        '''
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
    
    def backward(self, output):        
        return output * (1 - output)

class relu():
    def forward(self, x):
        self.activation = x
        return np.maximum(x, 0)        
        
    def backward(self, x):
        dx = np.array(x, copy=True) # just copy x
        for i, w in enumerate(dx):
            dx[i, w <= 0] = 0   # When x <= 0, set dx to 0 
            dx[i, w > 0] = 1   # When x > 0, set dx to 1.
        return dx
#---------------------------------------------------------------------#
class MLP():
    def __init__(self,x_train, y_train, hidden_layers, learning_rate, minibatch_size, lambd, epoch, act_function):
        self.x_train = x_train # [n, m]
        # original y_train is used for evaluation
        self.y_train_ori = y_train
        # turn y_train into one hot label
        y_train_onehot = turn_one_hot(y_train)    
        self.y_train = pd.DataFrame(y_train_onehot)
        
        # save parameters
        self.minibatch_size = minibatch_size
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate        
        self.lambd = lambd
        self.epoch = epoch
        self.act_function = act_function
        
        # save data shape 
        self.num_feature = x_train.shape[1] # total number of feature in data set
        self.num_row = x_train.shape[0] # total number of data in data set
        self.num_class = self.y_train.shape[1] #total number of class value in data set
        
        # initialise weights
        self.reset()
        
    def reset (self):
        # https://www.brilliantcode.net/1381/backpropagation-2-forward-pass-backward-pass/
        # list of layers with number of node in each layer
        # e.g. [number of feature, 100, 80, number of class]
        self.layers = [self.num_feature] + self.hidden_layers + [self.num_class] 
        
        # initialize random weights and bs
        self.weights = [np.random.randn(self.layers[i], self.layers[i+1])*0.01 for i in range(len(self.layers)-1)]
        self.bs = [np.zeros(self.layers[i+1]) for i in range(len(self.layers)-1)]       
                
        # save input for sigmoid per layer
        self.sigmoid_input = [np.zeros(self.layers[i]) for i in range(len(self.layers) - 1)]
        
        # save derivatives per layer
        self.derivatives_w =  [np.zeros((self.layers[i], self.layers[i + 1])) for i in range(len(self.layers) - 1)]
        self.derivatives_bs = [np.zeros(self.layers[i+1]) for i in range(len(self.layers)-1)]
        # reset loss memory
        self.loss_flow = pd.DataFrame(columns = ["loss", "top_1", "epoch"])
        
    def forwardpropagate (self, x, weight, b, layer):
        '''
        Parameters:
            x: feature
            weight: self.weight 
            b: self.bs
            layer: number of layer currently propagating (0 to layer -1)
        
        s0 = input dot weight
        s1 = s0 + b
        s2 = sigmoid (s1)
        
        Return: S2 [n,node of next layer]
        '''
        x_dot_weight = dot_matrix(x, weight) # [m,node].[n,m]
        s0 = x_dot_weight.forward()
        wx_plusb = plus_oper()
        s1 = wx_plusb.forward(s0, b) # [n,node]
        self.wxplusb[layer] = s1
        if layer != len(self.layers) - 2: # skip sigmoid for last layer
            if self.act_function[layer] == 'relu':
                s2 = relu().forward(s1) # [n,node]
            else: 
                s2 = sigmoid().forward(s1) # [n,node]
        else: 
            s2 = s1
        return s2
        
    def back_propagate(self, delta, layer):
        '''
        Parameters
        ----------
        delta : TYPE
            deviation of next layer(upper stream).
        layer : TYPE int
            number of layer currently propagating (0 to layer -1)

        Returns
        -------
        delta : TYPE
            ds0 of current layer, will pass to lower layer for backpropagation
        d_loss_dw1 : TYPE
            deviation of weights of current layer.
        d_loss_db1 : TYPE
            deviation of bias of current layer.

        '''
        # d_loss_ds2 = delta
        # d_loss_ds1 = ds2_ds1 * dloss_ds2
        if layer != len(self.layers) - 2: # skip sigmoid for last layer
            if self.act_function[layer] == 'relu':
                d_loss_ds1 = relu().backward(self.wxplusb[layer]) * delta # [n, node]
            else:
                d_loss_ds1 = sigmoid().backward(self.activations[layer + 1]) * delta # [n, node]
        else: 
            d_loss_ds1 = delta
            
        # ds1_ds0 = 1, d_loss_ds0 = d_loss_ds1
        d_loss_ds0 = d_loss_ds1 # [n, node]
        # ds1_db = 1, d_loss_db = d_loss_ds1
        d_loss_db1 = d_loss_ds1 # [n, node]
        
        # Compute deviatives of loss with respect to activation(delta) and weight
        x_dot_weight = dot_matrix(self.activations[layer], self.weights[layer])      
        d_loss_dw1 = np.dot(x_dot_weight.backward_dw(), d_loss_ds0) # [m,c]
        delta = np.dot(d_loss_ds0, x_dot_weight.backward_dx())  # [n, node]
        return delta, d_loss_dw1, d_loss_db1
    
    def run_one_data(self, x, y, epoch):
        '''
        Given minibatch of train feature and train class label and current epoch currently in
        do forward propagation, output loss
        do backward propagation, update weights
        '''
        
        # save activations and wx+b per layer
        self.activations = []
        self.wxplusb = [i for i in range(len(self.layers) - 1)] # will update during forward propagate
        # --------------forward propagation
        self.activations.append(x) # the first input is the feature set 
        # iterate through the network layers and do forward propagation
        # the input of i+1 layer is the output of i layer
        for i, w in enumerate(self.weights):
            output = self.forwardpropagate(self.activations[i], w, self.bs[i], i)
            self.activations.append(output)  
        # ---------------soft max output---------- 
        # put the output of the last layer into softmax
        softM = softmax(self.activations[-1]) # forward propagate the last layer to softmax
        softmax_score = softM.forward() # softmax probability
        loss = softM.softmax_loss(y) # total loss of current mini-batch
        # --------------back propagation----------- 
        # deviation of loss with respect to softmax input
        delta = softM.backward() # [n, class]
        # backpropagation for each layer
        for i in reversed(range(len(self.derivatives_w))):
            delta, self.derivatives_w[i], self.derivatives_bs[i] = self.back_propagate(delta, i)
        # ================ Update Weights ================ 
        for i in range(len(self.derivatives_w)):
            self.weights[i] = self.weights[i] - self.learning_rate * (self.derivatives_w[i]+ self.lambd * (self.weights[i])) 
            self.bs[i] = self.bs[i] - self.learning_rate * self.derivatives_bs[i].sum(axis = 0)
        
        return loss
    
    def save_result(self, file_name):
        # save the loss and accuracy for all epoch after training
        file_name = "MLP_train_result_"+file_name+".csv"
        self.loss_flow.to_csv(file_name)
        
    def create_minibatch(self):
        # return list of minibatch data
        x_train_mini, y_train_mini = [], []
        # loop over every batch size of data and add to the train_mini list
        for i in range(self.minibatch_size,self.num_row + self.minibatch_size, self.minibatch_size):
            x_train_mini.append(self.x_train.iloc[i-self.minibatch_size:i])
            y_train_mini.append(self.y_train.iloc[i-self.minibatch_size:i])
        x_train_mini, y_train_mini = np.array(x_train_mini, dtype=object), np.array(y_train_mini, dtype=object)
        return x_train_mini, y_train_mini
    
    def train(self):
        # Create minibatch for training
        x_train_mini, y_train_mini = self.create_minibatch()       
        
        for epoch in range(self.epoch):   
            loss = 0 # reset loss in the beginning of each epoch
            
            # run forward and backward propagation, update weights and return loss
            for i in range(x_train_mini.shape[0]):            
                loss = loss + self.run_one_data(x_train_mini[i],y_train_mini[i], epoch)
            loss = loss/self.num_row # mean of loss      
            
            # calculate the accuracy of the training data set
            accuracy_top, accuracy_top5 = evaluate(self, self.x_train, self.y_train_ori)            
            
            # save loss and accuracy of current epoch
            self.loss_flow = self.loss_flow.append(pd.DataFrame({"loss":[loss],"top_1": [accuracy_top], "epoch":[epoch]}), ignore_index = True)
        
        
    def predict(self, x_test):
  
        activations = x_test # the first input is the feature set 
        # iterate through the network layers and do forward propagation
        # the input of next layer is the output of i layer
        self.wxplusb = [i for i in range(len(self.layers) - 1)]
        for i, w in enumerate(self.weights):            
            activations = self.forwardpropagate(activations, w, self.bs[i], i)        
        # ---------------soft max output---------- 
        # put the output of the last layer into softmax
        softM = softmax(activations) 
        softmax_score = softM.forward()
        # --------------- Retrun list of results        
        result_top1 = np.argmax(softmax_score, axis=1) 
        result_top5 = np.argpartition(softmax_score, -5, axis= 1)[:,-5:]         
        return np.array(result_top1), np.array(result_top5)
    
    def set_params (self, parm_name, parm_value):
        # for changing parameters during hyper-parameter search
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
#---------------------------------------------------------------------# 
class grid_search_MLP():
    '''
    hyper parameters search
    it will update the specified hyper-parameters automatically,run the model,
    and do the test on validation data set
    '''
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
        # save the result (accuracies of different parameters)
        if self.par2_list is None: 
            file_name = "MLP_result_"+self.par1_name+".csv"
        else: file_name = "MLP_result_"+self.par1_name+self.par2_name+".csv"        
        self.result.to_csv(file_name)     
        
#---------------------------------------------------------------------#
def main():    
    
    np.random.seed(42)     
    # -----------------------image data set---------------------------------
    # import feature and class labels
    x_train, x_val, y_train, y_val = import_feature()
    # -----------------------Validation test---------------------------------
    
    # build model
    model = MLP(x_train, y_train, [80, 80], 0.001, 120, 0, 1000, ['sigmoid', 'relu'])
    
    # Cadidate list for validation
    learning_rate= [0.001, 0.0001, 0.00001]
    lambd = [0, 0.001]      
    
    minibatch_size = [60, 120, 250, 500] 
    num_node = [[70,50],[80,60],[80,70],[80,80], [100,80]]    
    
    # test 2 parameters @ a time
    MLPval = grid_search_MLP(learning_rate,"learning_rate",lambd, "lambd", model, x_val, y_val)
    print(MLPval.result)
    MLPval.save_csv()  
    
    MLPval = grid_search_MLP(minibatch_size,"minibatch_size",num_node, "num_node", model, x_val, y_val)
    print(MLPval.result)
    MLPval.save_csv()   

    # -----------------------Final test---------------------------------
    
    x_test, y_test = load_testdata()  
    
    # test model of one hiddne layer
    model = MLP(x_train, y_train, [80], 0.001, 250, 0, 1000, ['sigmoid'])
    model.train()
    model.save_result("one_hidden_layer")
    result_top, result_top5 = evaluate(model, x_test, y_test)
    print("result for one hidden layer:", result_top, result_top5)
    
    # test model of no hiddne layer
    model = MLP(x_train, y_train, [], 0.001, 250, 0, 1000, ['sigmoid'])
    model.train()
    model.save_result("no_hidden")
    evaluate(model, x_test, y_test)
    result_top, result_top5 = evaluate(model, x_test, y_test)
    print("result for no hidden layer:", result_top, result_top5)
    
    # test model of two hidden layer
    model = MLP(x_train, y_train, [80,80], 0.001, 120, 0, 100, ['sigmoid', 'relu'])
    model.train()
    model.save_result("two_hidden_layer")
    evaluate(model, x_test, y_test)
    result_top, result_top5 = evaluate(model, x_val, y_val)
    print("result for two hidden layer:", result_top, result_top5)   
        
if __name__ == "__main__":
    main()