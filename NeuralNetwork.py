import math

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from pre_processing import *
import warnings
from sklearn.metrics import confusion_matrix
from scipy.special import logsumexp
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')
class NeuralNetwork():
    def __init__(self, Features, Classes, Learning_Rate, Num_epochs, Data, Bias , layers,neurons,function_name):
        self.Init(Features, Classes, Learning_Rate, Num_epochs, Data, Bias , layers,neurons,function_name)

    def Init(self, Features, Classes, Learning_Rate, Num_epochs, Data, Bias , layers,neurons,function_name):
        self.X = Data[Features]
        self.Y = Data.iloc[:,0:1]
        self.X, self.Y = Pre_Processing(self.X, self.Y, Classes)

        self.bias = Bias
        self.Learning_Rate = Learning_Rate
        self.epochs = Num_epochs
        self.layers = layers
        self.Features = Features
        self.neurons = neurons
        self.function_name = function_name
        self.Classes = Classes
        if (Bias == 1):
            self.X.insert(0, 'Ones', 1)
            self.weights = []
            self.neurons.insert(0,len(self.Features)+1)
            self.neurons.append(3)
            index = 1
            for i in range(len(self.neurons) -1 ):
                Weights = np.random.rand(self.neurons[index],self.neurons[index-1])
                # Weights = np.ones(shape=(self.neurons[index],self.neurons[index-1]))

                self.weights.append(Weights)
                index = index+1
        else:
            self.weights = []
            self.neurons.insert(0, len(self.Features))
            self.neurons.append(3)
            index = 1
            for i in range(len(self.neurons) - 1):
                Weights = np.random.rand(self.neurons[index], self.neurons[index - 1])
                # Weights = np.ones(shape=(self.neurons[index], self.neurons[index - 1]))
                self.weights.append(Weights)
                index = index + 1

        self.X = np.array(self.X)
        self.Y = np.array(self.Y)
        self.Split_Data()
    def Tarin(self):

        predicates=[]
        for i in range(self.epochs):
            for x in range(self.Xtrain.shape[0]):
                feed_forward = []
                back_prop = []
                Sample = self.Xtrain[x].T
                Net_value = np.dot(self.weights[0], Sample) # 2 * 1
                feed_forward.insert(0,self.predictedFunction(Net_value))
                for j in range(1,len(self.weights)):
                    Net_values = np.dot( self.weights[j], feed_forward[j-1])
                    feed_forward.insert(j, self.predictedFunction(Net_values))
                feed_forward[len(feed_forward)-1]

                last_index_forward = len(feed_forward)-1
                Last_error =  feed_forward[last_index_forward] - self.YTrain[x].T  # 3 * 1
                Last_error = Last_error * self.DerivativeFunction(feed_forward[last_index_forward]) # 3 * 1
                back_prop.insert(0,Last_error)
                for w in range(1, len(self.weights)):
                    output_Derv = self.DerivativeFunction(feed_forward[last_index_forward - 1])
                    SumError = np.dot(self.weights[last_index_forward].T, back_prop[w - 1])
                    total_error = output_Derv * SumError
                    last_index_forward = last_index_forward - 1
                    back_prop.insert(w,total_error)

                first_back_error = len(back_prop) - 1
                self.weights[0] = self.weights[0] - self.Learning_Rate * np.dot(back_prop[first_back_error].reshape(1,-1).reshape(-1,1),self.Xtrain[x].reshape(1,-1))
                for k in range(1, len(self.weights)):
                    Updated = self.Learning_Rate * np.dot(back_prop[first_back_error-1].reshape(1,-1).reshape(-1,1),feed_forward[k-1].reshape(1,-1))

                    self.weights[k] = self.weights[k] - Updated

                    first_back_error = first_back_error-1
            # print('Iteration {0} = {1}'.format(i,self.weights))

        Net_value = np.dot(self.weights[0], self.Xtrain.T)  # 2 * 1
        predicates.insert(0,self.predictedFunction(Net_value))
        for acc in range(1, len(self.weights)):
            Net_values =np.dot(self.weights[acc],predicates[acc - 1])
            predicates.insert(acc, self.predictedFunction(Net_values))

        output = predicates[len(predicates) - 1]
        max_elements = np.amax(output, axis=0)
        max_elements = max_elements[None, :]
        new_arr = np.where(output == max_elements, 1, 0)


        print('accuracy = ',accuracy_score(y_true=self.YTrain.T,y_pred=new_arr))
        self.test_function()

    def predictedFunction(self, value):
        if(self.function_name == 'sigmoid'):
                return 1.0/(1.0 + np.exp(-value))
        elif (self.function_name == 'tanh'):
                return ( ( np.exp(value) - np.exp(-value) )  /   ( np.exp(value) + np.exp(-value) )  )
    def Error(self, predicted, actual):
        Error = predicted - actual
        return Error

    def DerivativeFunction(self,value):
        if (self.function_name == 'sigmoid'):
                return  value * (1.0 - value)
        else:
                return ( 1 - ( self.predictedFunction(value) * self.predictedFunction(value) ) )

    def Split_Data(self):
        self.Xtrain, self.Xtest, self.YTrain, self.Ytest = train_test_split(self.X, self.Y, test_size=0.4,
                                                                            train_size=0.6, shuffle=True,
                                                                               stratify=self.Y)
    def test_function(self):
        predicates = []
        Net_value = np.dot(self.weights[0], self.Xtest.T)  # 2 * 1
        predicates.insert(0, self.predictedFunction(Net_value))
        for acc in range(1, len(self.weights)):
            Net_values = np.dot(self.weights[acc], predicates[acc - 1])
            predicates.insert(acc, self.predictedFunction(Net_values))

        output = predicates[len(predicates) - 1]
        max_elements = np.amax(output, axis=0)
        max_elements = max_elements[None, :]
        new_arr = np.where(output == max_elements, 1, 0)
        print('accuracy Test = ', accuracy_score(y_true=self.Ytest.T, y_pred=new_arr))
    def Confusion_Matrix(self, pred):
        confusion = confusion_matrix(self.Ytest, pred)
        print('Confusion Matrix\n')
        print(confusion)