from matplotlib import pyplot as plt
import numpy as np
import random
from math import e as exp

class NeuralNetwork(object):
    def __init__(self, w = 1, w_list = None):
        self.w = []
        if not(w_list):
            for e in range(w):
                self.w.append(round(np.random.randn(), 5))
        else:
            self.w = w_list

        self.b = np.random.randn()
        self.output = 0
        self.learning_rate = 0.091     
        self.ws = []
        self.bs = []
        self.acertivity_control = 0
        self.train_control = True

        if self.acertivity_control > 30:
            self.train_control = False

    #transformer all the numbers in a number beetwen 0 and 1
    def sigmoid(self, x):
        #making x a small value
        x = x/100
        if x < -4:
            x = -10
        else:
            x = 10
        return 1 / (1 + exp**(-x))

    #test if my weight is already trained based on his acertivity
    def test_weight(self, y):
        if int(self.output) == y:
            self.acertivity_control += 1
        else:
            self.acertivity_control = 0

    #derivate of my sigmoid function
    def sigmoid_p(self, x):
        x = round(x, 10)
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    #using my IA
    def use(self, point):
        output = 0
        self.output = 0
        for e in range(len(self.w)):
            output += point[e]*self.w[e] #Sum of each neuron plus his weight 
        output += self.b #Sum bias
        output = int(output)
        self.output = self.sigmoid(output) #Mensuring our result beetwen 0 or 1
        
  
    def tanh_p(self, y):
        return (1-y)*(1+y)

    #Train loop
    def training_loop(self, point, output_expected):
        if True:
            output = 0
            for e in range(len(self.w)):
                output += point[e]*self.w[e] #Sum of each neuron plus his weight 
            output += self.b #Sum bias
            output = round(output, 25)
            
            self.output = self.sigmoid(output) #Getting our result 
            self.test_weight(output_expected) #Testing if weight is soo good

            target = output_expected #Result wait

            cost = np.square(self.output - target) #Getting our error margin named cost

            dcost_pred = 2 * (self.output - target) #Deriving cost
            d_output = self.sigmoid_p(self.output) #Deriving output
            dcost = dcost_pred * d_output #multiplying cost derivate for output derivate
                
            dz_dw = [] #derivating the weights
            for e in point:
                dz_dw.append(e)
            dz_db = 1 #derivating the bias

            dcost_dw = [] 
            #readjust the weights derivate and bias derivate multiplying it for our dcost
            for e in dz_dw:
                dcost_dw.append(dcost * e) #readjust weights derivate
            dcost_db = dcost * dz_db #readjust bias derivate

            #readjust weights and bias by their derivate
            for e in range(0, len(dcost_dw)):
                self.w[e] -= round((self.learning_rate * dcost_dw[e]), 25) #readjust weights
                self.w[e] = round(self.w[e], 25)
            self.b -= self.learning_rate * dcost_db #readjust bias
            
            output = 0


a = NeuralNetwork(3)
data_base = open('DataBase.ini')
dados = data_base.readlines()
print(len(dados))

for e in dados:
    in1, in2, in3, saida, void = e.split("|")
    in1, in2, in3, saida = int(in1), int(in2), int(in3), int(saida)
    data = [in1, in2, in3]
    a.training_loop(data, saida)
        
acertos, erros = 0, 0 
for e in dados[:1000]:
    in1, in2, in3, saida, void = e.split("|")
    in1, in2, in3, saida = int(in1), int(in2), int(in3), int(saida)
    data = [in1, in2, in3]
    a.use(data)
    if round(a.output) == saida:
#        print("Hehehehe")
        acertos += 1
    else:
        erros += 1
#        print("Fon!")
print(acertos, erros, len(dados))

