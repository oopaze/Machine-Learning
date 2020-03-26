from matplotlib import pyplot as plt
import numpy as np


class NeuralNetwork(object):
    def __init__(self, w = [np.random.randn(), np.random.randn()], b = np.random.randn()):
        self.w = w
        self.b = b
        self.output = 0
        self.learning_rate = 0.1
        self.ws = []
        self.bs = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def sigmoid_p(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def use(self, point):
        output = 0
        for e in range(len(self.w)):
            output += point[e]*self.w[e] #Sum of each neuron plus his weight 
        output += self.b #Sum bias
                
        self.output = self.sigmoid(output) #Getting our result 
            
    def traning_loop(self, data):
        for i in range(10000):
            
            randomIndex = np.random.randint(len(data)) #Taking a random number into 0 to len(data)
            point = data[randomIndex] #Take data based on number taken before

            output = 0
            for e in range(len(self.w)):
                output += point[e]*self.w[e] #Sum of each neuron plus his weight 
            output += self.b #Sum bias
                
            self.output = self.sigmoid(output) #Getting our result 
            target = point[2] #Result wait

            cost = np.square(self.output - target) #Getting our error margin named cost

            dcost_pred = 2 * (self.output - target) #Deriving cost
            d_output = self.sigmoid_p(self.output) #Deriving output
            dcost = dcost_pred * d_output #multiplying cost derivate for output derivate

            dz_dw = [point[0], point[1]] #derivating the weights
            dz_db = 1 #derivating the bias

            dcost_dw = [] 
            #readjust the weights derivate and bias derivate multiplying it for our dcost
            for e in dz_dw:
                dcost_dw.append(dcost * e) #readjust weights derivate
            dcost_db = dcost * dz_db #readjust bias derivate

            #readjust weights and bias by their derivate
            for e in range(len(self.w)):
                self.w -= self.learning_rate * dcost_dw[e] #readjust weights
            self.b -= self.learning_rate * dcost_db #readjust bias


data = [[3, 1.5, 1], 
       [2, 1, 0], 
       [4, 1.5, 1],
       [3, 1, 0], 
       [3.5, .5, 1], 
       [2, .5, 0], 
       [5.5, 1, 1], 
       [1, 1, 0],
       [4.5, 1, 1]]

a = NeuralNetwork()
a.traning_loop(data)

for elem in data:
    a.use(elem)
    print("in: {}, {} - Out: {}".format(elem[0], elem[1], round(a.output)), end=" ")
    if round(a.output) == elem[2]:
        print("- Yeah!!")
    else:
        print("- Fuck!!")
