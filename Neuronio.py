import numpy as np
import random
import time

#Faz de o dobro dos termos de uma lista
def dobro(x):
    y = []
    for e in x:
        y.append(2*e)
    return y

#Escrevendo pesos já filtrados como prioridade
def writing_w():
    good_w, good_w_found = reading_w()
    arq_w = open("w.ini", "a")
    for e in good_w:
        if e in good_w and e not in good_w_found:
            arq_w.write("{0}|2| \n".format(e))

#Testando e Filtrando pesos já salvos
def reading_w():
    good_w = []
    good_w_found = []
    arq_w = open("w.ini", "r")
    x = list(range(90, 101))
    for e in arq_w.readlines():
        w, boolen, void = e.split("|")

        if boolen == "2": #Testando se peso já tava na prioridade
            w = float(w)
            good_w_found.append(w) #Adicionando peso a lista de atualizados
        else:
            w = float(w)
            a = Perceptron(w)
            for e in x: 
                a.use(e) #Testando se peso deve entrar na lista de prioridade
            if round(a.result, 2) == 2*e: #Testando
                good_w.append(w) #Adicionando peso atualizado a lista de atualizados
    
    arq_w.close()
    return (good_w, good_w_found) #Retornando os pesos atualizados - os que já foram achados e os que foram achados agora


class Perceptron(object):
    def __init__(self, w):
        self.learningHate = 0.01
        self.w = w
        self.result = 0 
        self.coeficiente = 100000
        self.train_control = True

    def use(self, x):
        self.x = x/self.coeficiente
        self.result = np.tanh(self.x*self.w)*self.coeficiente
    
    def train(self, x, y):
        self.reading_w()
        if self.train_control:
            self.x = x/self.coeficiente
            self.y = y/self.coeficiente
            self.result = np.tanh(self.x*self.w)
            self.w += self.x*(self.learningHate*self.derivada(self.y-self.result))
            self.result = self.result * self.coeficiente

    def derivada(self, n):
        return n*(1-n)

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def reading_w(self):
        arq_w = open("w.ini", "r")
        for e in arq_w.readlines():
            w, boolen, void = e.split("|")
            if boolen == "2":
                self.w = float(w)
                self.train_control = False

writing_w() #Salvando os melhores dentre os pesos já salvos
x = list(range(101, 21494)) #Construindo os casos de teste de x
y, w = dobro(x), random.random() #Gerando y e o peso
a = Perceptron(w) #Construindo o Perceptron

#Treinando o Perceptron
for e in range(0, len(x)):
    a.train(x[e], y[e])

#Testando sua acertividade
x = list(range(1, 1001))
control = True
for e in x:
    a.use(e) #Testando o Perceptron
    print("in: %d =" %e, "Out: %.2f -"%a.result, "W: %.4f"%a.w) #Mostrando Entrada, Saida e o Peso
    if ("%.2f"%(2*e) != "%.2f"%a.result): #Vendo se a acertividade foi menor que duas casas decimais
        control = False

#Guardando valor do peso se ele tiver uma acertividade de duas casas decimais 
if control and a.train_control:
    print("You've found it!")
    arq_w = open("w.ini", "a")
    arq_w.write("{0}|1| \n".format(a.w)) #Escrevendo valor do peso no arquivo
    arq_w.close()



