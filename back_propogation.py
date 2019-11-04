import numpy as np

X = np.array(([0,0,0], [1,1,1], [0,1,0], [1,0,0], [0,1,1], [1,1,0]), dtype=float)
Y = np.array(([0], [1], [1], [1], [0], [0]), dtype=float)

class NeuralNetwork(object):
    def __init__(self):
        self.inputSize = 3
        self.outputSize = 1
        self.hiddenSize = 15
        self.eta = 0.01

        self.W1 = np.zeros((self.inputSize, self.hiddenSize))
        self.W2 = np.zeros((self.hiddenSize, self.outputSize))

        self.b1 = np.zeros((1, self.hiddenSize))
        self.b2 = np.zeros((1, self.outputSize))

    def feedForward(self, X):

        self.net1 = np.add(np.dot(X, self.W1), self.b1)
        self.out1 = self.sigmoid(self.net1)
        
        self.net2 = np.add(np.dot(self.out1, self.W2), self.b2)
        self.out2 = self.sigmoid(self.net2)

        return self.out2
      
    def sigmoid(self, s):
        return 1/(1 + np.exp(-(s)))

    def backPropagation(self, X, Y, output):

        error = output - Y
        print(error)
        self.W2 = self.W2 - self.eta * np.dot(self.out1.T, (error * (self.out2 * (1 - self.out2))))
        self.b2 = self.b2 - self.eta * (error * (self.out2 * (1 - self.out2)))
    
        self.W1 = self.W1 - self.eta * np.dot(X.reshape(-1,1), ((self.out1 * (1 - self.out1)) * (np.dot(self.out1.T, (error * (self.out2 * (1 - self.out2))))).T))    
        self.b1 = self.b1 - self.eta * ((self.out1 * (1 - self.out1)) * (np.dot(self.out1.T, (error * (self.out2 * (1 - self.out2))))).T)

    def train(self, X, Y):
        output = self.feedForward(X)
        self.backPropagation(X, Y, output)
        
NN = NeuralNetwork()
for j in range(6):
    print("======================NEXT==========================")
    for i in range(10000):
        NN.train(X[j], Y[j])

print(NN.W1)
print(NN.W2)
print(NN.b1)
print(NN.b2)
