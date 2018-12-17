import numpy as np


class MLP:

    #Pesi già con bias
    def __init__(self,W_h,W_o):
        self._W_h = W_h
        self._W_o = W_o
        self.out_h = None
        self.out_o = None

        self._dW_o_old = np.zeros(W_o.shape)
        self._dW_h_old = np.zeros(W_h.shape)

    # X già con bias
    def feedforward(self,X):

        net_h = np.dot(X,self._W_h.T)
        #f(net_h)

        self.out_h = net_h #DA AGGIORNARE

        #print("net_h",net_h.shape)

        net = np.ones((net_h.shape[0],net_h.shape[1] +1))
        net[:,1:] = net_h
        #print("net con bias", net)
        net_o = np.dot(net,self._W_o.T)

        self.out_o = net_o #DA AGGIORNARE

    def backpropagation(self,X,T):

        delta_o = (T-self.out_o) #f'(net_o)****************
        #print("delta_o",delta_o)

        out_h_bias = np.ones((self.out_h.shape[0],self.out_h.shape[1] +1))
        out_h_bias[:,1:] = self.out_h
        #print(out_h_bias.shape)
        delta_W_o = np.dot(delta_o.T,out_h_bias)
        #print("deltaW_o",delta_W_o)

        delta_h = np.dot(delta_o,self._W_o[:,1:]) #f'(net_h) *******************
        #print("delta_h",delta_h)

        delta_W_h = np.dot(delta_h.T,X)
        #print("delta_W_h",delta_W_h)
        return delta_W_o, delta_W_h

    def train(self,X,T,eta = 0.1,alfa = 0., lambd = 0.):

        self.feedforward(X)

        #CALCOLA ERRORE!!!!

        dW_o, dW_h = self.backpropagation(X,T)

        dW_o_new = eta * dW_o + alfa * self._dW_o_old
        self._W_o = self._W_o + dW_o_new - (lambd * self._W_o)

        dW_h_new = eta * dW_h + alfa * self._dW_h_old
        self._W_h = self._W_h + dW_h_new - (lambd * self._W_h)

        self._dW_o_old = dW_o_new
        self._dW_h_old = dW_h_new

    def predict_class(self,X,treshold = 0.5):
        self.feedforward(X)
        predictions = np.zeros(self.out_o.shape)
        predictions[self.out_o >= treshold] = 1
        return predictions

    def predict_value(self):
        return

X = np.array([
    [1,2,-1],
    [1,0,0]
])

T = np.array([
    [1],
    [0]
])


W_h = np.array([
    [2,1,0],
    [2,0,1]
])
W_o = np.array([
    [-1,0,1]
])


mlp = MLP(W_h,W_o)
mlp.train(X,T)
print("W_h inizio",W_h)
print("aggiornamento W_h", mlp._dW_h_old)
print("W_h new",mlp._W_h)

print("W_o inizio", W_o)
print("aggiornamento W_o", mlp._dW_o_old)
print("W_o new",mlp._W_o)
print("out",mlp.out_o)
print("Prediction",mlp.predict_class(X))