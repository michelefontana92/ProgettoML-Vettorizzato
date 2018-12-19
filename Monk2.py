from Monk import *
from MLP import *
from Activation_Functions import *
from matplotlib import pyplot as plt
from Utility import *
X, Y = load_monk("monks-2.train")

mlp = MLP(17,3,1,TanhActivation(),SigmoidActivation(),eta=0.9,alfa=0.6,lambd=0)
mlp.train(addBias(X),Y,1500,1e-6)

plt.plot(mlp.errors_tr)
plt.show()