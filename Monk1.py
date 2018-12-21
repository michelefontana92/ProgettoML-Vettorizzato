from Monk import *
from MLP import *
from Activation_Functions import *
from matplotlib import pyplot as plt
from Utility import *
import time

X, Y = load_monk("monks-1.train")
X_val, Y_val = load_monk("monks-1.test")


mlp = MLP(17,3,1,TanhActivation(),SigmoidActivation(),eta=0.9,alfa=.9,lambd=0,fan_in_h=True,range_start_h=-0.4,range_end_h=0.4)

start = time.time()
mlp.train(addBias(X),Y,addBias(X_val),Y_val,500,1e-6)
end = time.time()

print("VECTORIZED TIME ELAPSED = %3f sec per epoch"%((end-start)/500))
st = plt.suptitle("Monk 1")
plt.subplot(2,1,1)
plt.plot(mlp.errors_tr,label='Training Error',ls="-")
plt.plot(mlp.errors_vl,label='Validation Error',ls="dashed")
plt.ylabel('loss')
plt.grid(True)
plt.xlabel('epoch')
plt.legend(loc='upper right',prop={'size':12})

plt.subplot(2,1,2)
plt.plot(mlp.accuracies_tr,label='Training Accuracy',ls="-")
plt.plot(mlp.accuracies_vl,label='Validation Accuracy',ls="dashed")
plt.ylabel('Accuracy')
plt.grid(True)
plt.xlabel('epoch')
plt.legend(loc='lower right',prop={'size':12})
plt.show()