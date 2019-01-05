from KFoldCV import *
from Monk import *
from matplotlib import pyplot as plt

eta_values = [0.5,0.6,0.7,0.8]
alfa_values = [0.7,0.8]
hidden_values =[2,3]
weight_values = [0.7]
lambda_values = [0,0.01,0.1]
n_epochs = 500
n_trials = 5
k = 3
n_features = 17

X,T = load_monk("monks-1.train")

"KFOLD CV"
best_eta,best_alfa,best_hidden,best_lambda,best_weight,best_mean_vl_error,best_std_vl_error=kFoldCV(n_features,X,T,k,500,
    TanhActivation(),SigmoidActivation(),
    eta_values,alfa_values,hidden_values,weight_values,lambda_values,n_trials)


"RETRAINING"
X_tr,T_tr = load_monk("monks-1.train")
X_vl,T_vl = load_monk("monks-1.test")

mlp, mean_err_tr, std_err_tr, mean_acc_tr, std_acc_tr, mean_err_vl, std_err_vl, mean_acc_vl, std_acc_vl = run_trials(n_features,
                                X_tr, T_tr, X_vl, T_vl, n_epochs,TanhActivation(),SigmoidActivation(), best_eta, best_alfa, best_hidden, best_weight,
                                best_lambda, n_trials,)

print("TR ERR = %3f TR ACC = %3f VL ERR = %3f VL ACC = %3f" % (mlp.errors_tr[-1], mlp.accuracies_tr[-1],
                                                               mlp.errors_vl[-1], mlp.accuracies_vl[-1]))

st = plt.suptitle("Monk 1(Best model)\neta=%s alpha=%s lambda=%s n_hidden=%s"%(mlp.eta,mlp.alfa,mlp.lambd,mlp.n_hidden))
plt.subplot(2, 1, 1)
plt.plot(mlp.errors_tr,label='Training Error',ls="-")
plt.plot(mlp.errors_vl,label='Validation Error',ls="dashed")
plt.ylabel('loss')
plt.grid(True)
plt.xlabel('epoch')
plt.legend(loc='upper right',prop={'size':12})
plt.subplot(2, 1, 2)
plt.plot(mlp.accuracies_tr,label='Training Accuracy',ls="-")
plt.plot(mlp.accuracies_vl,label='Validation Accuracy',ls="dashed")
plt.ylabel('Accuracy')
plt.grid(True)
plt.xlabel('epoch')
plt.legend(loc='lower right',prop={'size':12})
plt.show()