#BASTARDA!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


from MLP import *
from Utility import *
import numpy as np
from Monk import *
from matplotlib import pyplot as plt

eta_values = np.linspace(0.1,1,10)
alfa_values = np.linspace(0.1,1,10)
hidden_values = range(1,10,2)
weight_values = np.linspace(0.1,0.8,8)


eta_values = [0.8, 0.85, 0.90, 0.95]
alfa_values = [0.85, 0.88, 0.9]
hidden_values =[3,4,5,8,9]
weight_values = [0.2,0.3,0.5]

for h in weight_values:
    print(h)

X1, Y1 = load_monk("monks-1.train")
X_val1, Y_val1 = load_monk("monks-1.test")

X2, Y2 = load_monk("monks-2.train")
X_val2, Y_val2 = load_monk("monks-2.test")

X3, Y3 = load_monk("monks-3.train")
X_val3, Y_val3 = load_monk("monks-3.test")

best_vl_error = 1e10
best_mlp = None
best_eta = 0
best_alfa = 0
best_hidden = 0
best_weight = 0

for eta in eta_values:
    for alfa in alfa_values:
        for hidden in hidden_values:
            for weight in weight_values:

                print("Provo eta=%s alfa=%s #hidden=%s weight=%s"%(eta,alfa,hidden,weight))



                mlp = MLP(17, hidden, 1, TanhActivation(), SigmoidActivation(), eta=eta, alfa=alfa, lambd=0, fan_in_h=True,
                          range_start_h=-weight, range_end_h=weight)

                mlp.train(addBias(X1), Y1, addBias(X_val1), Y_val1, 500, 1e-30)

                if best_mlp is None:
                    best_mlp = mlp
                    best_vl_error = mlp.errors_vl[-1]
                    best_eta = eta
                    best_alfa = alfa
                    best_hidden = hidden
                    best_weight = weight

                elif mlp.errors_vl[-1] < best_vl_error:
                    best_mlp = mlp
                    best_vl_error = mlp.errors_vl[-1]
                    best_eta = eta
                    best_alfa = alfa
                    best_hidden = hidden
                    best_weight = weight
                    print("SELEZIONATO: ERRORE = ",best_vl_error)

print("CONFIGURAZIONE SCELTA eta=%s alfa=%s #hidden=%s weight=%s"%(best_eta,best_alfa,best_hidden,best_weight))

plt.subplot(2, 1, 1)
plt.plot(mlp.errors_tr)
plt.plot(mlp.errors_vl)
plt.subplot(2, 1, 2)
plt.plot(mlp.accuracies_tr)
plt.plot(mlp.accuracies_vl)
plt.show()


for eta in eta_values:
    for alfa in alfa_values:
        for hidden in hidden_values:
            for weight in weight_values:

                print("Provo eta=%s alfa=%s #hidden=%s weight=%s"%(eta,alfa,hidden,weight))



                mlp = MLP(17, hidden, 1, TanhActivation(), SigmoidActivation(), eta=eta, alfa=alfa, lambd=0, fan_in_h=True,
                          range_start_h=-weight, range_end_h=weight)

                mlp.train(addBias(X2), Y2, addBias(X_val2), Y_val2, 500, 1e-30)

                if best_mlp is None:
                    best_mlp = mlp
                    best_vl_error = mlp.errors_vl[-1]
                    best_eta = eta
                    best_alfa = alfa
                    best_hidden = hidden
                    best_weight = weight

                elif mlp.errors_vl[-1] < best_vl_error:
                    best_mlp = mlp
                    best_vl_error = mlp.errors_vl[-1]
                    best_eta = eta
                    best_alfa = alfa
                    best_hidden = hidden
                    best_weight = weight
                    print("SELEZIONATO: ERRORE = ",best_vl_error)

print("CONFIGURAZIONE SCELTA eta=%s alfa=%s #hidden=%s weight=%s"%(best_eta,best_alfa,best_hidden,best_weight))

plt.subplot(2, 1, 1)
plt.plot(mlp.errors_tr)
plt.plot(mlp.errors_vl)
plt.subplot(2, 1, 2)
plt.plot(mlp.accuracies_tr)
plt.plot(mlp.accuracies_vl)
plt.show()

for eta in eta_values:
    for alfa in alfa_values:
        for hidden in hidden_values:
            for weight in weight_values:

                print("Provo eta=%s alfa=%s #hidden=%s weight=%s"%(eta,alfa,hidden,weight))



                mlp = MLP(17, hidden, 1, TanhActivation(), SigmoidActivation(), eta=eta, alfa=alfa, lambd=0, fan_in_h=True,
                          range_start_h=-weight, range_end_h=weight)

                mlp.train(addBias(X3), Y3, addBias(X_val3), Y_val3, 500, 1e-30)

                if best_mlp is None:
                    best_mlp = mlp
                    best_vl_error = mlp.errors_vl[-1]
                    best_eta = eta
                    best_alfa = alfa
                    best_hidden = hidden
                    best_weight = weight

                elif mlp.errors_vl[-1] < best_vl_error:
                    best_mlp = mlp
                    best_vl_error = mlp.errors_vl[-1]
                    best_eta = eta
                    best_alfa = alfa
                    best_hidden = hidden
                    best_weight = weight
                    print("SELEZIONATO: ERRORE = ",best_vl_error)

print("CONFIGURAZIONE SCELTA eta=%s alfa=%s #hidden=%s weight=%s"%(best_eta,best_alfa,best_hidden,best_weight))

plt.subplot(2, 1, 1)
plt.plot(mlp.errors_tr)
plt.plot(mlp.errors_vl)
plt.subplot(2, 1, 2)
plt.plot(mlp.accuracies_tr)
plt.plot(mlp.accuracies_vl)
plt.show()