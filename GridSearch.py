from MLP import *
from Utility import *
import numpy as np


def run_trials(n_features,X_tr,T_tr,X_vl,T_vl,n_epochs,hidden_act,output_act,eta,alfa,n_hidden,weight,lambd,n_trials):

    best_vl_error = 1e10
    best_mlp = None
    best_idx = -1

    errors_tr = np.zeros((n_trials,n_epochs))
    acc_tr = np.zeros((n_trials,n_epochs))
    errors_vl = np.zeros((n_trials,n_epochs))
    acc_vl = np.zeros((n_trials,n_epochs))

    for trial in range(n_trials):

        print(100*'-')
        print("Trial %s/%s: "%(trial +1, n_trials))
        mlp = MLP(n_features, n_hidden, 1, hidden_act, output_act, eta=eta, alfa=alfa, lambd=lambd,
                  fan_in_h=True,
                  range_start_h=-weight, range_end_h=weight)

        mlp.train(addBias(X_tr), T_tr, addBias(X_vl), T_vl, n_epochs, 1e-30, suppress_print=True)

        errors_tr[trial] = mlp.errors_tr
        errors_vl[trial] = mlp.errors_vl
        acc_tr[trial] = mlp.accuracies_tr
        acc_vl[trial] = mlp.accuracies_vl

        if best_mlp is None:
            best_mlp = mlp
            best_vl_error = mlp.errors_vl[-1]
            best_idx = trial + 1

        elif mlp.errors_vl[-1] < best_vl_error:
            print("\nTROVATO ERRORE MIGLIORE = %s -> %s\n" % (best_vl_error, mlp.errors_vl[-1]))
            best_mlp = mlp
            best_vl_error = mlp.errors_vl[-1]
            best_idx = trial + 1

    mean_err_tr = np.mean(errors_tr,axis=0,keepdims=True).T
    std_err_tr = np.std(errors_tr,axis=0,keepdims= True).T
    mean_acc_tr = np.mean(acc_tr,axis=0, keepdims=True).T
    std_acc_tr = np.std(acc_tr,axis=0, keepdims=True).T
    mean_err_vl = np.mean(errors_vl, axis=0,keepdims=True).T
    std_err_vl = np.std(errors_vl,axis=0, keepdims=True).T
    mean_acc_vl = np.mean(acc_vl,axis=0, keepdims=True).T
    std_acc_vl = np.std(acc_vl,axis=0, keepdims=True).T

    print(100*"-")
    print("Returning model number ",best_idx)
    print(100 * "-")
    print("STATISTICS:")
    print("TR ERR = %3f +- %3f\nTR ACC = %3f +- %3f\nVL ERR = %3f +- %3f\nVL ACC = %3f +- %3f" %(mean_err_tr[-1],std_err_tr[-1],mean_acc_tr[-1],
                                                                                                 std_acc_tr[-1],mean_err_vl[-1],std_err_vl[-1],
                                                                                                 mean_acc_vl[-1],std_acc_vl[-1]))

    print(100*"-")
    print("\n")
    return best_mlp,mean_err_tr,std_err_tr,mean_acc_tr,std_acc_tr,mean_err_vl,std_err_vl,mean_acc_vl,std_acc_vl


def gridSearch(n_features,X_tr,T_tr,X_vl,T_vl,n_epochs,hidden_act,output_act, eta_values, alfa_values, hidden_values, weight_values, lambda_values,n_trials):
    best_vl_error = 1e10
    best_mlp = None
    best_eta = 0
    best_alfa = 0
    best_hidden = 0
    best_weight = 0
    best_lambda = 0
    best_mean_err_tr = np.zeros((n_epochs,1))
    best_std_err_tr = np.zeros((n_epochs,1))
    best_mean_acc_tr = np.zeros((n_epochs,1))
    best_std_acc_tr = np.zeros((n_epochs,1))
    best_mean_err_vl = np.zeros((n_epochs,1))
    best_std_err_vl = np.zeros((n_epochs,1))
    best_mean_acc_vl = np.zeros((n_epochs,1))
    best_std_acc_vl = np.zeros((n_epochs,1))

    for eta in eta_values:
        for alfa in alfa_values:
            for hidden in hidden_values:
                for weight in weight_values:
                    for lambd in lambda_values:
                        print(100 * '-')
                        print("Provo eta=%s alfa=%s #hidden=%s weight=%s lambda = %s" % (
                        eta, alfa, hidden, weight, lambd))

                        mlp,mean_err_tr,std_err_tr,mean_acc_tr,std_acc_tr,mean_err_vl,std_err_vl,mean_acc_vl,std_acc_vl = run_trials(n_features,X_tr,T_tr,
                                                                                                            X_vl,T_vl,n_epochs,hidden_act,
                                                                                                        output_act,eta,alfa,hidden,
                                                                                                weight,lambd,n_trials)

                        if best_mlp is None:
                            best_mlp = mlp
                            best_vl_error = mean_err_vl[-1]

                            best_eta = eta
                            best_alfa = alfa
                            best_hidden = hidden
                            best_weight = weight
                            best_lambda = lambd

                            best_mean_err_tr = mean_err_tr
                            best_std_err_tr = std_err_tr
                            best_mean_acc_tr = mean_acc_tr
                            best_std_acc_tr = std_acc_tr
                            best_mean_err_vl = mean_err_vl
                            best_std_err_vl = std_err_vl
                            best_mean_acc_vl = mean_acc_vl
                            best_std_acc_vl = std_acc_vl

                        elif mean_err_vl[-1] < best_vl_error:
                            print("\nTROVATO ERRORE MIGLIORE = %s -> %s\n" % (best_vl_error, mean_err_vl[-1]))
                            best_mlp = mlp
                            best_vl_error = mean_err_vl[-1]

                            best_eta = eta
                            best_alfa = alfa
                            best_hidden = hidden
                            best_weight = weight
                            best_lambda = lambd
                            best_mean_err_tr = mean_err_tr
                            best_std_err_tr = std_err_tr
                            best_mean_acc_tr = mean_acc_tr
                            best_std_acc_tr = std_acc_tr
                            best_mean_err_vl = mean_err_vl
                            best_std_err_vl = std_err_vl
                            best_mean_acc_vl = mean_acc_vl
                            best_std_acc_vl = std_acc_vl

    print("CONFIGURAZIONE SCELTA eta=%s alfa=%s #hidden=%s weight=%s lambda=%s" % (
    best_eta, best_alfa, best_hidden, best_weight,best_lambda))
    print("STATISTICS:")
    print("TR ERR = %3f +- %3f\nTR ACC = %3f +- %3f\nVL ERR = %3f +- %3f\nVL ACC = %3f +- %3f" % (
    best_mean_err_tr[-1], best_std_err_tr[-1], best_mean_acc_tr[-1],
    best_std_acc_tr[-1], best_mean_err_vl[-1], best_std_err_vl[-1],
    best_mean_acc_vl[-1], best_std_acc_vl[-1]))
    print("MIGLIOR MODELLO DEL GRUPPO:")
    print("TR ERR = %3f TR ACC = %3f VL ERR = %3f VL ACC = %3f" % (best_mlp.errors_tr[-1], best_mlp.accuracies_tr[-1],
                                                                                                  best_mlp.errors_vl[-1], best_mlp.accuracies_vl[-1]))

    return best_mlp,best_mean_err_tr,best_std_err_tr,best_mean_acc_tr,best_std_acc_tr,best_mean_err_vl,\
           best_std_err_vl,best_mean_acc_vl,best_std_acc_vl