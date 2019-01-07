from MLP import *
from Utility import *
import numpy as np

"""
Per multiple minima: si esegueguono piu trials
- Class/Regress

ADD...
Possibile modifica per ritornare anche tabella/plot trial per ogni trial-itarazione,( ma sempre ritorno media+std+modello)...
"""


def run_trials(n_features, X_tr, T_tr, X_vl, T_vl, n_epochs, hidden_act, output_act, eta, alfa, n_hidden, weight, lambd,
               n_trials, classification=True):
    best_vl_error = 1e10
    best_mlp = None
    best_idx = -1

    # Classificazione/Regressione:
    errors_tr = np.zeros((n_trials, n_epochs+1))
    errors_vl = np.zeros((n_trials, n_epochs+1))
    if classification:
        acc_tr = np.zeros((n_trials, n_epochs+1))
        acc_vl = np.zeros((n_trials, n_epochs+1))
    else:
        errors_MEE_tr = np.zeros((n_trials, n_epochs+1))
        errors_MEE_vl = np.zeros((n_trials, n_epochs+1))

    for trial in range(n_trials):

        print(100 * '-')
        print("Trial %s/%s: " % (trial + 1, n_trials))
        mlp = MLP(n_features, n_hidden, T_tr.shape[1], hidden_act, output_act, eta=eta, alfa=alfa, lambd=lambd,
                  fan_in_h=True, range_start_h=-weight, range_end_h=weight, classification=classification)

        mlp.train(addBias(X_tr), T_tr, addBias(X_vl), T_vl, n_epochs, 1e-30, suppress_print=True)

        # Classificazione/Regressione:
        errors_tr[trial] = mlp.errors_tr
        errors_vl[trial] = mlp.errors_vl
        if classification:
            acc_tr[trial] = mlp.accuracies_tr
            acc_vl[trial] = mlp.accuracies_vl
        else:
            errors_MEE_tr[trial] = mlp.errors_mee_tr
            errors_MEE_vl[trial] = mlp.errors_mee_vl

        # Se il migliore => prendo lui come modello (vedi slide matematica Validation part 2)
        ## Classificazione/Regressione:
        if classification:  # class==MONK; uso MSE
            if best_mlp is None:
                best_mlp = mlp
                best_vl_error = mlp.errors_vl[-1]  # Vedi slide Multipla minima (min sull'error vl MSE)
                best_idx = trial + 1
            elif mlp.errors_vl[-1] < best_vl_error:
                print("\nTROVATO ERRORE MIGLIORE = %s -> %s\n" % (best_vl_error, mlp.errors_vl[-1]))
                best_mlp = mlp
                best_vl_error = mlp.errors_vl[-1]  # Vedi slide Multipla minima (min sull'error vl MSE)
                best_idx = trial + 1
        else:  # regressione==CUP; uso MEE
            if best_mlp is None:
                best_mlp = mlp
                best_vl_error = mlp.errors_mee_vl[-1]  # MEE
                best_idx = trial + 1
            elif mlp.errors_vl[-1] < best_vl_error:
                print("\nTROVATO ERRORE MIGLIORE = %s -> %s\n" % (best_vl_error, mlp.errors_vl[-1]))
                best_mlp = mlp
                best_vl_error = mlp.errors_mee_vl[-1]  # MEE
                best_idx = trial + 1

    # Per avere una stima dell'error TR/VL (MSE) e dell'accuracy TR/VL se class,else MEE TR/VL se regressione
    ## Nota: Fondamentale stime(MSE)+LearningCurve => elementi di valitazione nella fase di progettazione!
    mean_err_tr = np.mean(errors_tr, axis=0, keepdims=True).T  # Media
    std_err_tr = np.std(errors_tr, axis=0, keepdims=True).T  # sqm (radice varianza)
    mean_err_vl = np.mean(errors_vl, axis=0, keepdims=True).T  # Media
    std_err_vl = np.std(errors_vl, axis=0, keepdims=True).T  # sqm (radice varianza)

    if classification:
        mean_acc_tr = np.mean(acc_tr, axis=0, keepdims=True).T  # Media
        std_acc_tr = np.std(acc_tr, axis=0, keepdims=True).T  # sqm (radice varianza)
        mean_acc_vl = np.mean(acc_vl, axis=0, keepdims=True).T  # Media
        std_acc_vl = np.std(acc_vl, axis=0, keepdims=True).T  # sqm (radice varianza)
    else:
        mean_error_MEE_tr = np.mean(errors_MEE_tr, axis=0, keepdims=True).T  # Media
        std_error_MEE_tr = np.std(errors_MEE_tr, axis=0, keepdims=True).T  # sqm (radice varianza)
        mean_error_MEE_vl = np.mean(errors_MEE_vl, axis=0, keepdims=True).T  # Media
        std_error_MEE_vl = np.std(errors_MEE_vl, axis=0, keepdims=True).T  # sqm (radice varianza)

    print(100 * "-")
    print("Returning model number ", best_idx)
    print(100 * "-")
    print("STATISTICS:")
    if classification:
        print("TR ERR = %3f +- %3f\nTR ACC = %3f +- %3f\nVL ERR = %3f +- %3f\nVL ACC = %3f +- %3f" % (
            mean_err_tr[-1], std_err_tr[-1], mean_acc_tr[-1],
            std_acc_tr[-1], mean_err_vl[-1], std_err_vl[-1],
            mean_acc_vl[-1], std_acc_vl[-1]))
    else:
        print("TR ERR = %3f +- %3f\nTR ACC = %3f +- %3f\nVL ERR = %3f +- %3f\nVL ACC = %3f +- %3f" % (
            mean_err_tr[-1], std_err_tr[-1], mean_error_MEE_tr[-1],
            std_error_MEE_tr[-1], mean_err_vl[-1], std_err_vl[-1],
            mean_error_MEE_vl[-1], std_error_MEE_vl[-1]))

    print(100 * "-")
    print("\n")
    if classification:
        return best_mlp, mean_err_tr, std_err_tr, mean_acc_tr, std_acc_tr, mean_err_vl, std_err_vl, mean_acc_vl, std_acc_vl
    else:
        return best_mlp, mean_err_tr, std_err_tr, mean_error_MEE_tr, std_error_MEE_tr, mean_err_vl, std_err_vl, mean_error_MEE_vl, std_error_MEE_vl


"Implementato direttamente nelle tecniche di validazione..."


def gridSearch(n_features, X_tr, T_tr, X_vl, T_vl, n_epochs, hidden_act, output_act, eta_values, alfa_values,
               hidden_values, weight_values, lambda_values, n_trials):
    best_vl_error = 1e10
    best_mlp = None
    best_eta = 0
    best_alfa = 0
    best_hidden = 0
    best_weight = 0
    best_lambda = 0
    best_mean_err_tr = np.zeros((n_epochs, 1))
    best_std_err_tr = np.zeros((n_epochs, 1))
    best_mean_acc_tr = np.zeros((n_epochs, 1))
    best_std_acc_tr = np.zeros((n_epochs, 1))
    best_mean_err_vl = np.zeros((n_epochs, 1))
    best_std_err_vl = np.zeros((n_epochs, 1))
    best_mean_acc_vl = np.zeros((n_epochs, 1))
    best_std_acc_vl = np.zeros((n_epochs, 1))

    for eta in eta_values:
        for alfa in alfa_values:
            for hidden in hidden_values:
                for weight in weight_values:
                    for lambd in lambda_values:
                        print(100 * '-')
                        print("Provo eta=%s alfa=%s #hidden=%s weight=%s lambda = %s" % (
                            eta, alfa, hidden, weight, lambd))

                        mlp, mean_err_tr, std_err_tr, mean_acc_tr, std_acc_tr, mean_err_vl, std_err_vl, mean_acc_vl, std_acc_vl = run_trials(
                            n_features, X_tr, T_tr,
                            X_vl, T_vl, n_epochs, hidden_act,
                            output_act, eta, alfa, hidden,
                            weight, lambd, n_trials)

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
        best_eta, best_alfa, best_hidden, best_weight, best_lambda))
    print("STATISTICS:")
    print("TR ERR = %3f +- %3f\nTR ACC = %3f +- %3f\nVL ERR = %3f +- %3f\nVL ACC = %3f +- %3f" % (
        best_mean_err_tr[-1], best_std_err_tr[-1], best_mean_acc_tr[-1],
        best_std_acc_tr[-1], best_mean_err_vl[-1], best_std_err_vl[-1],
        best_mean_acc_vl[-1], best_std_acc_vl[-1]))
    print("MIGLIOR MODELLO DEL GRUPPO:")
    print("TR ERR = %3f TR ACC = %3f VL ERR = %3f VL ACC = %3f" % (best_mlp.errors_tr[-1], best_mlp.accuracies_tr[-1],
                                                                   best_mlp.errors_vl[-1], best_mlp.accuracies_vl[-1]))

    return best_mlp, best_mean_err_tr, best_std_err_tr, best_mean_acc_tr, best_std_acc_tr, best_mean_err_vl, \
           best_std_err_vl, best_mean_acc_vl, best_std_acc_vl
