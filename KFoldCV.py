from KFold import *
from GridSearch import *

"""
Effettua lo shuffling delle due matrici X e T
"""
def shuffle_matrices(X,T):
    M = np.concatenate((X, T), axis=1)
    np.random.shuffle(M)
    X_shuffled = M[:,:X.shape[1]]
    T_shuffled = M[:,-T.shape[1]:]
    return np.reshape(X_shuffled,(-1,X.shape[1])), np.reshape(T_shuffled,(-1,T.shape[1]))

"""
Effettua la KFOLD CV.
Restituisce la configurazione migliore degli iperparametri.
    
NOTA: NON EFFETTUA IL RETRAINING FINALE SULL'INTERO (TR+VL) set

:param X: Matrice di input
:param T: Matrice di target
:param k: Numero di folds
:param n_epochs : numero di epoche di training
:param hidden_act: Funzione attivazione hidden layer
:param output_act: Funzione attivazione output layer
:param eta_values: Insieme di valori da provare per eta
:param alfa_values: Insieme di valori da provare per alfa
:param hidden_values: Insieme di valori da provare per il numero di hidden units
:param weight_values: Insieme di valori da provare per l'intervallo di inizializzazione dei pesi
:param lambda_values: Insieme di valori da provare per lambda
:param n_trials : Numero di volte che viene effettuato il train (multiple minima)

:return best_eta,best_alfa,best_hidden,best_lambda,best_weight : migliore configurazione trovata
:return best_mean_vl_error,best_std_vl_error : media e std del miglior validation error trovato

"""
def kFoldCV(n_features,X,T,k,n_epochs,hidden_act,output_act, eta_values, alfa_values,hidden_values, weight_values, lambda_values,n_trials,shuffle=True):

    if shuffle:
        X,T = shuffle_matrices(X,T)

    folds = kFold(X,T,k)

    best_mean_vl_error = 1e10
    best_std_vl_error = 0

    best_eta = 0
    best_alfa = 0
    best_hidden = 0
    best_weight = 0
    best_lambda = 0

    """
    PER OGNI CONFIGURAZIONE...
    """
    for eta in eta_values:
        for alfa in alfa_values:
            for hidden in hidden_values:
                for weight in weight_values:
                    for lambd in lambda_values:
                        print(100 * '-')
                        print("Provo eta=%s alfa=%s #hidden=%s weight=%s lambda = %s" % (
                            eta, alfa, hidden, weight, lambd))


                        """
                        PER OGNI FOLD: ...
                        """

                        """
                        Tengo traccia degli errori di vl per ogni fold"
                        """
                        vl_errors = np.zeros((k,1))
                        for (idx,fold_for_vl) in enumerate(folds):

                            print("FOLD ",idx+1)
                            X_tr,T_tr,X_vl,T_vl = split_dataset(X,T,folds,idx)

                            mlp, mean_err_tr, std_err_tr, mean_acc_tr, std_acc_tr, mean_err_vl, std_err_vl, mean_acc_vl, std_acc_vl = run_trials(
                                n_features,X_tr, T_tr, X_vl, T_vl, n_epochs,hidden_act, output_act, eta, alfa, hidden, weight, lambd, n_trials)

                            vl_errors[idx] = mean_err_vl[-1]
                            print("FOLD %s: VL ERROR = %3f"%(idx+1,mean_err_vl[-1]))
                            print(100*"-")
                            print()

                        """
                        Calcolo VL error medio e std fatti sui fold per questa configurazione
                        """
                        mean_vl_error_fold = np.mean(vl_errors)
                        std_vl_error_fold = np.std(vl_errors)
                        print("VL ERROR OVER ALL FOLDS: %3f +- %3f"%(mean_vl_error_fold,std_vl_error_fold))

                        """
                        Controllo se VL error medio ottenuto è il migliore al momento.
                        """

                        if mean_vl_error_fold < best_mean_vl_error:

                            print("\nTROVATO ERRORE MIGLIORE = %3f -> %3f\n" % (best_mean_vl_error, mean_vl_error_fold))
                            best_mean_vl_error = mean_vl_error_fold
                            best_std_vl_error = std_vl_error_fold

                            best_eta = eta
                            best_alfa = alfa
                            best_hidden = hidden
                            best_lambda = lambd
                            best_weight = weight

    print()
    print(100*"-")
    print("CONFIGURAZIONE SCELTA eta=%s alfa=%s #hidden=%s weight=%s lambda=%s" % (
        best_eta, best_alfa, best_hidden, best_weight, best_lambda))
    print("BEST VL ERROR: %3f +- %3f" % (best_mean_vl_error,best_std_vl_error))
    print(100*"-")
    print("FINE K_FOLD CV")
    print(100 * "-")
    print()
    return best_eta,best_alfa,best_hidden,best_lambda,best_weight,best_mean_vl_error,best_std_vl_error

