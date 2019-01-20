from Validation.KFoldClassification import *
from Validation.KFoldRegression import *

"""
Effettua la KFOLD CV.
Restituisce la configurazione migliore degli iperparametri.
    
NOTA1: NON EFFETTUA IL RETRAINING FINALE SULL'INTERO (TR+VL) set
NOTA2: NON EFFETTUA LO SPLITTING DATI INTERNO (TR/VL) E TEST(INTERNO)
ENTRAMBI FATTI SUI FILE "FINALI" (vedi file test_kfoldCV.py e test_HoldOut.py)

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
:param title_plot : Titolo da assegnare al plot
:param save_path_plot : Path in cui salvare i plot
:param save_path_results : Path in cui salvare i risultati dei vari folds scritti in un file

:return best_eta,best_alfa,best_hidden,best_lambda,best_weight : migliore configurazione trovata
:return best_mean_vl_error,best_std_vl_error : media e std del miglior validation error trovato
    - Classificazione => uso MSE
    - Regressione => uso MEE

REMEMBER=> Hyperparameter (esaustiva):
    # eta
    # alfa
    # hidden (in generale numero neuroni)
    # weight (Initialize weights by random values near zero)
    # lambda
    # on-line/batch/miniBatch !
    # stopping criteria (piu per CM...)
    # n_trials per multipla minima (forzatura)
    # n_epochs non fisso (forzatura)
    # k della kcrossvalidation (forzatura)

"""


def kFoldCV(n_features, X, T, k, n_epochs, hidden_act, output_act, eta_values, alfa_values, hidden_values,
            weight_values, lambda_values, n_trials, classification=True, shuffle=True, title_plot = "ML CUP", save_path_plot="../Plots/cup",
            save_path_results="../Results_CSV/cup"):


   if classification:
       return KFoldClassification(n_features, X, T, k, n_epochs, hidden_act, output_act, eta_values, alfa_values, hidden_values,
                                  weight_values, lambda_values, n_trials, shuffle=shuffle, title_plot = title_plot,
                                  save_path_plot=save_path_plot,save_path_results=save_path_results)

   else:
       return KFoldRegression(n_features, X, T, k, n_epochs, hidden_act, output_act, eta_values, alfa_values, hidden_values,
                              weight_values, lambda_values, n_trials, shuffle=shuffle, title_plot = title_plot,
                              save_path_plot=save_path_plot,save_path_results=save_path_results)