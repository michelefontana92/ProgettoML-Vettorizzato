from Validation.KFoldCV import *
from Monks.Monk import *
from matplotlib import pyplot as plt
from MLP.Activation_Functions import *


eta_values = [0.8,0.7,0.4]
alfa_values = [0.8]
hidden_values =[3]
weight_values = [0.7]
lambda_values = [0]
n_epochs = 500
n_trials = 10
k=5
n_features = 17
classifications = True

X,T = load_monk("../Datasets/monks-1.test")

"KFOLD CV"  # Classificazione
best_eta,best_alfa,best_hidden,best_lambda,best_weight,best_mean_vl_error,best_std_vl_error=kFoldCV(n_features,X,T,k,500,
    TanhActivation(),SigmoidActivation(),
    eta_values,alfa_values,hidden_values,weight_values,lambda_values,n_trials, classifications,title_plot="Monk 1",save_path_plot="../Plots/monk1",
                                                                                                    save_path_results="../Results_CSV/monk1")
