"""
Questo file contiene la classe MLP preposta ad implementare la rete neurale;
- Ogni elemento e Vettoriazzato
- Non necessita di classi come Neuron o Layers
- Usa le classi/file: Utility & ActivationFunction
- MLP avra un bool per effettuare operazioni di classificazione oppure di regressione: classification

"""

from Trainers.LineSearch import *
from Trainers.TrainBackprop import *

class MLP:
    "Costruttore classe con stati; NOTA: Inseriti Pesi con bias"

    def __init__(self, n_feature, n_hidden, n_output, activation_h, activation_o, eta=0.1, lambd=0, alfa=0.75,
                 fan_in_h=True, range_start_h=-0.7, range_end_h=0.7, fan_in_o=True, range_start_o=-0.7, range_end_o=0.7,
                 classification=True,trainer = TrainBackprop()):
        # Valori scalari
        # self.n_input = n_input  # righe di X
        self.n_feature = n_feature  # colonne di X, oppure neuroni input
        self.n_hidden = n_hidden
        self.n_output = n_output

        # Vettorizzato: Matrici
        ## NOTA: Indico gli indici delle dimensioni delle matrici/vettori
        self.W_h = init_Weights(n_hidden, n_feature, fan_in_h, range_start_h,
                                range_end_h)  # (n_neuroni_h x n_feature +1)
        self.W_o = init_Weights(n_output, n_hidden, fan_in_o, range_start_o,
                                range_end_o)  # (n_neuroni_o x n_neuroni_h +1)
        self.Out_h = None  # (n_esempi x n_neuroni_h)
        self.Out_o = None  # (n_esempi x n_neuroni_o) //Per Monk  un vettore
        self.Net_h = None  # (n_esempi x n_neuroni_h)
        self.Net_o = None  # (n_esempi x n_neuroni_o) //Per Monk un vettore

        # Si specifica il tipo di f. attivazione dei neuroni
        self.activation_h = activation_h
        self.activation_o = activation_o

        # Hyperparameter!
        self.eta = eta  # learning rate
        self.lambd = lambd  # regolarizzazione-penalityTerm
        self.alfa = alfa  # momentum

        # Lista per avere il plot LC, Accuracy(class->(N-num_err)/N), regress->MEE
        self.errors_tr = []  # MSE/num_epoche sul TR
        self.accuracies_tr = []  # Accuracy/num_epoche sul TR
        self.errors_vl = []  # MSE/num_epoche sul VL
        self.accuracies_vl = []  # Accuracy/num_epoche sul VL
        self.errors_mee_tr = []  # MEE sul TR
        self.errors_mee_vl = []  # MEE sulla VL

        # Servono nella fase di train->backperopagation; delta vecchio dei pesi hidden e output
        self.dW_o_old = np.zeros(self.W_o.shape)
        self.dW_h_old = np.zeros(self.W_h.shape)

        # Bool per Classificazione/Regressione
        self.classification = classification

        self.trainer = trainer

    "FeedFoward: X con bias"

    def feedforward(self, X):
        # Calcolo hidden layer
        self.Net_h = np.dot(X, self.W_h.T)
        self.Out_h = self.activation_h.compute_function(self.Net_h)  # Output_h=f(Net_h)

        # Calcolo output layer
        Out_h_bias = addBias(self.Out_h)
        self.Net_o = np.dot(Out_h_bias, self.W_o.T)
        self.Out_o = self.activation_o.compute_function(
            self.Net_o)  # Output_o=f(Net_o)=>Classificazione rete; Per Monk vettore

    "Backpropagation: X con bias"

    def backpropagation(self, X, T):
        assert T.shape == self.Out_o.shape

        # Calcolo della f'(Net_o), calcolo delta_neuroneOutput, calcolo delta peso
        ## NOTA: vedere slide Backprop.
        grad_f_o = self.activation_o.compute_function_gradient(self.Out_o)
        diff = (T - self.Out_o)
        delta_o = np.multiply(diff, grad_f_o)  # elemento-per-elemento
        Out_h_bias = addBias(self.Out_h)
        delta_W_o = np.dot(delta_o.T, Out_h_bias)

        # Calcolo della f'(Net_h), calcolo delta_o*pesi_interessati, calcolo delta hidden layer
        ## NOTA: vedere slide Backprop.
        grad_f_h = self.activation_h.compute_function_gradient(self.Out_h)
        W_o_nobias = removeBias(self.W_o)
        sp_h = np.dot(delta_o, W_o_nobias)
        delta_h = np.multiply(sp_h, grad_f_h)  # elemento-per-elemento
        delta_W_h = np.dot(delta_h.T, X)

        return delta_W_o / X.shape[0], delta_W_h / X.shape[0]


    "Train usando Backprop: The Basic Alg. (vedi Slide Corso ML)"
    """
    def train(self, X, T, X_val, T_val, n_epochs=1000, eps=10 ^ (-3), threshold=0.5, suppress_print=False, opt_a1=False):
        assert X.shape[0] == T.shape[0]
        # 1) Init pesi e iperparametri // fatto nel costruttore

        # 4) Condizioni di arresto
        error_MSE = 100
        for epoch in range(n_epochs):

            if (error_MSE < eps):
                break

            # 2) Effettuo la feedfoward;
            #   calcolo MSE class/regress (Learning Curve TR/VL), accuracy(accuracy curve TR/VL, class)/ MEE (regress);
            #   calcolo delta_W usando backpropagation
            self.feedforward(X)
            # print "n_output:", self.n_output
            # print "OUT_o", self.Out_o.shape
            # print "Target", T.shape
            error_MSE = compute_Error(T, self.Out_o)
            if self.classification:
                accuracy = compute_Accuracy_Class(T, convert2binary_class(self.Out_o, threshold))
                self.errors_tr.append(error_MSE)
                self.accuracies_tr.append(accuracy)
            else:
                error_MEE = compute_Regr_MEE(T, self.Out_o)
                self.errors_tr.append(error_MSE)
                self.errors_mee_tr.append(error_MEE)

            dW_o, dW_h = self.backpropagation(X, T)

            # CALCOLO IL VALIDATION ERROR
            self.feedforward(X_val)
            error_MSE_val = compute_Error(T_val, self.Out_o)
            if self.classification:
                accuracy_val = compute_Accuracy_Class(T_val, convert2binary_class(self.Out_o, threshold))
                self.errors_vl.append(error_MSE_val)
                self.accuracies_vl.append(accuracy_val)
            else:
                error_MEE_val = compute_Regr_MEE(T_val, self.Out_o)
                self.errors_vl.append(error_MSE_val)
                self.errors_mee_vl.append(error_MEE_val)

            # 3) Upgrade weights

            # TODO: LINE SEARCH-> CM+ML...
            # TODO: A0
            # TODO: A1
            # TODO: A2

            # A1
            if opt_a1:
                loss = error_MSE # lamba=0
                self.eta = AWLS(self, X, T, loss, -dW_h, -dW_o,0)

            #self.eta = AWLS(self,X,T,error_MSE,dW_h,dW_o)
            dW_o_new = self.eta * dW_o + self.alfa * self.dW_o_old
            self.W_o = self.W_o + dW_o_new - (self.lambd * self.W_o)

            dW_h_new = self.eta * dW_h + self.alfa * self.dW_h_old
            self.W_h = self.W_h + dW_h_new - (self.lambd * self.W_h)

            self.dW_o_old = dW_o_new
            self.dW_h_old = dW_h_new

            # per stampa per ogni epoca
            if not suppress_print:
                if self.classification:
                    print(
                        "Epoch %s/%s) TR Error(MSE) : %s VL Error(MSE) : %s TR Accuracy((N-num_err)/N) : %s VL Accuracy((N-num_err)/N) : %s" % (
                            epoch + 1, n_epochs, error_MSE, error_MSE_val, accuracy, accuracy_val))
                else:
                    print(
                        "Epoch %s/%s) TR Error(MSE) : %s VL Error(MSE) : %s TR (MEE) : %s VL ((MEE) : %s" % (
                            epoch + 1, n_epochs, error_MSE, error_MSE_val, error_MEE, error_MEE_val))

        # CALCOLO ERRROR E ACCURACY/MEE FINALI (metto nelle liste)
        self.feedforward(X)
        error_MSE = compute_Error(T, self.Out_o)
        if self.classification:
            accuracy = compute_Accuracy_Class(T, convert2binary_class(self.Out_o, threshold))
            self.errors_tr.append(error_MSE)
            self.accuracies_tr.append(accuracy)
        else:
            error_MEE = compute_Regr_MEE(T, self.Out_o)
            self.errors_tr.append(error_MSE)
            self.errors_mee_tr.append(error_MEE)

        self.feedforward(X_val)
        error_MSE_val = compute_Error(T_val, self.Out_o)
        if self.classification:
            accuracy_val = compute_Accuracy_Class(T_val, convert2binary_class(self.Out_o, threshold))
            self.errors_vl.append(error_MSE_val)
            self.accuracies_vl.append(accuracy_val)
        else:
            error_MEE_val = compute_Regr_MEE(T_val, self.Out_o)
            self.errors_vl.append(error_MSE_val)
            self.errors_mee_vl.append(error_MEE_val)

        # per stampa di risultato finale
        if suppress_print:
            if self.classification:
                print(
                    "Final Results_CSV: TR Error(MSE) : %s VL Error(MSE) : %s TR Accuracy((N-num_err)/N) : %s VL Accuracy((N-num_err)/N) : %s" % (
                        self.errors_tr[-1], self.errors_vl[-1], self.accuracies_tr[-1], self.accuracies_vl[-1]))
            else:
                print(
                    "Final Results_CSV: TR Error(MSE) : %s VL Error(MSE) : %s TR (MEE) : %s VL (MEE) : %s" % (
                        self.errors_tr[-1], self.errors_vl[-1], self.errors_mee_tr[-1], self.errors_mee_vl[-1]))
    """

    "Classificazione: predizione"

    def predict_class(self, X, treshold=0.5):
        self.feedforward(X)
        predictions = np.zeros(self.Out_o.shape)
        predictions[self.Out_o >= treshold] = 1
        return predictions

    "Regressione: predizione"

    def predict_value(self, X):
        self.feedforward(X)
        return self.Out_o
