"""
Questo file contiene varie funzioni utilizzate dagli altri moduli del progetto.

"""
#TODO: RICONTROLLARE PER BENE QUESTE FUNZIONI E CERCARE FORMULA PER MEE!!!!!!!!!!!!!!!!!!

import numpy as np

""""
Aggiunge il bias ad una generica matrice M.
A partire dalla matrice M, costruisce la matrice M' = [1 | M].
La matrice M passata in input non viene modificata.

:param M : Matrice di origine tramite cui comporre la nuova matrice M'.
:return : La matrice M' = [1 | M].
"""


def addBias(M):

    M_bias = np.ones((M.shape[0],M.shape[1]+1))
    M_bias[:,1:] = M
    return M_bias

"""
Rimuove il bias da una generica matrice M.
A partire dalla matrice M = [1 | M'], ricostruisce la matrice M' e la restituisce in output.
La matrice M non viene modificata.

:param M : Matrice a cui rimuovere il bias.
:return : Matrice M'.
"""


def removeBias(M):
    M_prime = M[:,1:]
    return M_prime

"""
Costruisce una matrice random di dimensione (n_rows * n_cols) i cui elementi sono
compresi nel range [range_start, range_end].
Se fan_in == True, allora gli elementi sono compresi nell'intervallo [range_start / (n_cols -1), range_end / (n_cols-1)]
Per le matrici dei pesi:
    n_rows = # elementi nel layer corrente 
    n_cols = # elementi nel layer precedente + colonna del bias

Gli elementi sono generati a partire da una distribuzione uniforme sull'intervallo specificato.

--------------------------
NOTA: LA MATRICE DEI PESI CONTIENE GIA' IL BIAS????   ;)
-------------------------

:param n_rows : numero di righe
:param n_cols : numero di colonne (compreso il bias!!)
:param fan_in : bool. Se True, allora viene applicato il fan_in.
:param range_start : estremo di inizio dell'intervallo
:param range_end : estremo di fine dell'intervallo 
"""


def init_Weights(n_rows,n_cols,fan_in = False,range_start = -0.7,range_end = 0.7):

    if fan_in:
        M = np.random.uniform(range_start / (n_cols - 1), range_end / (n_cols - 1), (n_rows,n_cols))
    else:
        M = np.random.uniform(range_start, range_end, (n_rows,n_cols))

    return M


"""
Calcola l'errore misurato mediante l'MSE.

MSE calcolato come ([frobenius_norm(T - OUT)]**2 / n_examples)

:param T : Matrice dei Target
:param OUT : Matrice degli Output della rete neurale

:return : MSE(T,OUT)
"""


def compute_Error(T,OUT):

    assert T.shape == OUT.shape
    n_examples = T.shape[0]
    return (np.linalg.norm(T - OUT, 'fro') ** 2) / n_examples

"""
Calcola l'accuracy della rete neurale, dati target e predizioni della rete neurale.
L'accuracy è calcola come [ (numero di elementi classificati correttamente) / #esempi]
La matrice OUT contiene le predizioni della rete (0/1 nel caso di classificazione binaria)

:param T : Matrice contenente i target.
:param OUT: Matrice contenente le predizioni della rete
:return Accuracy

"""


def compute_Accuracy_Class(T,OUT):

    assert T.shape == OUT.shape
    n_examples = T.shape[0]

    n_misclass = np.sum((T - OUT)**2)
    n_correct_class = n_examples - n_misclass
    return n_correct_class / n_examples


"""
Calcola l'accuracy misurata come MEE.
DA IMPLEMENTARE... FORMULA CON MATRICI????
"""


def compute_Accuracy_Regr(T,OUT):
    return NotImplementedError("Ancora non implementato!!!")
"""


TEST VARI....
"""
X = np.array([
    [2,3,4],
    [2,3,4]
])

n_row = 4
n_col = 10

W = init_Weights(n_row,n_col,fan_in=True)
print("W = ",W)
print(W.shape)

T = np.array([
    [1,0],
    [0,2]
])

Y = np.array([
    [0,1],
    [2,1]
])

# T - Y =
# [1 -1
# -2  1]
# ==> NORM(T-Y) = 1+1+4+1 = 7

print("Error =", compute_Error(T,Y))

T = np.array([
    [1],
    [0],
    [1],
    [1]
])

Y = np.array([
    [0],
    [1],
    [1],
    [1]])

print("Accuracy =", compute_Accuracy_Class(T,Y))