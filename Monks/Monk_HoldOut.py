from Monks.Monk import *
import numpy as np
from sklearn.model_selection import train_test_split


"""
NOTA: libreria sklearn.model_selection usata solo per fare lo split con shuffle delle righe
      per farlo uso metodo train_test_split()
    ----------------------------------------------------------------------------------------------------------
Prove Split su generica e semplice Matrice M (con stampa valori!)
    ----------------------------------------------------------------------------------------------------------
"""


M = np.array([
    [2, 3, 4],
    [5, 6, 7],
    [8, 9, 10],
    [11, 12, 13],
    [14, 15, 16],
    [17, 18, 19],
    [20, 21, 22],
    [23, 24, 25],
    [26, 27, 28],
    [29, 30, 31]
])
print("M", M)
print("->M size: ", M.shape)
print

# Divido internal set (TR+VL) e TS set
M1_internal, M1_test = train_test_split(M, test_size=0.4, train_size=0.6)  # shuffle=True default
print("M1_internal:", M1_internal)
print("M1_internal size:", M1_internal.shape)
print("M1_test:", M1_test)
print("M1_test:", M1_test.shape)
print

# Divido internal set: TR e VL set
M1_internal_tr, M1_internal_vl = train_test_split(M1_internal, test_size=0.2)  # shuffle=True default, , train_size=0.8
print("M1_internal_tr:", M1_internal_tr)
print("M1_internal_tr size:", M1_internal_tr.shape)
print("M1_internal_vl:", M1_internal_vl)
print("M1_internal_vl:", M1_internal_vl.shape)
print


"""
    ----------------------------------------------------------------------------------------------------------
Prove Split sui Monk
    ----------------------------------------------------------------------------------------------------------
"""
# Carico i dati nelle matrici/vettori con "ripulita" e split Dataset (Xi) + Target (Yi)
X1, Y1 = load_monk("monks-1.test")
X2, Y2 = load_monk("monks-2.test")
X3, Y3 = load_monk("monks-3.test")

# -------------------------------------------------- Prove su Monk1 ---------------------------------------------------
print ("->X1.shape: ", X1.shape)
print

# Divido internal set (TR+VL) e TS set
X1_internal, X1_test = train_test_split(X1, test_size=0.2)  # shuffle=True default, automatic internal -> train_size= 1-test_size
# print("X1_internal:", X1_internal)
print("X1_internal size:", X1_internal.shape)
# print("X1_test:", X1_test)
print("X1_test:", X1_test.shape)
print

# Divido internal set: TR e VL set
X1_internal_tr, X1_internal_vl = train_test_split(X1_internal, test_size=0.25)  # shuffle=True default, automatic TR -> train_size= 1-test_size
# print("X1_internal_tr:", X1_internal_tr)
print("X1_internal_tr size:", X1_internal_tr.shape)
# print("X1_internal_vl:", X1_internal_vl)
print("X1_internal_vl:", X1_internal_vl.shape)
print

# -------------------------------------------------- Prove su Monk2 ---------------------------------------------------
print ("->X2.shape: ", X2.shape)
print

# Divido internal set (TR+VL) e TS set
X2_internal, X2_test = train_test_split(X2, test_size=0.3)  # shuffle=True default,  automatic internal -> train_size= 1-test_size
# print("X2_internal:", X2_internal)
print("X2_internal size:", X2_internal.shape)
# print("X2_test:", X2_test)
print("X2_test:", X2_test.shape)
print

# Divido internal set: TR e VL set
X2_internal_tr, X2_internal_vl = train_test_split(X2_internal, test_size=0.3)  # shuffle=True default, automatic TR -> train_size= 1-test_size
# print("X2_internal_tr:", X2_internal_tr)
print("X2_internal_tr size:", X2_internal_tr.shape)
# print("X2_internal_vl:", X2_internal_vl)
print("X2_internal_vl:", X2_internal_vl.shape)
print

# -------------------------------------------------- Prove su Monk3 ---------------------------------------------------
print ("->X3.shape: ", X3.shape)
print

# Divido internal set (TR+VL) e TS set
X3_internal, X3_test = train_test_split(X3, test_size=0.4)  # shuffle=True default,  automatic internal -> train_size= 1-test_size
# print("X3_internal:", X3_internal)
print("X3_internal size:", X3_internal.shape)
# print("X3_test:", X3_test)
print("X3_test:", X3_test.shape)
print

# Divido internal set: TR e VL set
X3_internal_tr, X3_internal_vl = train_test_split(X3_internal, test_size=0.2)  # shuffle=True default, automatic TR -> train_size= 1-test_size
# print("X3_internal_tr:", X3_internal_tr)
print("X3_internal_tr size:", X3_internal_tr.shape)
# print("X3_internal_vl:", X3_internal_vl)
print("X3_internal_vl:", X3_internal_vl.shape)
print


"""
prove varie...
"""

# Divido internal set: TR e VL set
X_tr = train_test_split(X3, shuffle=True)
print("X_tr:", X_tr)
print
