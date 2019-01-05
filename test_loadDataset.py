from LoadDataset import *


X,T = load_cup_dataset("ML-CUP18-TR.csv")
print("X's shape = ",X.shape)
print("T's shape = ",T.shape)
print("X =\n", X[:10])
print("T=\n", T[:10])