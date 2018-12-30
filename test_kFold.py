from KFold import *

X = np.array([
    [1,1,1],
    [2,2,2],
    [3,3,3],
    [4,4,4],
    [5,5,5]
])
Y = np.ones((5,1))

folds = kFold(X,Y,4)
X_i ,Y_i = get_fold(X,Y,folds,3)
print(X_i)
print(Y_i)

split_dataset(X,Y,folds,3)