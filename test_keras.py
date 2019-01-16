from HoldOut import *
from KFoldCV import *
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD

P = loadMatrixFromFile("DatasetTrVl.csv")
X = P[:, : - 2]
T = P[:, -2:]

(trainX, testX, trainY, testY) = train_test_split(X,T, test_size=0.25, random_state=42)


model = Sequential()
model.add(Dense(20,input_dim=trainX.shape[1],activation="tanh"))
model.add(Dense(2,activation='linear'))

INIT_LR = 0.01
EPOCHS = 1000

# compile the model using SGD as our optimizer and categorical
# cross-entropy loss (you'll want to use binary_crossentropy
# for 2-class classification)
print("[INFO] training network...")
opt = SGD(lr=INIT_LR,momentum=0.)
model.compile(loss="mean_squared_error", optimizer=opt)


# train the neural network
H = model.fit(trainX, trainY, validation_data=(testX, testY),
	epochs=EPOCHS, batch_size=trainX.shape[0])

N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
#plt.plot(N, H.history["acc"], label="train_acc")
#plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend()
plt.show()