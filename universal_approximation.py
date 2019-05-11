#!/usr/bin/python3
# Multilayer Perceptron to learn "f" in "f(x,y,z)=t",
#                                       t=x if z>0
#                                       t=y else

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten

# --------------- function definition --------------


def zMax(x):
    array = []
    for a in x:
        array.append(a[2] > 0 and a[0] or a[1])
    return np.array(array)


# -------------- define model parameters --------------

new_samples = 20
n_samples = 100
n_epochs = 200
batch_size = 30


# ------ define and fit models ------
#  ----------------------
#       perceptron Model
# -----------------------------------

perceptronModel = Sequential([
    Dense(6, input_dim=3, activation='sigmoid'),
    Dense(12, activation='sigmoid'),
    Dense(6, activation='sigmoid'),
    Dense(1)
])

perceptronModel.compile(
    loss='mae',
    optimizer='adam',
    metrics=['accuracy']
)

# --------------------------
#       ReLU Model
# --------------------------
reluModel = Sequential([
    Dense(8, input_dim=3, activation='relu'),
    Dense(8, activation='relu'),
    Dense(4, activation='relu'),
    Dense(1)
])

reluModel.compile(
    loss='mae',
    optimizer='adam',
    metrics=['accuracy']
)

# -------------- training phase -----------------

X = np.random.randint(10, size=(n_samples, 3))
Y = zMax(X)

perceptronModel.fit(X, Y, epochs=n_epochs)
reluModel.fit(X, Y, epochs=n_epochs)

# ---------------- test phase -----------------

new_X = np.random.randint(10, size=(new_samples, 3))
perceptronPred_Y = perceptronModel.predict(new_X)
perceptronPred_Y = reluModel.predict(new_X)
new_Y = zMax(new_X)

perceptronScore = perceptronModel.evaluate(new_X, new_Y)
reluScore = reluModel.evaluate(new_X, new_Y)

print("-----------------\n\
Perceptron score |\n\
-----------------\n", perceptronScore)
print("-----------\n\
reLU score |\n\
-----------\n", reluScore, "\n")
