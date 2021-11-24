import numpy as np

def sigmoid(z):
    return 1.0/(1 + np.exp(-z))

def loss(y_hat, y):
    return -np.mean(y*(np.log(y_hat)) - (1-y)*np.log(1-y_hat))

def cost(W, b, m, lossF):
    return 1/m * (np.sum(lossF))

def get_gradient(X, y, y_hat):
    m = X.shape[0]

    dz = (y_hat - y)
    dw = (1/m)* np.dot(X.T, dz)
    db = (1/m)* np.sum(dz)

    return dw, db

def train(X, y, epochs, lr):
    m, n = X.shape
    y = y.reshape(m, 1)

    b = 0
    W = np.zeros((n, 1))

    for epoch in range(epochs):
        z = np.dot(X, W) + b
        a = sigmoid(z)

        dw, db = get_gradient(X, y, a)

        W -= lr*dw
        b -= lr*db

        cost_ = cost(W, b, m, loss(a, y))
        print(f"Epoch: {epoch+1:2d} |---| Cost: {cost_}")

    return W, b, cost_

def predict(X, W, b):
    y_pred = sigmoid(np.dot(X, W) + b)

    class_pred = [1 if i > 0.5 else 0 for i in y_pred]

    return np.array(class_pred)
