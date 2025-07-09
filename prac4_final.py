import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('./train.csv')
data = np.array(data)
X = data[:, 1:].T / 255.0
Y = data[:, 0]
m = X.shape[1]

data1 = pd.read_csv('./test.csv')
data1 = np.array(data1)
X1 = data1[:, :].T / 255.0
m1 = X1.shape[1]

def one_hot(Y):
    one_hot_Y = np.zeros((10, Y.size))
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y

Y_one_hot = one_hot(Y)

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def deriv_leaky_relu(z, alpha=0.01):
    return np.where(z > 0, 1, alpha)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def init_params():
    W1 = np.random.randn(128, 784) * np.sqrt(2 / 784)
    b1 = np.zeros((128, 1))
    W2 = np.random.randn(64, 128) * np.sqrt(2 / 128)
    b2 = np.zeros((64, 1))
    W3 = np.random.randn(10, 64) * np.sqrt(2 / 64)
    b3 = np.zeros((10, 1))
    return W1, b1, W2, b2, W3, b3

def forward(X, W1, b1, W2, b2, W3, b3):
    Z1 = W1 @ X + b1
    A1 = leaky_relu(Z1)
    Z2 = W2 @ A1 + b2
    A2 = leaky_relu(Z2)
    Z3 = W3 @ A2 + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

def back_prop(X, Y, Z1, A1, Z2, A2, A3, W1, W2, W3):
    m = X.shape[1]
    dZ3 = A3 - Y
    dW3 = (dZ3 @ A2.T) / m
    db3 = np.sum(dZ3, axis=1, keepdims=True) / m
    dZ2 = (W3.T @ dZ3) * deriv_leaky_relu(Z2)
    dW2 = (dZ2 @ A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = (W2.T @ dZ2) * deriv_leaky_relu(Z1)
    dW1 = (dZ1 @ X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    return dW1, db1, dW2, db2, dW3, db3

def get_accuracy(predictions, labels):
    return np.mean(predictions == labels)

def compute_loss(A3, Y):
    m = Y.shape[1]
    return -np.sum(Y * np.log(A3 + 1e-8)) / m

def train(X, Y, Y_raw, epochs, batch_size, alpha):
    W1, b1, W2, b2, W3, b3 = init_params()
    m = X.shape[1]
    for epoch in range(epochs):
        permutation = np.random.permutation(m)
        X_shuffled = X[:, permutation]
        Y_shuffled = Y[:, permutation]
        Y_raw_shuffled = Y_raw[permutation]
        for i in range(0, m, batch_size):
            X_batch = X_shuffled[:, i:i+batch_size]
            Y_batch = Y_shuffled[:, i:i+batch_size]
            Z1, A1, Z2, A2, Z3, A3 = forward(X_batch, W1, b1, W2, b2, W3, b3)
            dW1, db1, dW2, db2, dW3, db3 = back_prop(X_batch, Y_batch, Z1, A1, Z2, A2, A3, W1, W2, W3)
            W1 -= alpha * dW1
            b1 -= alpha * db1
            W2 -= alpha * dW2
            b2 -= alpha * db2
            W3 -= alpha * dW3
            b3 -= alpha * db3
        if epoch % 5 == 0 or epoch == epochs - 1:
            _, _, _, _, _, A3_full = forward(X, W1, b1, W2, b2, W3, b3)
            predictions = np.argmax(A3_full, axis=0)
            acc = get_accuracy(predictions, Y_raw)
            loss = compute_loss(A3_full, Y)
            print(f"Epoch {epoch}: Accuracy = {acc:.4f}, Loss = {loss:.4f}")
    return W1, b1, W2, b2, W3, b3

def check_random(X1, W1, b1, W2, b2, W3, b3, index):
    if index < 0 or index >= X1.shape[1]:
        raise ValueError("Index out of bounds for the test dataset.")
    test_image = X1[:, index : index + 1]

    _, _, _, _, _, A3 = forward(test_image, W1, b1, W2, b2, W3, b3)
    prediction = np.argmax(A3, axis=0)[0]
    image = test_image.reshape(28, 28)

    plt.imshow(image, cmap='gray')
    plt.title(f'Predicted Digit: {prediction}')
    plt.axis('off')
    plt.show()
    print(f"Random Test Image Index: {index}")
    print(f"Predicted Label: {prediction}")

W1, b1, W2, b2, W3, b3 = train(X, Y_one_hot, Y, epochs=15, batch_size=64, alpha=0.1)

check_random(X1, W1, b1, W2, b2, W3, b3, index = 83)