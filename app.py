from flask import Flask, request, jsonify, send_file
import numpy as np
import random, gzip, urllib.request, os, ssl

np.random.seed(42)
random.seed(42)

WEIGHTS_FILE = 'mlp_weights.npz'
DATA_DIR     = 'mnist_data'

class NeuralNetwork:
    def __init__(self, layer_sizes, lr=0.001, l2=0.0003, dropout=0.3):
        self.num_layers   = len(layer_sizes) - 1
        self.lr           = lr
        self.initial_lr   = lr
        self.l2           = l2
        self.dropout      = dropout
        self.weights      = []
        self.biases       = []
        self.m_w = []; self.v_w = []
        self.m_b = []; self.v_b = []
        self.beta1 = 0.9; self.beta2 = 0.999
        self.eps = 1e-8; self.t = 0

        for i in range(self.num_layers):
            in_d, out_d = layer_sizes[i], layer_sizes[i+1]
            w = np.random.randn(in_d, out_d) * np.sqrt(2. / in_d)
            b = np.zeros((1, out_d))
            self.weights.append(w);  self.biases.append(b)
            self.m_w.append(np.zeros_like(w)); self.v_w.append(np.zeros_like(w))
            self.m_b.append(np.zeros_like(b)); self.v_b.append(np.zeros_like(b))

    def relu(self, z):    return np.maximum(0, z)
    def relu_d(self, z):  return (z > 0).astype(np.float32)

    def softmax(self, z):
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def forward(self, X, training=True):
        self.a     = [X]
        self.z     = []
        self.masks = []
        act = X
        for i in range(self.num_layers):
            z = act @ self.weights[i] + self.biases[i]
            self.z.append(z)
            if i == self.num_layers - 1:
                act = self.softmax(z)
            else:
                act = self.relu(z)
                if training and self.dropout > 0:
                    mask = (np.random.rand(*act.shape) > self.dropout) / (1 - self.dropout)
                    act *= mask
                    self.masks.append(mask)
                else:
                    self.masks.append(None)
            self.a.append(act)
        return self.a[-1]

    def backward(self, X, y_true):
        m = X.shape[0]; self.t += 1
        dW = [None]*self.num_layers; db = [None]*self.num_layers
        dz = self.a[-1] - y_true
        for i in reversed(range(self.num_layers)):
            l2_term  = (self.l2 / m) * self.weights[i]
            dW[i]    = self.a[i].T @ dz / m + l2_term
            db[i]    = dz.sum(0, keepdims=True) / m
            if i > 0:
                dz = dz @ self.weights[i].T * self.relu_d(self.z[i-1])
                if self.masks[i-1] is not None:
                    dz *= self.masks[i-1]
        for i in range(self.num_layers):
            self.m_w[i] = self.beta1*self.m_w[i] + (1-self.beta1)*dW[i]
            self.m_b[i] = self.beta1*self.m_b[i] + (1-self.beta1)*db[i]
            self.v_w[i] = self.beta2*self.v_w[i] + (1-self.beta2)*dW[i]**2
            self.v_b[i] = self.beta2*self.v_b[i] + (1-self.beta2)*db[i]**2
            mw = self.m_w[i]/(1-self.beta1**self.t)
            mb = self.m_b[i]/(1-self.beta1**self.t)
            vw = self.v_w[i]/(1-self.beta2**self.t)
            vb = self.v_b[i]/(1-self.beta2**self.t)
            self.weights[i] -= self.lr * mw / (np.sqrt(vw)+self.eps)
            self.biases[i]  -= self.lr * mb / (np.sqrt(vb)+self.eps)

    def cosine_lr(self, epoch, total):
        min_lr = self.initial_lr * 0.01
        self.lr = min_lr + 0.5*(self.initial_lr-min_lr)*(1+np.cos(np.pi*epoch/total))

    def predict(self, X):
        return np.argmax(self.forward(X, training=False), axis=1)

    def accuracy(self, X, y):
        return float(np.mean(self.predict(X) == y))

def download_mnist():
    base = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
    files = ['train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz',
             't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
    os.makedirs(DATA_DIR, exist_ok=True)
    ctx = ssl._create_unverified_context()