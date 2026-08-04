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
