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