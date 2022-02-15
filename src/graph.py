import numpy as np
from numba import njit, prange

@njit(parallel = True)
def make_graph(data, transition_space):
    graph = np.zeros((len(transition_space), len(transition_space)))
    for i in prange(1, len(data)):
        for j in prange(len(transition_space)):
            for k in prange(len(transition_space)):
                if data[i-1] == transition_space[j] and data[i] == transition_space[k]:
                    graph[j, k] += 1
    for i in prange(len(graph)):
        if np.sum(graph[i]) > 0:
            graph[i] = graph[i] / np.sum(graph[i])
    return graph

def encode(data, precision = 3):
    return (np.round(data, precision)*100 - 100)*(10**(precision - 1))

def decode(data, precision = 3):
    return (data/(10**(precision - 1)) + 100)/100

def calc_transition_space(data):
    return np.unique(data)