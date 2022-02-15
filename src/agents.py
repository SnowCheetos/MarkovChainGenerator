import numpy as np
import tensorflow as tf

def to_prob(val, action_space):
    probs = np.zeros(action_space.shape)
    probs[np.where(action_space == val)] = 1
    return tf.constant(probs)

def to_action(prob, action_space):
    if len(prob.shape) == 1:
        return action_space[np.argmax(prob)]
    else:
        return np.array([action_space[np.argmax(p)] for p in prob])

def random_walks(graph, init, steps, action_space):
    graph = tf.constant(graph)
    history = [to_prob(init, action_space)]
    for i in range(steps):
        prob = tf.linalg.matvec(graph, history[-1])
        randc = np.random.choice(action_space, size = 1, p = prob / tf.reduce_sum(prob).numpy())
        history.append(to_prob(randc, action_space))
        print(f"{i+1}/{steps} step completed.", end = "\r", flush = True)
    return to_action(np.array(history), action_space)

def run(graph, init, steps, action_space):
    graph = tf.constant(graph)
    history = [to_prob(init, action_space)]
    for i in range(steps):
        history.append(tf.linalg.matvec(graph, history[-1]))
        print(f"{i+1}/{steps} step completed.", end = "\r", flush = True)
    return history[-1]