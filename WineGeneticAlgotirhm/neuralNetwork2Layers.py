import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def forward_pass(weights, inputs, input_size, hidden_size1=5, hidden_size2=5, output_size=1):

    if inputs.ndim == 1:
        inputs = inputs.reshape(1, -1)

    W1_end = input_size * hidden_size1
    W1 = weights[:W1_end].reshape((input_size, hidden_size1))
    b1_end = W1_end + hidden_size1
    b1 = weights[W1_end:b1_end]

    W2_end = b1_end + hidden_size1 * hidden_size2
    W2 = weights[b1_end:W2_end].reshape((hidden_size1, hidden_size2))
    b2_end = W2_end + hidden_size2
    b2 = weights[W2_end:b2_end]

    W3_end = b2_end + hidden_size2 * output_size
    W3 = weights[b2_end:W3_end].reshape((hidden_size2, output_size))
    b3 = weights[W3_end:]

    hidden1 = sigmoid(np.dot(inputs, W1) + b1)
    hidden2 = sigmoid(np.dot(hidden1, W2) + b2)
    output = sigmoid(np.dot(hidden2, W3) + b3)
    return output

def fitness_func_factory(X, Y, input_size, hidden_size1=5, hidden_size2=5, output_size=1):

    def fitness_func(ga_instance, solution, solution_idx):
        predictions = forward_pass(solution, X, input_size, hidden_size1, hidden_size2, output_size)
        predictions = np.round(predictions).flatten()
        accuracy = np.mean(predictions == Y)
        return accuracy

    return fitness_func

def on_generation(ga_instance):
    if ga_instance.generations_completed % 10 == 0:
        best_fitness = np.max(ga_instance.last_generation_fitness)
        print(f"Generation {ga_instance.generations_completed}: Best Fitness = {best_fitness:.4f}")
