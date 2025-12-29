import numpy as np

#aktivaciska funkcija
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def forward_pass(weights, inputs, input_size, hidden_size, output_size):
    if inputs.ndim == 1:
        inputs = inputs.reshape(1, -1)

    W1_end = input_size * hidden_size
    W1 = weights[:W1_end].reshape((input_size, hidden_size))

    b1_end = W1_end + hidden_size
    b1 = weights[W1_end:b1_end]

    W2_end = b1_end + hidden_size * output_size
    W2 = weights[b1_end:W2_end].reshape((hidden_size, output_size))

    b2 = weights[W2_end:]

    hidden = sigmoid(np.dot(inputs, W1) + b1)
    output = sigmoid(np.dot(hidden, W2) + b2)
    return output


def fitness_func_factory(X, Y, input_size, hidden_size, output_size):
    # sozdava fitness funkcija koja mozhe da bide povikana od PYGAD bibliotekata

    def fitness_func(ga_instance, solution, solution_idx):
        predictions = forward_pass(solution, X, input_size, hidden_size, output_size)
        predictions = np.round(predictions).flatten()
        accuracy = np.mean(predictions == Y)
        return accuracy

    return fitness_func

def on_generation(ga_instance):
    if ga_instance.generations_completed % 10 == 0:
        best_fitness = np.max(ga_instance.last_generation_fitness)
        print(f"Generation {ga_instance.generations_completed}: Best Fitness = {best_fitness:.4f}")
