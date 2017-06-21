
from numpy import exp, array, random, dot
import numpy as np



class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1


class NeuralNetwork():
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural network
            output_from_layer_1, output_from_layer_2 = self.think(training_set_inputs)

            # Calculate the error for layer 2 (The difference between the desired output
            # and the predicted output).
            layer2_error = training_set_outputs - output_from_layer_2
            layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2)

            # Calculate the error for layer 1 (By looking at the weights in layer 1,
            # we can determine by how much layer 1 contributed to the error in layer 2).
            layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
            layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)

            # Calculate how much to adjust the weights by
            layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)

            # Adjust the weights.
            self.layer1.synaptic_weights += layer1_adjustment
            self.layer2.synaptic_weights += layer2_adjustment

    # Sieć neuronowa mysli
    def think(self, inputs):
        output_from_layer1 = self.__sigmoid(dot(inputs, self.layer1.synaptic_weights))
        output_from_layer2 = self.__sigmoid(dot(output_from_layer1, self.layer2.synaptic_weights))
        return output_from_layer1, output_from_layer2

    # Sieć neuronów wyświetla wagi
    def print_weights(self):
        print ("   Layer 1 (4 neurony, każdy 3 wejscia): ",self.layer1.synaptic_weights)
        print ("   Layer 2 (1 neuron, każdy 4 wejscia):",self.layer2.synaptic_weights)





if __name__ == "__main__":

    #Seed random
    random.seed(1)

    # Tworzymy layer 1, 4 neurony z 3 wejściami
    layer1 = NeuronLayer(4, 3)

    # Tworzymy layer 2, 1 neuron z 4 wejściami
    layer2 = NeuronLayer(1, 4)

    # Połączenie warstw by utworzyć sieć neuronową
    neural_network = NeuralNetwork(layer1, layer2)

    print ("Stage 1) Randomowe wagi: ", neural_network.print_weights())





    # Zestaw treningowy, 7 przykładów, każdy 3 wejścia
    # i 1 wyjście

    #[umiejętności,   wynik testu wewnetrznego    ,doswiadczenie]
    training_set_inputs1 = np.array(([9, 1, 2], [2, 8, 4], [10, 6, 1],[9, 8, 8], [4, 2, 7], [7, 9, 2],[3 ,4, 3]), dtype=float)
    # wynik, czy warto przyjać pracownika [%]

    print("Stage 2) zestawu treningowy na wejściu: ", training_set_inputs1)


    training_set_outputs = np.array(([48], [53], [84],[94], [41], [80], [28]), dtype=float)
    training_set_inputs = training_set_inputs1 / np.amax(training_set_inputs1, axis=0)
    training_set_outputs = training_set_outputs / 100  # Max to 100%


    # Trenujemy sieć używając nasz zestaw treningowy
    # 60 000 razy i dokonujemy korekt
    neural_network.train(training_set_inputs, training_set_outputs, 60000)

    print ("Stage 2.1) Nowe wagi po treningu: ", neural_network.print_weights())





    # Test nowej sieci


    x = input("Umiejetnosci[1-10]: ")
    y = input("Wynik testu[1-10]: ")
    z = input("Doswiadczenie[1-10]: ")
    #nowy test [umiejetnosci,wynik testu wewnetrznego, doswiadczenie]
    array_result = np.array(( [x,y,z] ), dtype=float)

    result = array_result/np.amax(training_set_inputs1, axis=0)
    hidden_state, output = neural_network.think(result)
    print ("Stage 3) Rozpatrzenie nowej sytuacji: Szanse na przyjęcie[%]:  -> ", array_result , "=>" , result, "-=>" ,output*100,"% szansy")

