import random
import os
import numpy as np
import matplotlib.pyplot as plt
from Neuron import Neuron

class Network(object):

    def __init__(self, layers_topology, activacion_func_topology, momentum=0.2, learning_rate=0.1, bias=1, epoches=1000, error_measure_frequency=10):
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.bias = bias
        self.epoches = epoches
        self.error_measure_frequency = error_measure_frequency
        self.error = 0
        self.error_sum = 0
        self.error_max = 0
        self.error_X = []
        self.error_Y = []
        self.correct_Y = []
        self.predicted = []
        self.expected = []
        self.layers = [[Neuron(layers_topology[i-1], neuron_number+1, activacion_func_topology[i], self.momentum, self.learning_rate, self.bias) for neuron_number in range(layers_topology[i])] for i in range(len(layers_topology))]
    
    def predict(self, input):
        for neuron, x in zip(self.layers[0], input):
            neuron.output_value = x
        for layer, previous_layer in zip(self.layers[1:], self.layers[0:]):
            for neuron in layer:
                neuron.predict(previous_layer)

    def mean_squared_error(self, expected_output):
        for neuron, output in zip(self.layers[-1], expected_output):
            delta = output - neuron.output_value
            self.error += (delta*delta)/2

    def back_propagation(self, expected_output):
        self.mean_squared_error(expected_output)
        for neuron, output in zip(self.layers[-1], expected_output):
            neuron.output_layer_factor(output)
        for layer, next_layer in zip(reversed(self.layers[1:-1]), reversed(self.layers[2:])):
            for neuron in layer:
                neuron.hidden_layer_factor(next_layer)
        for layer, previous_layer in zip(reversed(self.layers[1:]), reversed(self.layers[0:-1])):
            for neuron in layer:
                neuron.update_weights(previous_layer)

    def train(self, learning_data, name, path, extend_record=False):
        name =  str(name) + '_train_'
        if(extend_record):
            error_save = open(os.path.join(path, str(name) + "error_tracking.txt"),"w+")
            data_control = open(os.path.join(path, str(name) + "date_predicting_control.txt"),"w+")
        for i in range(self.epoches):
            self.error = 0
            np.random.shuffle(learning_data)
            X = []
            Y = []
            self.predicted.clear()
            self.expected.clear()
            for row in learning_data:
                X.append(row[0:len(self.layers[0])])
                if row[-1] == 1:
                    Y.append([1,0,0])
                elif row[-1] == 2:
                    Y.append([0,1,0])
                elif row[-1] == 3:
                    Y.append([0,0,1])
            for data_in, data_out in zip(X, Y):
                self.predict(data_in)
                self.back_propagation(data_out)
                self.check_correct(data_out)
                if(extend_record):
                    if i % self.error_measure_frequency == 0 or i == 1 or i == self.epoches - 1:
                        data_control.write( "Epoka: " + str(i) + "\n" )
                        data_control.write( " Training input:  " + str(data_in) + "\n")
                        data_control.write( " Expected output: " + str(data_out) + "\n")
                        data_control.write( " Received output: ")
                        for neuron in self.layers[-1]:
                            data_control.write( str(neuron.output_value) + " " )
                        data_control.write("\n")
                        data_control.write( " Approximated output: ")
                        for neuron in self.layers[-1]:
                            data_control.write( str(round(neuron.output_value,3)) + " " )
                        data_control.write("\n")
            if(extend_record):
                if i % self.error_measure_frequency == 0 or i == 1 or i == self.epoches - 1:
                    error_save.write( str(self.error/len(self.layers[-1])) + '\n')
            self.error_Y.append(self.error/len(self.layers[-1]))
            self.error_X.append(i)
            self.sum_correct()
        if(extend_record):
            error_save.close()
            data_control.close()
        self.print_error_plot(0, path, name)
        self.print_correct_plot(0, path, name)

    def print_error_plot(self, save_0_print_1_none_2, path, name):
        plt.plot(self.error_X, self.error_Y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Error function')
        if save_0_print_1_none_2==0:
            plt.savefig(os.path.join(path, str(name) + 'error.png'))
            plt.close()
        elif save_0_print_1_none_2==1:
            plt.show()
            plt.close()

    def print_correct_plot(self, save_0_print_1_none_2, path, name):
        plt.plot(self.error_X, self.correct_Y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Percentage objects classyfication correctness')
        if save_0_print_1_none_2==0:
            plt.savefig(os.path.join(path, str(name) + 'correct.png'))
            plt.close()
        elif save_0_print_1_none_2==1:
            plt.show()
            plt.close()

    def check_correct(self, expected_output):
        y = 0
        j = 0
        for neuron, i in zip(self.layers[-1], range(len(self.layers[-1]))):
            if neuron.output_value > y:
                y = neuron.output_value
                j = i
        self.predicted.append(j)
        for i, num in zip(expected_output, range(3)):
            if i == 1:
                self.expected.append(num)

    def sum_correct(self):
        sum = 0
        for y, d in zip(self.predicted, self.expected):
            if y == d:
                sum += 1
        self.correct_Y.append(sum * 100 / len(self.predicted))

    def test(self, learning_data, name, path, extend_record=False):
        name =  str(name) + '_test_'
        if(extend_record):
            error_save = open(os.path.join(path, str(name) + "error_tracking.txt"),"w+")
            data_control = open(os.path.join(path, str(name) + "date_predicting_control.txt"),"w+")
        X = []
        Y = []
        for row in learning_data:
            X.append(row[0:len(self.layers[0])])
            if row[-1] == 1:
                Y.append([1,0,0])
            elif row[-1] == 2:
                Y.append([0,1,0])
            elif row[-1] == 3:
                Y.append([0,0,1])
        self.error_sum = 0
        self.error_max = 0
        self.error_X.clear()
        self.error_Y.clear()
        self.predicted.clear()
        self.expected.clear()
        self.correct_Y.clear()
        for data_in, data_out, i in zip(X, Y, range(len(X))):
            self.error = 0
            self.predict(data_in)
            self.check_correct(data_out)
            self.mean_squared_error(data_out)
            if(extend_record):
                self.error_sum += self.error
                if self.error_max < self.error:
                    self.error_max = self.error
                data_control.write( "Epoka: " + str(i) + "\n" )
                data_control.write( " Training input:  " + str(data_in) + "\n")
                data_control.write( " Expected output: " + str(data_out) + "\n")
                data_control.write( " Received output: ")
                for neuron in self.layers[-1]:
                    data_control.write( str(neuron.output_value) + " " )
                data_control.write("\n")
                data_control.write( " Approximated output: ")
                for neuron in self.layers[-1]:
                    data_control.write( str(round(neuron.output_value,3)) + " " )
                data_control.write("\n")
                error_save.write( str(self.error/len(self.layers[-1])) + '\n')
            self.error_Y.append(self.error/len(self.layers[-1]))
            self.error_X.append(i+1)
            self.sum_correct()
        if(extend_record):
            error_save.close()
            data_control.close()
        self.print_error_plot(0, path, name)
        if(extend_record):
            self.print_correct_plot(0, path, name)
        self.resoult(path, name)

    def resoult(self, path, name):
        tabela = open(os.path.join(path, str(name) + "resoult_in_table.txt"),"w+")

        tab = []
        tab.append([0,0,0])
        tab.append([0,0,0])
        tab.append([0,0,0])

        for i, j in zip(self.expected, self.predicted):
            tab[i][j] += 1

        tabela.write('   | A | B | C \n')
        tabela.write(' A | ' + str(tab[0][0]) + ' | ' + str(tab[0][1])  + ' | ' + str(tab[0][2])  + ' \n')
        tabela.write(' B | ' + str(tab[1][0]) + ' | ' + str(tab[1][1])  + ' | ' + str(tab[1][2])  + ' \n')
        tabela.write(' C | ' + str(tab[2][0]) + ' | ' + str(tab[2][1])  + ' | ' + str(tab[2][2])  + ' \n')

        tabela.close()