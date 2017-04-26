#include "nn.h"

nn *nn_init(int inputs, int hidden_layers, int hidden, int outputs) {

    /* parameters valitation */
    if (hidden_layers < 0) return 0; // no negative numbers
    if (inputs < 1) return 0; // at least 1 input
    if (outputs < 1) return 0; // at least 1 output_weights
    if (hidden_layers > 0 && hidden < 1) return 0; // if there is a hidden layer it should be at lest 1 node

    const int hidden_weights = hidden_layers ? (inputs+1) * hidden + (hidden_layers-1) * (hidden+1) * hidden : 0; // connetion to the hidden layer + nodes in the hidden
    const int output_weights = (hidden_layers ? (hidden+1) : (inputs+1)) * outputs; // number of outputs connection from hidden layer + output nodes
    const int total_weights = (hidden_weights + output_weights

    const int total_neurons = (inputs + hidden * hidden_layers + outputs);



}
