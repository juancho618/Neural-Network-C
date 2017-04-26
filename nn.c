#include "nn.h"

double nn_act_sigmoid(double a) {
    if (a < -45.0) return 0; // ?
    if (a > 45.0) return 1;
    return 1.0 / (1 + exp(-a));
}

nn *nn_init(int inputs, int hidden_layers, int hidden, int outputs) {

    /* parameters valitation */
    if (hidden_layers < 0) return 0; // no negative numbers
    if (inputs < 1) return 0; // at least 1 input
    if (outputs < 1) return 0; // at least 1 output_weights
    if (hidden_layers > 0 && hidden < 1) return 0; // if there is a hidden layer it should be at lest 1 node

    const int hidden_weights = hidden_layers ? (inputs+1) * hidden + (hidden_layers-1) * (hidden+1) * hidden : 0; // connetion to the hidden layer + nodes in the hidden
    const int output_weights = (hidden_layers ? (hidden+1) : (inputs+1)) * outputs; // number of outputs connection from hidden layer + output nodes
    const int total_weights = (hidden_weights + output_weights);

    const int total_neurons = (inputs + hidden * hidden_layers + outputs);

    /* Allocate extra size for weights, outputs, and deltas. */
    const int size = sizeof(nn) + sizeof(double) * (total_weights + total_neurons + (total_neurons - inputs));
    nn *ret = malloc(size);
    if (!ret) return 0;

    ret->inputs = inputs;
    ret->hidden_layers = hidden_layers;
    ret->hidden = hidden;
    ret->outputs = outputs;

    ret->total_weights = total_weights;
    ret->total_neurons = total_neurons;

    /* Set pointers. */
    ret->weight = (double*)((char*)ret + sizeof(nn));
    ret->output = ret->weight + ret->total_weights;
    ret->delta = ret->output + ret->total_neurons;

    nn_randomize(ret);

    ret->activation_hidden = nn_act_sigmoid;
    ret->activation_output = nn_act_sigmoid;

    return ret;



}

void nn_randomize(nn *ann) {
    int i;
    for (i = 0; i < ann->total_weights; ++i) {
        double r = NN_RANDOM();
        /* Sets weights from -0.5 to 0.5. */
        ann->weight[i] = r - 0.5;
    }
}
