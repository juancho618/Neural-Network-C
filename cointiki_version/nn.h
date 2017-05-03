
#include <stdio.h>

#ifndef NN_RANDOM
/* We use the following for uniform random numbers between 0 and 1.
 * If you have a better function, redefine this macro. */
#define NN_RANDOM() (((double)rand())/RAND_MAX)
#endif

typedef double (*nn_actfun)(double a);

typedef struct nn {

    /* How many inputs, outputs, and hidden neurons. */
    int inputs, hidden_layers, hidden, outputs;

    /* Which activation function to use for hidden neurons. Default: nn_act_sigmoid */
    nn_actfun activation_hidden;

    /* Which activation function to use for output. Default: nn_act_sigmoid */
    nn_actfun activation_output;

    /* Total number of weights, and size of weights buffer. */
    int total_weights;

    /* Total number of neurons + inputs and size of output buffer. */
    int total_neurons;

    /* All weights (total_weights long). */
    double *weight;

    /* Stores input array and output of each neuron (total_neurons long). */
    double *output;

    /* Stores delta of each hidden and output neuron (total_neurons - inputs long). */
    double *delta;

  } nn;

/* Creates and returns a new ann. */
nn *nn_init(int inputs, int hidden_layers, int hidden, int outputs);

/* Sets weights randomly. Called by init. */
void nn_randomize(nn *ann);

/* Runs the feedforward algorithm to calculate the ann's output. */
double const *nn_run(nn const *ann, double const *inputs);


/* Does a single backprop update. */
void nn_train(nn const *ann, double const *inputs, double const *desired_outputs, double learning_rate);



double nn_act_sigmoid(double a);
double nn_act_threshold(double a);
double nn_act_linear(double a);
