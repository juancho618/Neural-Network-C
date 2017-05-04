#include <assert.h>
#include "floattostring.c"
#include <math.h>
#include "lib/mmem.h" // library replacing memory allocation!

#define LOOKUP_SIZE 4096
//static struct mmem mmem; // initialize memory managments
double nn_act_sigmoid(double a) {

    if (a < -45.0) return 0; // ?
    if (a > 45.0) return 1;

    return 1.0 / (1 + -a);
    //return 1.0 / (1 + exp(-a));
}

double nn_act_threshold(double a) {
    return a > 0;
}


double nn_act_linear(double a) {
    return a;
}

// Neuronal Network Initialier
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
    //const int size = sizeof(nn) + sizeof(double) * (total_weights + total_neurons + (total_neurons - inputs));
    //nn *ret = mmem_alloc(&mmem,size);
    //if (!ret) return 0;
    struct nn *ret;

    (*ret).inputs = inputs;
    (*ret).hidden_layers = hidden_layers;
    (*ret).hidden = hidden;
    (*ret).outputs = outputs;

    (*ret).total_weights = total_weights;
    (*ret).total_neurons = total_neurons;

    /* Set pointers. */
    (*ret).weight = (double*)((char*)ret + sizeof(nn));
    (*ret).output = (*ret).weight + (*ret).total_weights;
    (*ret).delta = (*ret).output + (*ret).total_neurons;

    nn_randomize(ret);

    (*ret).activation_hidden = nn_act_sigmoid;
    (*ret).activation_output = nn_act_sigmoid;

    return ret;
}


// Run algortihm givin the defined architecture and the data input
double const *nn_run(nn const *ann, double const *inputs) {
    double const *w = (*ann).weight;
    double *o = (*ann).output + (*ann).inputs;
    double const *i = (*ann).output;

    /* Copy the inputs to the scratch area, where we also store each neuron's
     * output, for consistency. This way the first layer isn't a special case. */
    memcpy((*ann).output, inputs, sizeof(double) * (*ann).inputs);

    int h, j, k;

    const nn_actfun act = (*ann).activation_hidden;
    const nn_actfun acto = (*ann).activation_output;

    /* Figure hidden layers, if any. */
    for (h = 0; h < (*ann).hidden_layers; ++h) {
        for (j = 0; j < (*ann).hidden; ++j) {
            double sum = 0;
            for (k = 0; k < (h == 0 ? (*ann).inputs : (*ann).hidden) + 1; ++k) {
                if (k == 0) {
                    sum += *w++ * -1.0;
                } else {
                    sum += *w++ * i[k-1];
                }
            }
            *o++ = act(sum);
        }


        i += (h == 0 ? (*ann).inputs : (*ann).hidden);
    }

    double const *ret = o;

    /* Figure output layer. */
    for (j = 0; j < (*ann).outputs; ++j) {
        double sum = 0;
        for (k = 0; k < ((*ann).hidden_layers ? (*ann).hidden : (*ann).inputs) + 1; ++k) {
            if (k == 0) {
                sum += *w++ * -1.0;
            } else {
                sum += *w++ * i[k-1];
            }
        }
        *o++ = acto(sum);
    }

    /* Sanity check that we used all weights and wrote all outputs. */
    assert(w - (*ann).weight == (*ann).total_weights);
    assert(o - (*ann).output == (*ann).total_neurons);

    return ret;
}

// Training the algorithm
void nn_train(nn const *ann, double const *inputs, double const *desired_outputs, double learning_rate) {
    /* To begin with, we must run the network forward. */
      nn_run(ann, inputs);

    int h, j, k;

    /* First set the output layer deltas. */
    {
        double const *o = (*ann).output + (*ann).inputs + (*ann).hidden * (*ann).hidden_layers; /* First output. */
        double *d = (*ann).delta + (*ann).hidden * (*ann).hidden_layers; /* First delta. */
        double const *t = desired_outputs; /* First desired output. */


        /* Set output layer deltas. */
        if ((*ann).activation_output == nn_act_linear) {
            for (j = 0; j < (*ann).outputs; ++j) {
                *d++ = *t++ - *o++;
            }
        } else {
            for (j = 0; j < (*ann).outputs; ++j) {
                *d++ = (*t - *o) * *o * (1.0 - *o);
                ++o; ++t;
            }
        }
    }


    /* Set hidden layer deltas, start on last layer and work backwards. */
    /* Note that loop is skipped in the case of hidden_layers == 0. */
    for (h = (*ann).hidden_layers - 1; h >= 0; --h) {

        /* Find first output and delta in this layer. */
        double const *o = (*ann).output + (*ann).inputs + (h * (*ann).hidden);
        double *d = (*ann).delta + (h * (*ann).hidden);

        /* Find first delta in following layer (which may be hidden or output). */
        double const * const dd = (*ann).delta + ((h+1) * (*ann).hidden);

        /* Find first weight in following layer (which may be hidden or output). */
        double const * const ww = (*ann).weight + (((*ann).inputs+1) * (*ann).hidden) + (((*ann).hidden+1) * (*ann).hidden * (h));

        for (j = 0; j < (*ann).hidden; ++j) {

            double delta = 0;

            for (k = 0; k < (h == (*ann).hidden_layers-1 ? (*ann).outputs : (*ann).hidden); ++k) {
                const double forward_delta = dd[k];
                const int windex = k * ((*ann).hidden + 1) + (j + 1);
                const double forward_weight = ww[windex];
                delta += forward_delta * forward_weight;
            }

            *d = *o * (1.0-*o) * delta;
            ++d; ++o;
        }
    }


    /* Train the outputs. */
    {
        /* Find first output delta. */
        double const *d = (*ann).delta + (*ann).hidden * (*ann).hidden_layers; /* First output delta. */

        /* Find first weight to first output delta. */
        double *w = (*ann).weight + ((*ann).hidden_layers
                ? (((*ann).inputs+1) * (*ann).hidden + ((*ann).hidden+1) * (*ann).hidden * ((*ann).hidden_layers-1))
                : (0));

        /* Find first output in previous layer. */
        double const * const i = (*ann).output + ((*ann).hidden_layers
                ? ((*ann).inputs + ((*ann).hidden) * ((*ann).hidden_layers-1))
                : 0);

        /* Set output layer weights. */
        for (j = 0; j < (*ann).outputs; ++j) {
            for (k = 0; k < ((*ann).hidden_layers ? (*ann).hidden : (*ann).inputs) + 1; ++k) {
                if (k == 0) {
                    *w++ += *d * learning_rate * -1.0;
                } else {
                    *w++ += *d * learning_rate * i[k-1];
                }
            }

            ++d;
        }

        assert(w - (*ann).weight == (*ann).total_weights);
    }


    /* Train the hidden layers. */
    for (h = (*ann).hidden_layers - 1; h >= 0; --h) {

        /* Find first delta in this layer. */
        double const *d = (*ann).delta + (h * (*ann).hidden);

        /* Find first input to this layer. */
        double const *i = (*ann).output + (h
                ? ((*ann).inputs + (*ann).hidden * (h-1))
                : 0);

        /* Find first weight to this layer. */
        double *w = (*ann).weight + (h
                ? (((*ann).inputs+1) * (*ann).hidden + ((*ann).hidden+1) * ((*ann).hidden) * (h-1))
                : 0);


        for (j = 0; j < (*ann).hidden; ++j) {
            for (k = 0; k < (h == 0 ? (*ann).inputs : (*ann).hidden) + 1; ++k) {
                if (k == 0) {
                    *w++ += *d * learning_rate * -1.0;
                } else {
                    *w++ += *d * learning_rate * i[k-1];
                }
            }
            ++d;
        }

    }

}


void nn_free(nn *ann) {
    /* The weight, output, and delta pointers go to the same buffer. */
    //mmem_free(&mmem);
}

// Assign random weights
void nn_randomize(nn *ann) {
    int i;
    for (i = 0; i < (*ann).total_weights; ++i) {
        double r = NN_RANDOM();
        /* Sets weights from -0.5 to 0.5. */
        (*ann).weight[i] = r - 0.5;
    }
}
