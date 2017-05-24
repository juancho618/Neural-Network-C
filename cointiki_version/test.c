#include "contiki.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include <assert.h>
#include <math.h>
#include "lib/random.h"
//#include "nn.h"
//#include "nn.c"
//#include "cfs/cfs.h"
//#include "cfs/cfs-coffee.h"

typedef float (*nn_actfun)(float a);
float nn_act_threshold(float a);
float nn_act_linear(float a);


float nn_act_sigmoid(float a) {
    return (1.0 /(1 + expf((float)a)); 
}


float nn_act_linear(float a) {
    return a;
}

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

  // Assign random weights
  void nn_randomize(struct nn *ann) {
      void random_init(unsigned short seed);
      int i;
      for (i = 0; i < (*ann).total_weights; ++i) {
          float r = (((float)random_rand())/RANDOM_RAND_MAX); 
          (*ann).weight[i] = r-0.5;
      }
  }

  int const *nn_run(struct nn const *ann, int const *inputs) {
    float const *w = (*ann).weight;
    float *o = (*ann).output + (*ann).inputs;
    float const *i = (*ann).output;
    memcpy((*ann).output, inputs, sizeof(int) * (*ann).inputs);

    int h, j, k;

   const nn_actfun act = (*ann).activation_hidden;
   const nn_actfun acto = (*ann).activation_output;


   // Figure hidden layers, if any.
   for (h = 0; h < (*ann).hidden_layers; ++h) {
       for (j = 0; j < (*ann).hidden; ++j) {

           float sum = 0;
           for (k = 0; k < (h == 0 ? (*ann).inputs : (*ann).hidden) + 1; ++k) {
             if (k == 0) {
                 sum += *w++ * -1.0; // threshold parameter.
             } else {
                 sum += *w++ * i[k-1];
             }
             *o++ = act(sum);
           }
       }

       i += (h == 0 ? (*ann).inputs : (*ann).hidden);
   }
    double const *ret = o;

    /* Figure output layer. */
    for (j = 0; j < (*ann).outputs; ++j) {
        float sum = 0;
        for (k = 0; k < ((*ann).hidden_layers ? (*ann).hidden : (*ann).inputs) + 1; ++k) {
            if (k == 0) {
                sum += *w++ * -1.0;
            } else {
                sum += *w++ * i[k-1];
            }
        }
        *o++ = acto(sum);
    }
    return ret;
  }

  // Neuronal Network Initialier
  nn *nn_init(int inputs, int hidden_layers, int hidden, int outputs) {


      /* parameters valitation */
      if (hidden_layers < 0) return 0; // no negative numbers
      if (inputs < 1) return 0; // at least 1 input
      if (outputs < 1) return 0; // at least 1 output_weights
      if (hidden_layers > 0 && hidden < 1) return 0; // if there is a hidden layer it should be at lest 1 node

      const int hidden_weights = hidden_layers ? (inputs+1) * hidden + (hidden_layers-1) * (hidden+1) * hidden : 0; // connetion to the hidden layer + nodes in the hidden +1 due to the threshold
      const int output_weights = (hidden_layers ? (hidden+1) : (inputs+1)) * outputs; // number of outputs connection from hidden layer + output nodes +1 due to the threshold
      const int total_weights = (hidden_weights + output_weights);

      const int total_neurons = (inputs + hidden * hidden_layers + outputs);

      
      struct nn *ret;

      (*ret).inputs = inputs;
      (*ret).hidden_layers = hidden_layers;
      (*ret).hidden = hidden;
      (*ret).outputs = outputs;

      (*ret).total_weights = total_weights;
      (*ret).total_neurons = total_neurons;
      /* Set pointers. */
      (*ret).weight = (float*)(sizeof(nn));
      (*ret).output = (*ret).weight + (*ret).total_weights;
      (*ret).delta = (*ret).output + (*ret).total_neurons;
      nn *neu = &ret; // doing the pointer to the neuronal network
      nn_randomize(neu); // send pointer as a prameter

      (*ret).activation_hidden = nn_act_sigmoid;
      (*ret).activation_output = nn_act_sigmoid;

      return ret;
  }

  void nn_train(nn const *ann, int const *inputs, int const *desired_outputs, float learning_rate) {
    nn_run(ann, inputs);
    //TODO: continue adding the values for the Training function

    // deltas calulation
    int h, j, k;

    {
        float const *o = (*ann).output + (*ann).inputs + (*ann).hidden * (*ann).hidden_layers; // First output.
        float *d = (*ann).delta + (*ann).hidden * (*ann).hidden_layers; // First delta.
        float const *t = desired_outputs; // First desired output.



        // Set output layer deltas.
        if ((*ann).activation_output == nn_act_linear) {
            for (j = 0; j < (*ann).outputs; ++j) {
                *d++ = *t++ - *o++;
            }
        } else {
            for (j = 0; j < (*ann).outputs; ++j) {
                /**d++ = (*t - *o) * *o * (1.0 - *o); //TODO: output error probably due to memory or a double
                ++o; ++t;*/
                ++o;
                ++t;
            }
        }
      }

    // Set hidden layer deltas, start on last layer and work backwards.
    // Note that loop is skipped in the case of hidden_layers == 0.
    for (h = (*ann).hidden_layers - 1; h >= 0; --h) {

        // Find first output and delta in this layer.
        float const *o = (*ann).output + (*ann).inputs + (h * (*ann).hidden);
        float *d = (*ann).delta + (h * (*ann).hidden);

        // Find first delta in following layer (which may be hidden or output).
        float const * const dd = (*ann).delta + ((h+1) * (*ann).hidden);

        // Find first weight in following layer (which may be hidden or output).
        float const * const ww = (*ann).weight + (((*ann).inputs+1) * (*ann).hidden) + (((*ann).hidden+1) * (*ann).hidden * (h));

        for (j = 0; j < (*ann).hidden; ++j) {

            float delta = 0;
            
            for (k = 0; k < (h == (*ann).hidden_layers-1 ? (*ann).outputs : (*ann).hidden); ++k) {
                const float forward_delta = dd[k];       TODO: another overflow problem!
                const int windex = k * ((*ann).hidden + 1) + (j + 1);
                const float forward_weight = ww[windex];
                delta += forward_delta * forward_weight;*
            }
            *d = *o * (1.0-*o) * delta;
            ++d; ++o;
            
        }
    }

     /* Train the outputs. */
    {
        /* Find first output delta. */
        double const *d = ann->delta + ann->hidden * ann->hidden_layers; /* First output delta. */

        /* Find first weight to first output delta. */
        double *w = ann->weight + (ann->hidden_layers
                ? ((ann->inputs+1) * ann->hidden + (ann->hidden+1) * ann->hidden * (ann->hidden_layers-1))
                : (0));

        /* Find first output in previous layer. */
        double const * const i = ann->output + (ann->hidden_layers
                ? (ann->inputs + (ann->hidden) * (ann->hidden_layers-1))
                : 0);

        /* Set output layer weights. */
        for (j = 0; j < ann->outputs; ++j) {
            for (k = 0; k < (ann->hidden_layers ? ann->hidden : ann->inputs) + 1; ++k) {
                if (k == 0) {
                    *w++ += *d * learning_rate * -1.0;
                } else {
                    *w++ += *d * learning_rate * i[k-1];
                }
            }

            ++d;
        }

        assert(w - ann->weight == ann->total_weights);
    }


    /* Train the hidden layers. */
    for (h = ann->hidden_layers - 1; h >= 0; --h) {

        /* Find first delta in this layer. */
        float const *d = ann->delta + (h * ann->hidden);

        /* Find first input to this layer. */
        float const *i = ann->output + (h
                ? (ann->inputs + ann->hidden * (h-1))
                : 0);

        /* Find first weight to this layer. */
        float *w = ann->weight + (h
                ? ((ann->inputs+1) * ann->hidden + (ann->hidden+1) * (ann->hidden) * (h-1))
                : 0);


        for (j = 0; j < ann->hidden; ++j) {
            for (k = 0; k < (h == 0 ? ann->inputs : ann->hidden) + 1; ++k) {
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

PROCESS(neuronal, "testing nn");
AUTOSTART_PROCESSES(&neuronal);

PROCESS_THREAD(neuronal, ev, data)
{
  PROCESS_BEGIN();
    printf("Neuronal Network Implementation\n");
    printf("Please work!.\n");


    const int input[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}}; //integer definition
    const int output[4] = {0, 1, 1, 0}; // integer definition
    int i;
    nn *ann = nn_init(2, 1, 2, 1);

    // Training the Neuronal Network
    for (i = 0; i < 200; ++i) {
      nn_train(ann, input[0], output + 0, 3);
      nn_train(ann, input[1], output + 1, 3);
      nn_train(ann, input[2], output + 2, 3);
      nn_train(ann, input[3], output + 3, 3);
    }

    printf("Output for [%1.i, %1.i] is %1.i.\n", input[0][0], input[0][1], *nn_run(ann, input[0]));
    printf("Output for [%1.i, %1.i] is %1.i.\n", input[1][0], input[1][1], *nn_run(ann, input[1]));
    printf("Output for [%1.i, %1.i] is %1.i.\n", input[2][0], input[2][1], *nn_run(ann, input[2]));
    printf("Output for [%1.i, %1.i] is %1.i.\n", input[3][0], input[3][1], *nn_run(ann, input[3]));
   

    PROCESS_END();
    }
