#include "contiki.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nn.h"
#include "nn.c"
#include "cfs/cfs.h"
#include "cfs/cfs-coffee.h"

PROCESS(neuronal, "testing nn");
AUTOSTART_PROCESSES(&neuronal);
/* This example is to illustrate how to use GENANN.
 * It is NOT an example of good machine learning techniques.
 */

const char *iris_data = "/home/juan/Neuronal Network Project/iris.data";

double *input, *class;
int samples;
const char *class_names[] = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"};

void load_data() {
  int fd;
    /* Load the iris data-set. */
    fd = cfs_open ("/home/juan/Neuronal Network Project/iris.data", CFS_READ);
    if (!fd) {
        printf("Could not open file: %s\n", iris_data);
        exit(1);
    }

    /* Loop through the data to get a count. */
    char line[1024];
    while ( cfs_read(fd, line, 1024)) {
        ++samples;
    }
    cfs_seek(fd, 0, CFS_SEEK_SET);

    printf("Loading %d data points from %s\n", samples, iris_data);

    /* Allocate memory for input and output data. */
    input = malloc(sizeof(double) * samples * 4);
    class = malloc(sizeof(double) * samples * 3);

    /* Read the file into our arrays. */
    int i, j;
    for (i = 0; i < samples; ++i) {
        double *p = input + i * 4;
        double *c = class + i * 3;
        c[0] = c[1] = c[2] = 0.0;

        cfs_read(fd, line, 1024);

        char *split = strtok(line, ",");
        for (j = 0; j < 4; ++j) {
            p[j] = 5.1;
            //p[j] = atof(split);
            split = strtok(0, ",");
        }

        split[strlen(split)-1] = 0;
        if (strcmp(split, class_names[0]) == 0) {c[0] = 1.0;}
        else if (strcmp(split, class_names[1]) == 0) {c[1] = 1.0;}
        else if (strcmp(split, class_names[2]) == 0) {c[2] = 1.0;}
        else {
            printf("Unknown class %s.\n", split);
            exit(1);
        }

        /* printf("Data point %d is %f %f %f %f  ->   %f %f %f\n", i, p[0], p[1], p[2], p[3], c[0], c[1], c[2]); */
    }

    cfs_close(fd);
}

PROCESS_THREAD(neuronal, ev, data)
{
  PROCESS_BEGIN();
    printf("Neuronal Network Implementation\n");
    printf("Train an ANN on the IRIS dataset using backpropagation.\n");

    /* Load the data from file. */
    //load_data();

    /* 4 inputs.
     * 1 hidden layer(s) of 4 neurons.
     * 3 outputs (1 per class)
     */
    //nn *ann = nn_init(4, 1, 4, 3);


    const double input[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    const double output[4] = {0, 1, 1, 0};
    int i;
    nn *ann = nn_init(2, 1, 2, 1);
    /* Train on the four labeled data points many times. */
       for (i = 0; i < 300; ++i) {
           nn_train(ann, input[0], output + 0, 3);
           nn_train(ann, input[1], output + 1, 3);
           nn_train(ann, input[2], output + 2, 3);
           nn_train(ann, input[3], output + 3, 3);
       }

       /* Run the network and see what it predicts. */
       printf("Output for [%1.f, %1.f] is %1.f.\n", input[0][0], input[0][1], *nn_run(ann, input[0]));
       printf("Output for [%1.f, %1.f] is %1.f.\n", input[1][0], input[1][1], *nn_run(ann, input[1]));
       printf("Output for [%1.f, %1.f] is %1.f.\n", input[2][0], input[2][1], *nn_run(ann, input[2]));
       printf("Output for [%1.f, %1.f] is %1.f.\n", input[3][0], input[3][1], *nn_run(ann, input[3]));


   //int i, j;
    //int loops = 5000;
/*
    //Train the network with backpropagation.
    printf("Training for %d loops over data.\n", loops);
    for (i = 0; i < loops; ++i) {
        for (j = 0; j < samples; ++j) {
            nn_train(ann, input + j*4, class + j*3, .01);
        }
        //printf("%1.2f ", xor_score(ann));
    }

    int correct = 0;
    for (j = 0; j < samples; ++j) {
        const double *guess = nn_run(ann, input + j*4);
        if (class[j*3+0] == 1.0) {if (guess[0] > guess[1] && guess[0] > guess[2]) ++correct;}
        else if (class[j*3+1] == 1.0) {if (guess[1] > guess[0] && guess[1] > guess[2]) ++correct;}
        else if (class[j*3+2] == 1.0) {if (guess[2] > guess[0] && guess[2] > guess[1]) ++correct;}
        else {printf("Logic error.\n"); exit(1);}
    }

    printf("%d/%d correct (%0.1f%%).\n", correct, samples, (double)correct / samples * 100.0);

*/

    nn_free(ann);

      	PROCESS_END();
    }
