/*
 * GENANN - Minimal C Artificial Neural Network
 *
 * Copyright (c) 2015-2018 Lewis Van Winkle
 *
 * http://CodePlea.com
 *
 * This software is provided 'as-is', without any express or implied
 * warranty. In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgement in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 *
 */

#include "genann.h"
#include <ostream>

#include "genann_cu.h"
#ifdef FAST
#define NUM_THREADS omp_get_max_threads()
#else
#define NUM_THREADS 1
#endif

#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>

#include <omp.h>

#ifndef genann_act
#define genann_act_hidden genann_act_hidden_indirect
#define genann_act_output genann_act_output_indirect
#else
#define genann_act_hidden genann_act
#define genann_act_output genann_act
#endif

double genann_act_hidden_indirect(const struct genann *ann, double a) {
    return ann->activation_hidden(ann, a);
}

double genann_act_output_indirect(const struct genann *ann, double a) {
    return ann->activation_output(ann, a);
}

double interval;
double lookup[LOOKUP_SIZE];


double genann_act_sigmoid(double a) {
    if (a < -45.0) return 0;
    if (a > 45.0) return 1;
    return 1.0 / (1 + exp(-a));
}

void genann_init_sigmoid_lookup() {
        const double f = (sigmoid_dom_max - sigmoid_dom_min) / LOOKUP_SIZE;
        int i;

        interval = LOOKUP_SIZE / (sigmoid_dom_max - sigmoid_dom_min);
        for (i = 0; i < LOOKUP_SIZE; ++i) {
            lookup[i] = genann_act_sigmoid(sigmoid_dom_min + f * i);
        }
}

double genann_act_sigmoid_cached(const genann *ann unused, double a) {
    assert(!isnan(a));

    if (a < sigmoid_dom_min) return lookup[0];
    if (a >= sigmoid_dom_max) return lookup[LOOKUP_SIZE - 1];

    size_t j = (size_t)((a-sigmoid_dom_min)*interval+0.5);

    /* Because floating point... */
    if (unlikely(j >= LOOKUP_SIZE)) return lookup[LOOKUP_SIZE - 1];

    return lookup[j];
}

double genann_act_linear(const struct genann *ann unused, double a) {
    return a;
}

double genann_act_threshold(const struct genann *ann unused, double a) {
    return a > 0;
}

genann *genann_init(int inputs, int hidden_layers, int hidden, int outputs) {
    printf("Num threads: %d\n", NUM_THREADS);
    if (hidden_layers < 0) return 0;
    if (inputs < 1) return 0;
    if (outputs < 1) return 0;
    if (hidden_layers > 0 && hidden < 1) return 0;


    const int hidden_weights = hidden_layers ? (inputs+1) * hidden + (hidden_layers-1) * (hidden+1) * hidden : 0;
    const int output_weights = (hidden_layers ? (hidden+1) : (inputs+1)) * outputs;
    const int total_weights = (hidden_weights + output_weights);

    const int total_neurons = (inputs + hidden * hidden_layers + outputs);

    /* Allocate extra size for weights, outputs, and deltas. */
    const int size = sizeof(genann) + sizeof(double) * (total_weights + total_neurons + (total_neurons - inputs));
    genann *ret = (genann*)malloc(size);
    if (!ret) return 0;

    ret->inputs = inputs;
    ret->hidden_layers = hidden_layers;
    ret->hidden = hidden;
    ret->outputs = outputs;

    ret->total_weights = total_weights;
    ret->total_neurons = total_neurons;

    /* Set pointers. */
    ret->weight = (double*)((char*)ret + sizeof(genann));
    ret->output = ret->weight + ret->total_weights;
    ret->delta = ret->output + ret->total_neurons;

    genann_randomize(ret);

    ret->activation_hidden = genann_act_sigmoid_cached;
    ret->activation_output = genann_act_sigmoid_cached;

#ifdef FAST
    interval = genann_init_sigmoid_lookup_cuda(lookup);
#else
    genann_init_sigmoid_lookup();
#endif

    return ret;
}


genann *genann_read(FILE *in) {
    int inputs, hidden_layers, hidden, outputs;
    int rc;

    errno = 0;
    rc = fscanf(in, "%d %d %d %d", &inputs, &hidden_layers, &hidden, &outputs);
    if (rc < 4 || errno != 0) {
        perror("fscanf");
        return NULL;
    }

    genann *ann = genann_init(inputs, hidden_layers, hidden, outputs);

    int i;
    for (i = 0; i < ann->total_weights; ++i) {
        errno = 0;
        rc = fscanf(in, " %le", ann->weight + i);
        if (rc < 1 || errno != 0) {
            perror("fscanf");
            genann_free(ann);

            return NULL;
        }
    }

    return ann;
}


genann *genann_copy(genann const *ann) {
    const int size = sizeof(genann) + sizeof(double) * (ann->total_weights + ann->total_neurons + (ann->total_neurons - ann->inputs));
    genann *ret = (genann*)malloc(size);
    if (!ret) return 0;

    memcpy(ret, ann, size);

    /* Set pointers. */
    ret->weight = (double*)((char*)ret + sizeof(genann));
    ret->output = ret->weight + ret->total_weights;
    ret->delta = ret->output + ret->total_neurons;

    return ret;
}


void genann_randomize(genann *ann) {
    int i;
    for (i = 0; i < ann->total_weights; ++i) {
        double r = GENANN_RANDOM();
        /* Sets weights from -0.5 to 0.5. */
        ann->weight[i] = r - 0.5;
    }
}


void genann_free(genann *ann) {
    /* The weight, output, and delta pointers go to the same buffer. */
    free(ann);
}


double const *genann_run(genann const *ann, double const *inputs) {
    double const *w = ann->weight;
    double *o = ann->output + ann->inputs;
    double const *i = ann->output;

    /* Copy the inputs to the scratch area, where we also store each neuron's
     * output, for consistency. This way the first layer isn't a special case. */
    memcpy(ann->output, inputs, sizeof(double) * ann->inputs);

    int h, j, k;

    if (!ann->hidden_layers) {
        double *ret = o;
        for (j = 0; j < ann->outputs; ++j) {
            double sum = *w++ * -1.0;
            for (k = 0; k < ann->inputs; ++k) {
                sum += *w++ * i[k];
            }
            *o++ = genann_act_output(ann, sum);
        }

        return ret;
    }

    /* Figure input layer */
    for (j = 0; j < ann->hidden; ++j) {
        double sum = *w++ * -1.0;
        for (k = 0; k < ann->inputs; ++k) {
            sum += *w++ * i[k];
        }
        *o++ = genann_act_hidden(ann, sum);
    }

    i += ann->inputs;

    /* Figure hidden layers, if any. */
    for (h = 1; h < ann->hidden_layers; ++h) {
        for (j = 0; j < ann->hidden; ++j) {
            double sum = *w++ * -1.0;
            for (k = 0; k < ann->hidden; ++k) {
                sum += *w++ * i[k];
            }
            *o++ = genann_act_hidden(ann, sum);
        }

        i += ann->hidden;
    }

    double const *ret = o;

    /* Figure output layer. */
    for (j = 0; j < ann->outputs; ++j) {
        double sum = *w++ * -1.0;
        for (k = 0; k < ann->hidden; ++k) {
            sum += *w++ * i[k];
        }
        *o++ = genann_act_output(ann, sum);
    }

    /* Sanity check that we used all weights and wrote all outputs. */
    assert(w - ann->weight == ann->total_weights);
    assert(o - ann->output == ann->total_neurons);

    return ret;
}


void set_hidden_deltas(const genann* ann)
{
    /* Set hidden layer deltas, start on last layer and work backwards. */
    /* Note that loop is skipped in the case of hidden_layers == 0. */
    for (int h = ann->hidden_layers - 1; h >= 0; --h) {

        /* Find first output and delta in this layer. */
        double const *o = ann->output + ann->inputs + (h * ann->hidden);
        double *d = ann->delta + (h * ann->hidden);

        /* Find first delta in following layer (which may be hidden or output). */
        double const * const dd = ann->delta + ((h+1) * ann->hidden);

        /* Find first weight in following layer (which may be hidden or output). */
        double const * const ww = ann->weight + ((ann->inputs+1) * ann->hidden) + ((ann->hidden+1) * ann->hidden * (h));

#pragma omp parallel firstprivate(d, o) num_threads(NUM_THREADS)
        {
            // Offset pointers so that each thread uses a different index
            const int num_threads = omp_get_num_threads();
            const int it_per_thread = (num_threads / ann->hidden);
            d += omp_get_thread_num() * it_per_thread;
            o += omp_get_thread_num() * it_per_thread;

            #pragma omp for schedule(static) nowait
            for (int j = 0; j < ann->hidden; ++j) {

                double delta = 0;

                for (int k = 0; k < (h == ann->hidden_layers-1 ? ann->outputs : ann->hidden); ++k)
                {
                    const double forward_delta = dd[k];
                    const int windex = k * (ann->hidden + 1) + (j + 1);
                    const double forward_weight = ww[windex];
                    delta += forward_delta * forward_weight;
                }

                *d = *o * (1.0-*o) * delta;
                // Use a num_threads stride
                d++;
                o++;
            }
        }
    }
}

void train_output(const genann* ann, double learning_rate)
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
    for (int j = 0; j < ann->outputs; ++j) {
        *w++ += *d * learning_rate * -1.0;
        for (int k = 1; k < (ann->hidden_layers ? ann->hidden : ann->inputs) + 1; ++k) {
            *w++ += *d * learning_rate * i[k-1];
        }

        ++d;
    }

    assert(w - ann->weight == ann->total_weights);
}

void train_hidden(const genann* ann, double learning_rate)
{
    /* Train the hidden layers. */
    for (int h = ann->hidden_layers - 1; h >= 0; --h)
    {
        /* Find first delta in this layer. */
        double const *d = ann->delta + (h * ann->hidden);

        /* Find first input to this layer. */
        double const *i = ann->output + (h
                ? (ann->inputs + ann->hidden * (h-1))
                : 0);

        /* Find first weight to this layer. */
        double *w = ann->weight + (h
                ? ((ann->inputs+1) * ann->hidden + (ann->hidden+1) * (ann->hidden) * (h-1))
                : 0);

        #pragma omp parallel firstprivate(d, w) num_threads(NUM_THREADS)
        {
            auto it_per_thread = ann->hidden / omp_get_num_threads();
            auto nested_iterations = ((h == 0 ? ann->inputs : ann->hidden) + 1);
            w += omp_get_thread_num() * it_per_thread * nested_iterations;
            d += omp_get_thread_num() * it_per_thread;


            #pragma omp for schedule(static) nowait
            for (int j = 0; j < ann->hidden; ++j) {
                *w++ += *d * learning_rate * -1.0;
                for (int k = 1; k < (h == 0 ? ann->inputs : ann->hidden) + 1; ++k) {
                    *w++ += *d * learning_rate * i[k-1];
                }
                ++d;
            }
        }
    }
}

void genann_train(genann const *ann, double const *inputs, double const *desired_outputs, double learning_rate) {
    /* To begin with, we must run the network forward. */
    genann_run(ann, inputs);

    int j;

    /* First set the output layer deltas. */
    {
        double const *o = ann->output + ann->inputs + ann->hidden * ann->hidden_layers; /* First output. */
        double *d = ann->delta + ann->hidden * ann->hidden_layers; /* First delta. */
        double const *t = desired_outputs; /* First desired output. */


        /* Set output layer deltas. */
        if (genann_act_output == genann_act_linear ||
                ann->activation_output == genann_act_linear) {
            for (j = 0; j < ann->outputs; ++j) {
                *d++ = *t++ - *o++;
            }
        } else {
            for (j = 0; j < ann->outputs; ++j) {
                *d++ = (*t - *o) * *o * (1.0 - *o);
                ++o; ++t;
            }
        }
    }


    set_hidden_deltas(ann);

    train_output(ann, learning_rate);

    train_hidden(ann, learning_rate);
}


void genann_write(genann const *ann, FILE *out) {
    fprintf(out, "%d %d %d %d", ann->inputs, ann->hidden_layers, ann->hidden, ann->outputs);

    int i;
    for (i = 0; i < ann->total_weights; ++i) {
        fprintf(out, " %.20e", ann->weight[i]);
    }
}
