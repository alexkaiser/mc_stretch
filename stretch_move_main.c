/*
Copyright (c) 2013, Alex Kaiser
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice,
 this list of conditions and the following disclaimer. Redistributions
 in binary form must reproduce the above copyright notice, this list
 of conditions and the following disclaimer in the documentation
 and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING,
 BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
 AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
 OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "stretch_move_sampler.h"
#include "cl-helper.h"
#include "stretch_move_util.h"
#include "constants.h"

//    Stretch Move MCMC sampler in OpenCL
//    Alex Kaiser, Courant Institute, 2012
//    Email: user: adkaiser  domain: gmail

 
void example_simple();
void example_with_data();
void example_sde_inference(); 


int main(int argc, char **argv){

    // Simplest example of sampling possible
    example_simple();

    // Example of sampling a Gaussian where mean and covariance are passed to the function
    example_with_data();

    // Example of a more complex distribution for inference
    // Computationally intensive so off by default
    example_sde_inference();

    return 1;
}


void example_simple(){
    // Simple example of running the sampler.

    // User set parameters
    cl_long chain_length       = 10000;                    // Allocate to store this much chain, sampler runs this many steps at once
    cl_long dimension          = 10;                       // Dimension of the state vector
    cl_long walkers_per_group  = 1024;                     // Total number of walkers is twice this
    size_t work_group_size     = 32;                       // Work group size. Use 1 for CPU, larger number for GPU
    double a                   = 2.0;                      // Coefficient for range of 'z' random variable
    cl_long pdf_number         = 0;                        // Use Gaussian debug problem
    cl_long data_length        = 0;                        // No data for this example
    cl_float *data_temp        = NULL;                     // Need to pass a NULL pointer for the data
    const char *plat_name      = CHOOSE_INTERACTIVELY;     // Choose the platform interactively at runtime
    const char *dev_name       = CHOOSE_INTERACTIVELY;     // Choose the device interactively at runtime

    // set parameters about which components to save
    cl_long num_to_save        = 3;
    cl_int *indices_to_save    = (cl_int *) malloc(num_to_save * sizeof(cl_int));
    indices_to_save[0]         = 0;
    indices_to_save[1]         = 3;
    indices_to_save[2]         = 5;


    // Initialize the sampler
    sampler *samp = initialize_sampler(chain_length, dimension, walkers_per_group, work_group_size, a, pdf_number,
                                       data_length, data_temp, num_to_save, indices_to_save, plat_name, dev_name);

    // Run burn-in for 5000 steps
    int burn_length = 5000;
    run_burn_in(samp, burn_length);

    // Run the sampler
    run_sampler(samp);

    // --------------------------------------------------------------------------
    // The array samp->samples_host now contains samples ready for use.
    //
    // Array is in component major order.
    // To access the i-th saved sample of sample j use
    //     samp->samples_host[i + j*(samp->num_to_save)]
    //
    // Dimension is (samp->N x samp->total_samples)
    // --------------------------------------------------------------------------

    // Free resources
    free_sampler(samp);
}




void example_with_data(){

	// Example of running the sampler.
    // Similar to above, but uses data


    // User set parameters
    cl_long chain_length       = 10000;                   // Allocate to store this much chain, sampler runs this many steps at once
    cl_long dimension          = 10;                      // Dimension of the state vector
    cl_long walkers_per_group  = 2048;                    // Total number of walkers is twice this
    size_t work_group_size     = 32;                      // Work group size. Use 1 for CPU, larger number for GPU
    double a                   = 1.5;                     // Coefficient for range of 'z' random variable
    cl_long pdf_number         = 1;                       // Use pdf 1 for this problem
    const char *plat_name      = CHOOSE_INTERACTIVELY;    // Choose the platform interactively at runtime
    const char *dev_name       = CHOOSE_INTERACTIVELY;    // Choose the device interactively at runtime


    // set parameters about which components to save
    cl_long num_to_save        = 3;
    cl_int *indices_to_save    = (cl_int *) malloc(num_to_save * sizeof(cl_int));
    indices_to_save[0]         = 0;
    indices_to_save[1]         = 3;
    indices_to_save[2]         = 5;


    // Generate the mean and inverse covariance matrix
    // Pack mean first, then matrix
    cl_int data_length = dimension + dimension*dimension;
    cl_float *data = (cl_float *) malloc(data_length * sizeof(cl_float)) ;
    if(!data) { perror("Allocation failure data"); abort(); }

    for(int i=0; i < dimension; i++)
        data[i] = (cl_float) i;

    for(int i=dimension; i < data_length; i++){
        data[i] = 0.0f;
    }

    for(int i=0; i < (dimension-1); i++){
        data[ i   +     i*dimension + dimension ] =  2.0f;
        data[ i+1 +     i*dimension + dimension ] = -1.0f;
        data[ i   + (i+1)*dimension + dimension ] = -1.0f;
    }
    data[ (dimension-1) + (dimension-1)*dimension + dimension ] = 2.0f;


    // Initialize the sampler
    sampler *samp = initialize_sampler(chain_length, dimension, walkers_per_group, work_group_size, a, pdf_number,
                                       data_length, data, num_to_save, indices_to_save, plat_name, dev_name);

    // Initialize structure members for samp->data_st here, values will be copied


    // Run burn-in
    int burn_length = 10000;
    run_burn_in(samp, burn_length);


    // Run the sampler
    run_sampler(samp);

    // --------------------------------------------------------------------------
    // The array samp->samples_host now contains samples ready for use.
    //
    // Array is in component major order.
    // To access the i-th saved sample of sample j use
    //     samp->samples_host[i + j*(samp->num_to_save)]
    //
    // Dimension is (samp->N x samp->total_samples)
    // --------------------------------------------------------------------------

    // Print summary of the run including basic some statistics
    print_run_summary(samp);

    // Run acor to estimate autocorrelation time
    run_acor(samp);

    // Output some histograms to Matlab, don't output gnuplot
    char matlab_hist = 1, gnuplot_hist = 0;
    output_histograms(samp, matlab_hist, gnuplot_hist);

    // Free resources
    free_sampler(samp);
}

void example_sde_inference(){

    // User set parameters
    cl_long chain_length = 200000;                    // Allocate to store this much chain, sampler runs this many steps at once

    int burn_length = 1000000;                        // Length of burn in

    cl_long annealing_loops = 20;                     // Run this many temperatures of simulated annealing
    cl_long steps_per_loop = 50000;                   // This many steps each

    cl_long dimension = N_TH + N_STEPS * NX ;         // Dimension of the state vector
    cl_long walkers_per_group = 2048;                 // Total number of walkers is twice this
    size_t work_group_size = 64;                      // Work group size. Use 1 for CPU, larger number for GPU
    double a = 1.4;                                   // Coefficient for range of 'z' random variable
    cl_int pdf_number = 3;                            // Use pdf 3 for this problem
    const char *plat_name = CHOOSE_INTERACTIVELY;
    const char *dev_name  = CHOOSE_INTERACTIVELY;

    cl_float x_initial = 100.000000f;


    // set this parameter for a debug run
    int easy = 0;
    if(easy){
        chain_length     = 10000;
        burn_length      = 10000;
        steps_per_loop   = 1000;
    }



    // set parameters about which components to save
    cl_long num_to_save        = N_TH;
    cl_int *indices_to_save   = (cl_int *) malloc(num_to_save * sizeof(cl_int));
    for(int i=0; i<num_to_save; i++)
        indices_to_save[i] = i;


    // read the observations from a file
    // note: N_OBS, NY are defined in "constants.h"
    cl_long data_length = N_OBS * NY;
    cl_float *data = (cl_float *) malloc(data_length * sizeof(cl_float)) ;
    if(!data){ perror("Allocation failure obs_temp"); abort(); }
    char file_name[] = "noisy_data.txt";
    read_arrays(data, N_OBS, NY, file_name);


    // Initialize the sampler
    sampler *samp = initialize_sampler(chain_length, dimension, walkers_per_group, work_group_size, a, pdf_number,
                                       data_length, data, num_to_save, indices_to_save, plat_name, dev_name);


    // initialize the initial conditions in the struct
    // initial conditions are written in constand array in definitions
    (samp->data_st)->x_initial[0] = x_initial;


    // Run simulated annealing too speed up convergence
    // set a generic cooling schedule, {1/n, 1/(n-1) ... 1}
    cl_float *cooling_schedule = (cl_float *) malloc(annealing_loops * sizeof(cl_float));
    int idx=0;
    for(int i=annealing_loops; i>0; i--) cooling_schedule[idx++] = 1.0f / ( (cl_float) i);

    // run the annealing
    run_simulated_annealing(samp, cooling_schedule, annealing_loops, steps_per_loop);
    free(cooling_schedule);



    run_burn_in(samp, burn_length);


    // run the sampler
    run_sampler(samp);

    // --------------------------------------------------------------------------
    // The array samp->samples_host now contains samples ready for use.
    //
    // Array is in component major order.
    // To access the i-th saved sample of sample j use
    //     samp->samples_host[i + j*(samp->num_to_save)]
    //
    // Dimension is (samp->N x samp->total_samples)
    // --------------------------------------------------------------------------


    // print summary of the run including basic some statistics
    print_run_summary(samp);

    // run acor to estimate autocorrelation time
    run_acor(samp);

    // Output some histograms to Matlab, don't output gnuplot
    char matlab_hist = 1, gnuplot_hist = 0;
    output_histograms(samp, matlab_hist, gnuplot_hist);

    // free resources
    free_sampler(samp);

}

