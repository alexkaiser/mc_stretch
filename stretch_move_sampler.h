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

#include "timing.h"
#include "cl-helper.h"
#include "constants.h"
#include "data_struct.h"

// If true this will make initial components nonnegative
// This is generically fine
#define NONNEGATIVE_BOX 1


// Chance the kernel compile flags to work with AMD compilers
// Note that the sampler is not currently tested with this hardware
// (but compiling will fail with this set to zero)
#define AMD 0


// Set the amount of output
// OUTPUT_LEVEL 0: Only fatal error messages
// OUTPUT_LEVEL 1: Print progress updates
// OUTPUT_LEVEL 2: Print information about selected device
#define OUTPUT_LEVEL 2


#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))


typedef struct{

        // user set parameters
        cl_int M;                           // Number of steps to run
        cl_int N;                           // Dimension of the problem and the walkers
        cl_int K_over_two;                  // Number of walkers in each group

        // derived parameters
        cl_int K;                           // Total walkers
        cl_int total_samples;               // Total samples produced


        // Note: components are in adjacent memory in the walker.
        // To access the ith component of walker j, take
        //     X_red_host[i + j*samp->N];

        // host arrays
        cl_float *X_red_host;                // Red walkers
        cl_float *log_pdf_red_host;          // Stored log pdf values
        cl_float *X_black_host;              // Black walkers
        cl_float *log_pdf_black_host;        // Stored log pdf values
        cl_ulong *accepted_host;             // Number accepted on each work item
        unsigned long accepted_total;        // Total number accepted
        cl_float *samples_host;              // The samples of the distribution
        cl_float *data_host;                 // Observation data
        cl_int data_length;                  // Number of observations
        data_struct *data_st;                // Structure for additional static data

        // store acor times for making histograms
        double *acor_times;

        // time stamps
        timestamp_type time1_total, time2_total;    // For total time
        timestamp_type time1, time2;                // For kernel time


        // OpenCL data

        // contexts and queues
        cl_context ctx;
        cl_command_queue queue;

        // work group size information
        size_t ldim[1];                             // Work group size
        size_t gdim[1];                             // Total dimension

        // kernels
        cl_kernel stretch_knl;                      // Sampler kernel
        cl_kernel init_rand_lux_knl;                // Random number generator initialization

        // device memory
        cl_mem X_red_device;                       // One array of walkers
        cl_mem log_pdf_red_device;                 // Stored log pdf values
        cl_mem X_black_device;                     // One array of walkers
        cl_mem log_pdf_black_device;               // Stored log pdf values
        cl_mem accepted_device;                    // Number accepted on each work item
        cl_mem data_device;                        // Data array
        cl_mem ranluxcltab;                        // State array for randluxcl
        cl_mem data_st_device;                     // Struct for user data

}sampler;


// sampling routines
sampler* initialize_sampler(cl_int chain_length, cl_int dimension,
                            cl_int walkers_per_group, size_t work_group_size,
                            cl_int pdf_number,
                            cl_int data_length, cl_float *data, data_struct *data_st,
                            const char *plat_name, const char *dev_name);

void run_simulated_annealing(sampler *samp, cl_float *cooling_schedule, cl_int annealing_loops, cl_int steps_per_loop);
void run_burn_in(sampler *samp, int burn_length);
void run_sampler(sampler *samp);
void print_run_summary(sampler *samp);
void run_acor(sampler *samp);
void output_histograms(sampler *samp, char matlab_hist, char gnuplot_hist);
void free_sampler(sampler* samp);

