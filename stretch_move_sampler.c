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

#include "stretch_move_sampler.h"
#include "stretch_move_util.h"


sampler* initialize_sampler(cl_int chain_length, cl_int dimension,
                            cl_int walkers_per_group, size_t work_group_size,
                            cl_int pdf_number,
                            cl_int data_length, cl_float *data,
                            const char *plat_name, const char *dev_name){

    /*
     Initialize stretch move MCMC sampler struct.
     Arrange parameters into sampler struct pointer.
     Allocate arrays on host, initialize walkers and other values as appropriate.
     Start OpenCL context and queue.
     Allocate device memory and transfer from host.
     Compile and initialize random number generator.
     Compile stretch move OpenCL kernel.

     Input:
          cl_int pdf_number                  Which PDF to sample. Passed to pdf.h as a compile time definition.
          cl_int chain_length                Allocate space for this many samples in the sampler struct.
                                                 Sampler fills this array when run_sampler is called.
          cl_int dimension                   Dimension of state vector of Markov chain.
          cl_int walkers_per_group           Number of walkers in each of two groups. Total walkers is twice this.
          size_t work_group_size             Work group size.
                                                 For CPU this must be set to one.
                                                 For GPU this should be set larger, powers of two are optimal, try 64, 128 or 256.
                                                 This number must divide walkers_per_group.
          cl_int data_length                 Length of observation data. If no data set this to zero.
          cl_float *data                     Observation data.
          const char *plat_name              String for platform name. Set to CHOOSE_INTERACTIVELY (no quotes) to do so.
          const char *dev_name               String for device name. Set to CHOOSE_INTERACTIVELY (no quotes) to do so.

     Output:
          returned: sampler *samp            Pointer to sampler struct with parameters, arrays, context, queue, kernel initialized.
     */


    if(OUTPUT_LEVEL > 0) printf("Initializing Stretch Move sampler.\n");


    // --------------------------------------------------------------------------
    // Set parameters
    // --------------------------------------------------------------------------

    // This environment variable forces headers to be reloaded each time
    // If not set and pdf if changed, changes may not be updated
    setenv("CUDA_CACHE_DISABLE", "1", 1);

    // allocate the structure for all the sampler parameters and arrays
    sampler * samp = (sampler *) malloc(sizeof(sampler));
    if(!samp) { perror("Allocation failure sampler"); abort(); }

    // user set parameters
    samp->M = chain_length;                           // Number of steps to run
    samp->N = dimension;                              // Dimension of the problem and the walkers
    samp->K_over_two = walkers_per_group ;            // Number of walkers in each group

    // derived parameters
    samp->K = 2 * samp->K_over_two;                   // Total walkers
    samp->total_samples = samp->M * samp->K;          // Total samples produced

    // Allocate the structure
    samp->data_st = (data_struct *) malloc(sizeof(data_struct));
    if(!(samp->data_st)) { perror("Allocation failure data_struct"); abort(); }


    // default value one, unless performing simulated annealing
    (samp->data_st)->beta = 1.0f;

    // error check on dimensions
    if(samp->K <= samp->N){
        fprintf(stderr, "Error: Must have more walkers than the dimension.\nExiting\n");
        abort();
    }

    // error check on work sizes
    if( (samp->K_over_two % work_group_size) != 0){
        fprintf(stderr, "Error: Number of walkers in each group must be multiple of work group size.\nExiting\n");
        abort();
    }


    // for later output
    samp->acor_times = (double *) malloc(samp->N * sizeof(double));

    // write parameter file for plotting
    write_parameter_file_matlab(samp->M, samp->N, samp->K, "Stretch Move", samp->N, pdf_number);

    // --------------------------------------------------------------------------
    // Set up OpenCL context and queues
    // --------------------------------------------------------------------------
    if(OUTPUT_LEVEL > 0) printf("Begin opencl contexts.\n");

    create_context_and_two_queues_on(plat_name, dev_name, 0, &(samp->ctx), &(samp->queue), &(samp->queue_mem) , 0);

    // print information on selected device
    if(OUTPUT_LEVEL > 1)  print_device_info_from_queue(samp->queue);

    // set the work group sizes
    samp->ldim[0] = work_group_size;
    samp->gdim[0] = samp->K_over_two;

    if(OUTPUT_LEVEL > 0) printf("Context built.\n");


    // --------------------------------------------------------------------------
    // Start total timing
    // --------------------------------------------------------------------------
    if(OUTPUT_LEVEL > 0) printf("Begin total timing.\n");
    get_timestamp(&(samp->time1_total));


    // --------------------------------------------------------------------------
    // Allocate host memory
    // --------------------------------------------------------------------------

    // counter for number of samples accepted
    samp->accepted_host = (cl_ulong *) malloc(samp->K_over_two * sizeof(cl_ulong));
    if(!(samp->accepted_host)){ perror("Allocation failure accepted host"); abort(); }
    for(int i=0; i< (samp->K_over_two); i++) samp->accepted_host[i] = 0;

    // Adjacent memory on x_red moves with in the walker
    // To access the ith component of walker j, take x_red[i + j*N];

    // red walkers
    samp->X_red_host = (cl_float *) malloc(samp->N * samp->K_over_two * sizeof(cl_float));
    if(!(samp->X_red_host)){ perror("Allocation failure X_red_host"); abort(); }

    // log likelihood
    samp->log_pdf_red_host = (cl_float *) malloc(samp->K_over_two * sizeof(cl_float));
    if(!(samp->log_pdf_red_host)){ perror("Allocation failure X_red_host"); abort(); }
    for(int i=0; i<(samp->K_over_two); i++) samp->log_pdf_red_host[i] = (-1.0f) / 0.0f;

    // black walkers
    samp->X_black_host = (cl_float *) malloc(samp->N * samp->K_over_two * sizeof(cl_float));
    if(!(samp->X_black_host)){ perror("Allocation failure X_black_host"); abort(); }

    // log likelihood
    samp->log_pdf_black_host = (cl_float *) malloc(samp->K_over_two * sizeof(cl_float));
    if(!(samp->log_pdf_black_host)){ perror("Allocation failure X_red_host"); abort(); }
    for(int i=0; i< (samp->K_over_two); i++) samp->log_pdf_black_host[i] = (-1.0f) / 0.0f;

    // samples on host
    cl_int samples_length = samp->N * samp->M * samp->K;                // length of the samples array
    samp->samples_host = (cl_float *) malloc(samples_length * sizeof(cl_float));         // samples to return
    if(!(samp->samples_host)){ perror("Allocation failure samples_host"); abort(); }


    // intialize the walkers to random values
    // set the seed value
    srand48(0);

    // initialize the walkers to small random values
    for(int j=0; j < samp->N * samp->K_over_two; j++){
        if(NONNEGATIVE_BOX){
            samp->X_black_host[j] = (cl_float) drand48();
            samp->X_red_host[j]   = (cl_float) drand48();
        }
        else{
            samp->X_black_host[j] = (cl_float) (0.1 * (drand48()-0.5));
            samp->X_red_host[j]   = (cl_float) (0.1 * (drand48()-0.5));
        }

    }


    // copy the observations over
    samp->data_length = data_length;

    // there are lots of complications that appear if this is empty
    // make it length one instead
    if(samp->data_length == 0){
        samp->data_length = 1;
        samp->data_host = (cl_float *) malloc(samp->data_length * sizeof(cl_float)) ;
        if(!(samp->data_host)){ perror("Allocation failure data_host"); abort(); }
        samp->data_host[0] = 0.0f;
    }
    else{
        // standard case
        samp->data_host = data;
    }


    // --------------------------------------------------------------------------
    // load kernels
    // --------------------------------------------------------------------------

    // stretch move kernel
    char *knl_text = read_file("stretch_move.cl");
    char options[200];
    sprintf(options, "-DNN=%d -DK_OVER_TWO=%d -DDATA_LEN=%d -DPDF_NUMBER=%d -I . ",
            samp->N, samp->K_over_two, samp->data_length, pdf_number);    // define N at kernel compile time

    if(OUTPUT_LEVEL > 0) printf("Options string for stretch move kernel:%s\n", options);

    samp->stretch_knl = kernel_from_string(samp->ctx, knl_text, "stretch_move", options);
    free(knl_text);

    if(OUTPUT_LEVEL > 0) printf("Stretch Move kernel compiled.\n");

    // random number generator initialization
    char * knl_text_rand = read_file("Kernel_Ranluxcl_Init.cl");
    char options_rand_lux[100];

    if(AMD)
        sprintf(options_rand_lux, "-DRANLUXCL_LUX=4 -I .");
    else
        sprintf(options_rand_lux, "-DRANLUXCL_LUX=4");

    samp->init_rand_lux_knl = kernel_from_string(samp->ctx, knl_text_rand, "Kernel_Ranluxcl_Init", options_rand_lux);
    free(knl_text_rand);

    if(OUTPUT_LEVEL > 0) printf("Ranluxcl init kernel compiled.\n");



    // --------------------------------------------------------------------------
    // allocate device memory
    // --------------------------------------------------------------------------
    cl_int status;

    samp->X_red_device = clCreateBuffer(samp->ctx, CL_MEM_READ_WRITE,
      sizeof(cl_float) * samp->N * samp->K_over_two, 0, &status);
    CHECK_CL_ERROR(status, "clCreateBuffer");

    samp->log_pdf_red_device = clCreateBuffer(samp->ctx, CL_MEM_READ_WRITE,
      sizeof(cl_float) * samp->K_over_two, 0, &status);
    CHECK_CL_ERROR(status, "clCreateBuffer");

    samp->X_black_device = clCreateBuffer(samp->ctx, CL_MEM_READ_WRITE,
      sizeof(cl_float) * samp->N * samp->K_over_two, 0, &status);
    CHECK_CL_ERROR(status, "clCreateBuffer");

    samp->log_pdf_black_device = clCreateBuffer(samp->ctx, CL_MEM_READ_WRITE,
      sizeof(cl_float) * samp->K_over_two, 0, &status);
    CHECK_CL_ERROR(status, "clCreateBuffer");

    samp->accepted_device = clCreateBuffer(samp->ctx, CL_MEM_READ_WRITE,
            samp->K_over_two * sizeof(cl_ulong), 0, &status);
    CHECK_CL_ERROR(status, "clCreateBuffer");


    // allocate for the observations
    samp->data_device = clCreateBuffer(samp->ctx, CL_MEM_READ_WRITE,
      sizeof(cl_float) * samp->data_length, 0, &status);
    CHECK_CL_ERROR(status, "clCreateBuffer");

    // data struct on device
    samp->data_st_device = clCreateBuffer(samp->ctx, CL_MEM_READ_WRITE,
            sizeof(data_struct), 0, &status);
    CHECK_CL_ERROR(status, "clCreateBuffer");


    // allocate for the state array for randluxcl
    // use a 1d work group
    size_t rand_lux_state_buffer_size = samp->gdim[0] * 7 * sizeof(cl_float4);
    samp->ranluxcltab = clCreateBuffer(samp->ctx, CL_MEM_READ_WRITE,
        rand_lux_state_buffer_size, 0, &status);
    CHECK_CL_ERROR(status, "clCreateBuffer");


    // --------------------------------------------------------------------------
    // transfer to device
    // --------------------------------------------------------------------------

    CALL_CL_GUARDED(clEnqueueWriteBuffer, (
        samp->queue, samp->X_red_device, /*blocking*/ CL_TRUE, /*offset*/ 0,
        samp->N * samp->K_over_two * sizeof(cl_float), samp->X_red_host,
        0, NULL, NULL));

    CALL_CL_GUARDED(clEnqueueWriteBuffer, (
        samp->queue, samp->log_pdf_red_device, /*blocking*/ CL_TRUE, /*offset*/ 0,
        samp->K_over_two * sizeof(cl_float), samp->log_pdf_red_host,
        0, NULL, NULL));

    CALL_CL_GUARDED(clEnqueueWriteBuffer, (
        samp->queue, samp->X_black_device, /*blocking*/ CL_TRUE, /*offset*/ 0,
        samp->N * samp->K_over_two * sizeof(cl_float), samp->X_black_host,
        0, NULL, NULL));

    CALL_CL_GUARDED(clEnqueueWriteBuffer, (
        samp->queue, samp->log_pdf_black_device, /*blocking*/ CL_TRUE, /*offset*/ 0,
        samp->K_over_two * sizeof(cl_float), samp->log_pdf_black_host,
        0, NULL, NULL));

    CALL_CL_GUARDED(clEnqueueWriteBuffer, (
        samp->queue, samp->data_device, /*blocking*/ CL_TRUE, /*offset*/ 0,
        samp->data_length * sizeof(cl_float), samp->data_host,
        0, NULL, NULL));

    CALL_CL_GUARDED(clEnqueueWriteBuffer, (
        samp->queue, samp->data_st_device, /*blocking*/ CL_TRUE, /*offset*/ 0,
        sizeof(data_struct), samp->data_st,
        0, NULL, NULL));

    CALL_CL_GUARDED(clFinish, (samp->queue));


    // --------------------------------------------------------------------------
    // Initialize random number generator
    // --------------------------------------------------------------------------

    // int for state variable initialization
    cl_int ins = 1;
    SET_2_KERNEL_ARGS(samp->init_rand_lux_knl, ins, samp->ranluxcltab);

    CALL_CL_GUARDED(clEnqueueNDRangeKernel,
          (samp->queue, samp->init_rand_lux_knl,
           /*dimensions*/ 1, NULL, samp->gdim, samp->ldim,
           0, NULL, NULL));

    CALL_CL_GUARDED(clFinish, (samp->queue));

    // --------------------------------------------------------------------------
    // Sampler initialization is done
    // --------------------------------------------------------------------------
    if(OUTPUT_LEVEL > 0) printf("Sampler initialized.\n");
    return samp;
}


void run_simulated_annealing(sampler *samp, cl_float *cooling_schedule, cl_int annealing_loops, cl_int steps_per_loop){
    /*
     Run the simulated annealing to allow the walkers to explore the space
         and (hopefully) increase convergence speed.
     Discard all the samples generated by this routine.
     Reset all the counters for acceptance rates.

     Input:
          sampler *samp                  Pointer to sampler structure which has been initialized.
          cl_float *cooling_schedule     Values of beta for the simulated annealing
                                         Values should be increasing and the final value should be one
          cl_int annealing_loops         Number of loops
          cl_int steps_per_loop          Iterations per loop

     Output:
                                         Pre-allocated sampler arrays now have had simulated annealing performed.
     */

    for(int annealing_step=0; annealing_step<annealing_loops; annealing_step++){

        // set the beta value for this iteration
        (samp->data_st)->beta = cooling_schedule[annealing_step];

        // update the data structure accordingly
        CALL_CL_GUARDED(clEnqueueWriteBuffer, (
            samp->queue, samp->data_st_device, /*blocking*/ CL_TRUE, /*offset*/ 0,
            sizeof(data_struct), samp->data_st,
            0, NULL, NULL));
        CALL_CL_GUARDED(clFinish, (samp->queue));

        for(int it=0; it<steps_per_loop; it++){
            SET_7_KERNEL_ARGS(samp->stretch_knl,
                  samp->X_red_device,
                  samp->log_pdf_red_device,
                  samp->X_black_device,
                  samp->ranluxcltab,
                  samp->accepted_device,
                  samp->data_device,
                  samp->data_st_device);

            CALL_CL_GUARDED(clEnqueueNDRangeKernel,
                  (samp->queue, samp->stretch_knl,
                   1, NULL, samp->gdim, samp->ldim,
                   0, NULL, NULL));

            SET_7_KERNEL_ARGS(samp->stretch_knl,
                  samp->X_black_device,
                  samp->log_pdf_black_device,
                  samp->X_red_device,
                  samp->ranluxcltab,
                  samp->accepted_device,
                  samp->data_device,
                  samp->data_st_device);

            CALL_CL_GUARDED(clEnqueueNDRangeKernel,
                  (samp->queue, samp->stretch_knl,
                   1, NULL, samp->gdim, samp->ldim,
                   0, NULL, NULL));

            CALL_CL_GUARDED(clFinish, (samp->queue));
        }

        if(OUTPUT_LEVEL > 0) printf("Annealing iteration %d\n", annealing_step * steps_per_loop);
    }

    // reset the acceptance counter after the annealing
    for(int i=0; i< (samp->K_over_two); i++) samp->accepted_host[i] = 0;
    CALL_CL_GUARDED(clEnqueueWriteBuffer, (
        samp->queue, samp->accepted_device, /*blocking*/ CL_TRUE, /*offset*/ 0,
        samp->K_over_two * sizeof(cl_ulong), samp->accepted_host,
        0, NULL, NULL));

    CALL_CL_GUARDED(clFinish, (samp->queue));

}



void run_burn_in(sampler *samp, int burn_length){
    /*
     Run the sampler to burn in.
     Discard all the samples generated by this routine.
     Reset all the counters for acceptance rates.

     Input:
          sampler *samp        Pointer to sampler structure which has been initialized.
          int burn_length      Number of burn in steps to run.

     Output:
          Pre-allocated sampler arrays now have had burn-in performed.
     */

    // reset beta
    (samp->data_st)->beta = 1.0f;

    // update the data structure accordingly
    CALL_CL_GUARDED(clEnqueueWriteBuffer, (
        samp->queue, samp->data_st_device, /*blocking*/ CL_TRUE, /*offset*/ 0,
        sizeof(data_struct), samp->data_st,
        0, NULL, NULL));
    CALL_CL_GUARDED(clFinish, (samp->queue));

    // do the burn in
    for(int it=0; it<burn_length; it++){

        SET_7_KERNEL_ARGS(samp->stretch_knl,
              samp->X_red_device,
              samp->log_pdf_red_device,
              samp->X_black_device,
              samp->ranluxcltab,
              samp->accepted_device,
              samp->data_device,
              samp->data_st_device );

        CALL_CL_GUARDED(clEnqueueNDRangeKernel,
              (samp->queue, samp->stretch_knl,
               1, NULL, samp->gdim, samp->ldim,
               0, NULL, NULL));

        SET_7_KERNEL_ARGS(samp->stretch_knl,
              samp->X_black_device,
              samp->log_pdf_black_device,
              samp->X_red_device,
              samp->ranluxcltab,
              samp->accepted_device,
              samp->data_device,
              samp->data_st_device);

        CALL_CL_GUARDED(clEnqueueNDRangeKernel,
              (samp->queue, samp->stretch_knl,
               1, NULL, samp->gdim, samp->ldim,
               0, NULL, NULL));

        CALL_CL_GUARDED(clFinish, (samp->queue));

        if( ((it % MAX((burn_length/10),1)) == 0) && (OUTPUT_LEVEL > 0))
                printf("Burn iteration %d\n", it);
    }


    // make sure everything is done with the burn in
    CALL_CL_GUARDED(clFinish, (samp->queue));

    // reset the acceptance counter after the burn in
    for(int i=0; i< (samp->K_over_two); i++) samp->accepted_host[i] = 0;
    CALL_CL_GUARDED(clEnqueueWriteBuffer, (
        samp->queue, samp->accepted_device, /*blocking*/ CL_TRUE, /*offset*/ 0,
        samp->K_over_two * sizeof(cl_ulong), samp->accepted_host,
        0, NULL, NULL));

    CALL_CL_GUARDED(clFinish, (samp->queue));

    if(OUTPUT_LEVEL > 0) printf("Burn in complete.\n");

}



void run_sampler(sampler *samp){
    /*
     Run the sampler and save output.
     Overlap sampling and communication with the device using two queues.
     While red walkers are being sampled, black walkers are being sent.
     This means that the first sampling iteration reads black walkers from the burn in,
     and the final iteration is thrown out.

     Runs in the following order.
         - Sample X_red non-blocking
         - Copy X_black to host non-blocking
         - Check both are finished
         - Sample X_black non-blocking
         - Copy X_red to host non-blocking
         - Check both are finished

     Input:
          sampler *samp        Pointer to sampler structure which has been initialized.
                                    Burn-in should also be performed before running this routine.
                                    Run for samp->M total times.

     Output:
                                    Array samp->samples_host is filled with new samples.
     */

    get_timestamp(& (samp->time1));

    // reset beta
    (samp->data_st)->beta = 1.0f;

    // update the data structure accordingly
    CALL_CL_GUARDED(clEnqueueWriteBuffer, (
        samp->queue, samp->data_st_device, /*blocking*/ CL_TRUE, /*offset*/ 0,
        sizeof(data_struct), samp->data_st,
        0, NULL, NULL));
    CALL_CL_GUARDED(clFinish, (samp->queue));

    // run the sampler
    unsigned int buffer_position = 0;


    for(int it=0; it<samp->M; it++){

        // update X_red
        SET_7_KERNEL_ARGS(samp->stretch_knl,
              samp->X_red_device,
              samp->log_pdf_red_device,
              samp->X_black_device,
              samp->ranluxcltab,
              samp->accepted_device,
              samp->data_device,
              samp->data_st_device);

        CALL_CL_GUARDED(clEnqueueNDRangeKernel,
              (samp->queue, samp->stretch_knl,
               /*dimensions*/ 1, NULL, samp->gdim, samp->ldim,
               0, NULL, NULL));

        // read the constant samples while others are updating
        CALL_CL_GUARDED(clEnqueueReadBuffer, (
            samp->queue_mem, samp->X_black_device, CL_FALSE, 0,
            samp->N * samp->K_over_two * sizeof(cl_float), samp->samples_host + buffer_position,
            0, NULL, NULL));

        buffer_position += samp->N * samp->K_over_two;

        // both must finish before next iteration
        CALL_CL_GUARDED(clFinish, (samp->queue_mem));
        CALL_CL_GUARDED(clFinish, (samp->queue));


        // update X_black
        SET_7_KERNEL_ARGS(samp->stretch_knl,
              samp->X_black_device,
              samp->log_pdf_black_device,
              samp->X_red_device,
              samp->ranluxcltab,
              samp->accepted_device,
              samp->data_device,
              samp->data_st_device);

        CALL_CL_GUARDED(clEnqueueNDRangeKernel,
              (samp->queue, samp->stretch_knl,
               /*dimensions*/ 1, NULL, samp->gdim, samp->ldim,
               0, NULL, NULL));

        // read the constant samples while others are updating
        CALL_CL_GUARDED(clEnqueueReadBuffer, (
            samp->queue_mem, samp->X_red_device, CL_FALSE, 0,
            samp->N * samp->K_over_two * sizeof(cl_float), samp->samples_host + buffer_position,
            0, NULL, NULL));

        buffer_position += samp->N * samp->K_over_two;

        // both must finish before next iteration
        CALL_CL_GUARDED(clFinish, (samp->queue_mem));
        CALL_CL_GUARDED(clFinish, (samp->queue));

        if( ((it % (MAX(samp->M/10,1))) == 0) && (OUTPUT_LEVEL > 0) )
            printf("Sample iteration %d\n", it);

    }

    // make sure everything is back in place
    CALL_CL_GUARDED(clFinish, (samp->queue));

    // take the end of the kernel timing
    get_timestamp(&(samp->time2));

    // save the acceptance probability
    CALL_CL_GUARDED(clEnqueueReadBuffer, (
        samp->queue, samp->accepted_device, CL_TRUE, 0,
        samp->K_over_two * sizeof(cl_ulong), samp->accepted_host,
        0, NULL, NULL));

    // ensure that all reads are finished
    CALL_CL_GUARDED(clFinish, (samp->queue));

    samp->accepted_total = 0;
    for(int i=0; i<samp->K_over_two; i++)
        samp->accepted_total += samp->accepted_host[i];

    if(OUTPUT_LEVEL > 0) printf("Sampler kernel ran and completed.\n\n");

}


void print_run_summary(sampler *samp){
    /*
     Stop the global timer and print a small summary of the run.

     Input:
          sampler *samp        Pointer to sampler structure which has been initialized.
                               Sampling must be performed before running this routine.

     Output:
          Print a short summary of the run, including sample rate and acceptance rate.
          Print the the mean and standard deviation of all components sampled.
     */

    double elapsed_sample = timestamp_diff_in_seconds(samp->time1, samp->time2);

    get_timestamp(&(samp->time2_total));
    double elapsed_total = timestamp_diff_in_seconds(samp->time1_total, samp->time2_total);

    // --------------------------------------------------------------------------
    // check output
    // --------------------------------------------------------------------------

    printf("Time steps = %d\n", samp->M);
    printf("Total samples = %d\n", samp->M * samp->K);
    printf("ldim = %d\tgdim = %d\n", (int) samp->ldim[0], (int) samp->gdim[0]);
    printf("Total accepted = %lu\n", samp->accepted_total);
    printf("Acceptance rate = %f\n", (cl_float) samp->accepted_total / ((cl_float) (samp->M * samp->K)) ) ;
    printf("Time for kernel runs = %f\n", elapsed_sample);
    printf("Sample rate, kernel time only = %f million samples / s\n", samp->M * samp->K * 1e-6 / elapsed_sample);

    printf("Total time = %f\n", elapsed_total);
    printf("Sample rate, total time = %f million samples / s\n", samp->M * samp->K * 1e-6 / elapsed_total);
    printf("\n");


    // Basic numerical estimate of mean and standard deviation of each component in the chain
    double mean, sigma;
    float *X = (float *) malloc(samp->total_samples * sizeof(float));
    if(!X){ perror("Allocation failure basic stats"); abort(); }

    for(int i=0; i<samp->N; i++){

        for(int j=0; j<samp->total_samples; j++)
            X[j] = samp->samples_host[i + j * (samp->N)];

        compute_mean_stddev(X, &mean, &sigma, samp->total_samples);

        printf("Statistics for X_%d:\t", i);
        printf("Mean = %f,\tsigma = %f\n", mean, sigma);

    }
    printf("\n");
    free(X);
}


void run_acor(sampler *samp){
    /*
     Run acor module to compute autocorrelation time.

     Input:
          sampler *samp        Pointer to sampler structure which has been initialized.
                                    Sampling must be performed before running this routine.

     Output:
          samp->acor_times     Array is filled with ensemble autocorrelation time for each component.
                                    Print a short summary of the ensemble autocorrelation times.
     */

    printf("From acor on ensemble:\n");

    double mean, sigma, tau;

    // one ensemble mean per time step
    double *ensemble_means = (double *) malloc(samp->M * sizeof(double));
    if(!ensemble_means){ perror("Allocation failure acor"); abort(); }


    // use every ensemble mean
    int L = samp->M;
    int acor_status;

    // For each component
    for(int i=0; i<samp->N; i++){

        // calculate the ensemble mean for this time step
        for(int t=0; t<samp->M; t++){
            ensemble_means[t] = 0.0;
            for(int kk=0; kk<samp->K; kk++)
                ensemble_means[t] += (double) samp->samples_host[i + (kk * samp->N) + t*(samp->N * samp->K)];
            ensemble_means[t] /= ((double) samp->K);
        }

        // generate the statistics.
        acor_status = acor(&mean, &sigma, &tau, ensemble_means, L);

        samp->acor_times[i] = tau;

        if(acor_status){
            printf("Acor error on component %d\n", i);
        }

        printf("Acor ensemble statistics for X_%d:\t", i);
        printf("Mean = %f,\tsigma = %f,\tAutocorrelation time tau %f", mean, sigma, tau);

        if(!acor_status)
            // acor passed
            printf(",\tEffective independent samples: %d\n", (int) (samp->total_samples / tau));
        else
            printf("\n");

    }

    printf("\n");

    free(ensemble_means);
}


void output_histograms(sampler *samp, char matlab_hist, char gnuplot_hist){
    /*
     Compute histograms for the given sampler.
     Number of bins in this script is always 100.
     Bin locations are picked dynamically to not lose samples.

     Input:
          sampler *samp            Sampler object with sampler already run.
          char matlab_hist         If true, will write matlab format data files.
          char gnuplot_hist        If true, will write gnuplot format data files.

     Output:
          Write data files for histograms.
     */

    int n_bins = 100;
    double tau;

    float *centers = (float *) malloc(n_bins * sizeof(float));
    if(!centers){ perror("Allocation failure Histogram"); abort(); }
    float *f_hat = (float *) malloc(n_bins * sizeof(float));
    if(!f_hat){ perror("Allocation failure Histogram"); abort(); }
    float *sigma_f_hat = (float *) malloc(n_bins * sizeof(float));
    if(!sigma_f_hat){ perror("Allocation failure Histogram"); abort(); }
    float *X = (float *) malloc(samp->total_samples * sizeof(float));
    if(!X){ perror("Allocation failure Histogram"); abort(); }

    for(int i=0; i< (samp->N); i++){

        for(int j=0; j < (samp->total_samples); j++)
            X[j] = samp->samples_host[i + j * samp->N];

        tau = samp->acor_times[i];
        histogram_data(n_bins, X, samp->total_samples, tau, centers, f_hat, sigma_f_hat);

        if(matlab_hist)
            histogram_to_matlab(n_bins, centers, f_hat, sigma_f_hat, i+1); // add one for matlab index
        if(gnuplot_hist)
            histogram_to_gnuplot(n_bins, centers, f_hat, i);
    }

    free(centers);
    free(f_hat);
    free(sigma_f_hat);
    free(X);


}


void free_sampler(sampler* samp){
    /* Free all resources allocated by the sampler and the sampler itself. */

    // free up OpenCL memory
    CALL_CL_GUARDED(clReleaseMemObject, (samp->X_red_device));
    CALL_CL_GUARDED(clReleaseMemObject, (samp->log_pdf_red_device));
    CALL_CL_GUARDED(clReleaseMemObject, (samp->X_black_device));
    CALL_CL_GUARDED(clReleaseMemObject, (samp->log_pdf_black_device));
    CALL_CL_GUARDED(clReleaseMemObject, (samp->accepted_device));
    CALL_CL_GUARDED(clReleaseMemObject, (samp->data_device));
    CALL_CL_GUARDED(clReleaseMemObject, (samp->ranluxcltab));
    CALL_CL_GUARDED(clReleaseMemObject, (samp->data_st_device));

    // kernels, context and queues
    CALL_CL_GUARDED(clReleaseKernel,       (samp->stretch_knl));
    CALL_CL_GUARDED(clReleaseKernel,       (samp->init_rand_lux_knl));
    CALL_CL_GUARDED(clReleaseCommandQueue, (samp->queue));
    CALL_CL_GUARDED(clReleaseContext,      (samp->ctx));

    // free host resources
    free(samp->X_red_host);
    free(samp->log_pdf_red_host);
    free(samp->X_black_host);
    free(samp->log_pdf_black_host);
    free(samp->samples_host);
    free(samp->accepted_host);
    free(samp->data_host);
    free(samp->data_st);

    // data resources
    free(samp->acor_times);

    free(samp);
}
