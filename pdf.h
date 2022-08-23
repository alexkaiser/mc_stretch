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

#include "constants.h"


// define this flag to put proposal array into local memory
#define USE_LOCAL_PROPOSAL

// if this is not defined
#ifdef USE_LOCAL_PROPOSAL
    #define PROPOSAL_TYPE  __local const float
#else
    #define PROPOSAL_TYPE          const float
#endif


// define this flag to put data array into local memory
#define USE_LOCAL_DATA

// if this is not defined
#ifdef USE_LOCAL_DATA
    #define DATA_ARRAY_TYPE  __local const float
#else
    #define DATA_ARRAY_TYPE __global const float
#endif



typedef struct{

    // Note that this struct must match the definition in data_struct.h
    // This includes the order of the members

    float beta;      // always include this
    int num_to_save;
    int save;

    // Add any other scalars here.
    // Small arrays of static length are also okay.
    // Pointers will cause errors, following OpenCL rules.

    #ifndef NX
        #define NX 1
    #endif

    // SDE solve specific extra array
    // this is a single scalar if NX is not defined
    float x_initial[NX];


}data_struct;


// logarithm of the PDF always has this form
float log_pdf(PROPOSAL_TYPE *x, __global const data_struct *data_st, DATA_ARRAY_TYPE *data);


// SDE solve specific functions
#ifdef PDF_NUMBER
    #if PDF_NUMBER == 3
        float log_prior(PROPOSAL_TYPE *x);
        float log_p_obs_given_parameters(PROPOSAL_TYPE *x, __global const data_struct *data_st, DATA_ARRAY_TYPE *data);
        void f(const float *x, PROPOSAL_TYPE *A, float *f_val);
        void g(const float *X, PROPOSAL_TYPE *B, float *Y);
    #endif
#endif



float log_pdf(PROPOSAL_TYPE *x, __global const data_struct *data_st, DATA_ARRAY_TYPE *data){
    /*
    Evaluate denormalized log of the PDF.

    Input:
        float *x                        Location at which to evaluate PDF
        data_struct data_st             Structure containing any additional scalars or statically defined arrays.
        DATA_ARRAY_TYPE *data           Any observations or data.

    Output:
        returned:                       Log of the denormalized pdf that we are going to sample.
    */

    if(PDF_NUMBER == 0){
        // Gaussian string, default demo problem
        float sum = x[0]*x[0] + x[NN-1]*x[NN-1];
        for(int i=0; i<NN-1; i++){
            sum += (x[i+1] - x[i]) * (x[i+1] - x[i]);
        }
        return -0.5f * sum;
    }

    if(PDF_NUMBER == 1){

        // Simple example of unpacking data
        // Data is packed as follows:
        //     Vector of length NN, means mu
        //       mu = data[0]  ... data[NN-1]
        //
        //     Matrix size NN*NN, inverse covariance matrix H
        //         inv_cov  = data[NN] ... data[NN + NN*NN - 1]
        //
        // Set pointers to the data rather than copy for speed.
        //

        // The first elements of the data array are the means
        DATA_ARRAY_TYPE *mu       = data;

        // NN elements later is the inverse covariance matrix
        DATA_ARRAY_TYPE *inv_cov  = data + NN;

            // temp array for inner products
        float temp[NN];

        // Compute the matrix vector product
        // temp = (x - mu)^(t) * inv_cov
        for(int j=0; j<NN; j++){
            temp[j] = 0.0f;
            for(int i=0; i<NN; i++)
                temp[j] +=  (x[i] - mu[i]) * inv_cov[i + j*NN];
        }

        // Compute the remaining dot product
        // sum = (x - mu)^(t) * inv_cov * (x - mu)
        float sum  = 0.0f;
        for(int i=0; i<NN; i++)
            sum += temp[i] * (x[i] - mu[i]);

        // add negative half in front of whole thing
        return -0.5f * sum;
    }

    if(PDF_NUMBER == 2){
        // Gaussian string with restriction
        // PDF is identical, but restricted to the case when all components
        // of the sample are non-negative
        float sum = x[0]*x[0] + x[NN-1]*x[NN-1];
        for(int i=0; i<NN-1; i++){

            // abort trap for the nonnegative case
            if(x[i] < 0)
                return (-1.0f)/0.0f;

            sum += (x[i+1] - x[i]) * (x[i+1] - x[i]);
        }

        // abort trap for the nonnegative case
        if(x[NN-1] < 0)
            return (-1.0f)/0.0f;

        return -0.5f * sum;
    }

    if(PDF_NUMBER == 3){
        // flat prior, either returns zero or -inf for out of bounds
        float log_p_theta         = log_prior(x);

        // abort for off the prior
        if(isinf(log_p_theta))
            return (-1.0f) / 0.0f;

        float log_p_d_given_theta = log_p_obs_given_parameters(x, data_st, data);

        // sum the two for the whole log probability
        return log_p_theta + log_p_d_given_theta ;
    }

    // No PDF selected.
    // Sampler then does nothing.
    return (-1.0f) / 0.0f;
}



// SDE solve specific functions
#ifdef PDF_NUMBER
    #if PDF_NUMBER == 3

        // definitions for start of indices in the array
        // parameters are listed (a, sig, y, zeta, X)
        // where X is the path
        // a,sig     - length NX
        // b, zeta   - length NY
        // X         - length NX * N_STEPS
        #define A_START_IDX                          0
        #define SIG_START_IDX        (A_START_IDX + NX)
        #define B_START_IDX        (SIG_START_IDX + NX)
        #define ZETA_START_IDX       (B_START_IDX + NY)
        #define X_START_IDX       (ZETA_START_IDX + NY)


        void f(const float *x, PROPOSAL_TYPE *A, float *f_val){
            // deterministic part of the dynamics

            // These dynamics: f = -a*x;
            for(int i=0; i<NX; i++)
                f_val[i] = -A[i] * x[i];

        }


        void g(const float *x, PROPOSAL_TYPE *B, float *Y){
            // deterministic part of observation

            for(int i=0; i<NY; i++)
                Y[i] = B[i] * x[i];

        }


        float log_prior(PROPOSAL_TYPE *x){
            // log of the prior

            // pointers for portions of the state array
            PROPOSAL_TYPE *A        = x +    A_START_IDX;
            PROPOSAL_TYPE *sigma    = x +  SIG_START_IDX;
            PROPOSAL_TYPE *B        = x +    B_START_IDX;
            PROPOSAL_TYPE *zeta     = x + ZETA_START_IDX;
            PROPOSAL_TYPE *x_path   = x +    X_START_IDX;

            // A
            for(int i=0; i<NX; i++)
                if( (A[i] < A_MIN) || (A[i] > A_MAX))
                    return (-1.0f)/0.0f;

            // sigma
            for(int i=0; i<NX; i++)
                if( (sigma[i] < SIG_MIN) || (sigma[i] > SIG_MAX))
                    return (-1.0f)/0.0f;

            // B
            for(int i=0; i<NY; i++)
                if( (B[i] < B_MIN) || (B[i] > B_MAX))
                    return (-1.0f)/0.0f;

            // zeta
            for(int i=0; i<NY; i++)
                if( (zeta[i] < ZETA_MIN) || (zeta[i] > ZETA_MAX))
                    return (-1.0f)/0.0f;

            // X
            for(int j=0; j<N_STEPS; j++)
                for(int i=0; i<NX; i++)
                    if( (x_path[i + j*NX] < X_MIN) || (x_path[i + j*NX] > X_MAX))
                        return (-1.0f)/0.0f;

            return 0.0f;
        }


        float log_p_obs_given_parameters(PROPOSAL_TYPE *x, __global const data_struct *data_st, DATA_ARRAY_TYPE *data){
            // Log of P(D|theta)

            // state variables
            float x_pred[NX];
            float x_prev[NX];
            float x_sample_current[NX];


            // pointers for portions of the state array
            PROPOSAL_TYPE *A        = x +    A_START_IDX;
            PROPOSAL_TYPE *sigma    = x +  SIG_START_IDX;
            PROPOSAL_TYPE *B        = x +    B_START_IDX;
            PROPOSAL_TYPE *zeta     = x + ZETA_START_IDX;
            PROPOSAL_TYPE *x_path   = x +    X_START_IDX;


            // temp variables
            float deviation;
            float f_val[NX];
            float g_val[NY];

            // read the initial conditions
            for(int i=0; i<NX; i++)
                x_prev[i] = data_st->x_initial[i];       // this is a constant global array


            // coefficients for variances
            float sig_squared_H[NX];
            for(int i=0; i<NX; i++)
                sig_squared_H[i] = H * sigma[i] * sigma[i];

            float zeta_squared[NY];
            for(int i=0; i<NY; i++)
                zeta_squared[i] =  zeta[i] * zeta[i];


            // compute the log PDF from the conditional distribution P(D|theta)
            float log_p_d_given_theta = 0.0f;

            // loop over total number of observations
            for(int i=0; i<N_OBS; i++){

                // run until the next observation
                for(int j=0; j<OBS_FREQ; j++){

                    // calculate the deterministic part of the dynamics
                    f(x_prev, A, f_val);

                    // loop over X components
                    for(int k=0; k<NX; k++){

                        // the sample for the current step
                        x_sample_current[k] = x_path[k + j*NX + i*OBS_FREQ*NX];

                        // fwd Euler step on the current dynamics
                        x_pred[k] = x_prev[k] + H * f_val[k] ;

                        // update the conditional
                        deviation = x_sample_current[k] - x_pred[k] ;
                        log_p_d_given_theta += deviation * deviation / sig_squared_H[k];

                        // Update the previous value from the current sample
                        x_prev[k] = x_sample_current[k];
                    }
                }

                // contribution of the observation
                for(int k=0; k<NY; k++){

                    // deterministic part of observation, g = b * X
                    g(x_sample_current, B, g_val);
                    deviation = g_val[k] - data[k + i*NY];
                    log_p_d_given_theta += deviation * deviation / zeta_squared[k];
                }
            }

            // add the negative half factor in front of the whole thing
            log_p_d_given_theta *= -0.5f;

            // add components for the variances
            for(int i=0; i<NX; i++)
                log_p_d_given_theta +=  -N_STEPS * log(sigma[i]);

            for(int i=0; i<NY; i++)
                log_p_d_given_theta +=    -N_OBS * log(zeta[i]);

            return log_p_d_given_theta;
        }

    #endif
#endif



