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

}data_struct;



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

    if(PDF_NUMBER==0){
        // Gaussian string, default demo problem
        float sum = x[0]*x[0] + x[NN-1]*x[NN-1];
        for(int i=0; i<NN-1; i++){
            sum += (x[i+1] - x[i]) * (x[i+1] - x[i]);
        }
        return -0.5f * sum;
    }

    if(PDF_NUMBER==1){

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

    if(PDF_NUMBER==2){
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

    // No PDF selected.
    // Sampler then does nothing.
    return (-1.0f) / 0.0f;
}

