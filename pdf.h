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


typedef struct{

    // Note that this struct must match the definition in data_struct.h
    // This includes the order of the members

    float beta;      // always include this

    // Add any other scalars here.
    // Small arrays of static length are also okay.
    // Pointers will cause errors, following OpenCL rules.

}data_struct;



float log_pdf(float *x, data_struct data_st, const __local float *data){
    /*
    Evaluate denormalized log of the PDF.

    Input:
        __local float *x              Location at which to evaluate PDF
        data_struct data_st           Structure containing any additional scalars or statically defined arrays.
        const __local float *data     Any observations or data.

    Output:
        returned:                     Log of the denormalized pdf that we are going to sample.
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
        const __local float *mu = data;   // mean

        // NN elements later is the inverse covariance matrix
        const __local float *inv_cov  = data + NN;

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
        float sum  = 0.0;
        for(int i=0; i<NN; i++)
            sum += temp[i] * (x[i] - mu[i]);

        // add negative half in front of whole thing
        return -0.5f * sum;
    }


    // No PDF selected.
    // Sampler then does nothing.
    return (-1.0f) / 0.0f;
}

