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


// Required compile time definitions. 
// These are set up automatically in the initialize routine 
// 
// #define NN                             | Dimension, passing as "N" sometimes creates conflict
// #define K_OVER_TWO                     | Number of walkers in each group 
// #define DATA_LEN                       | Length of data array
//


// default to highest quality generator if not specified 
#ifndef RANLUXCL_LUX
    #define RANLUXCL_LUX 4        
#endif 


#include "ranluxcl.cl"
#include "pdf.h"
#include "constants.h"


__kernel void stretch_move(
    __global float *X_moving,                  // walkers to be updated
    __global float *log_prob_moving,           // cached log probabilities of the moving walkers, will be updated 
    __global const float *X_fixed,             // fixed walkers 
    __global float4 *ranluxcltab,              // state information for random number generator 
    __global unsigned long *accepted,          // number of samples accepted 
    __global const float *data,                // data or observations
    __global const data_struct *data_st,       // structure with additional variables and parameters
    __constant int *indices_to_save,           // which components to save 
    __global float *X_save ){                  // the saved components 
     
      
    // start up data structures for the random number generator 
    // ranluxclstate is a struct of 7 * total_work_items float4 variables
    // storing the state of the generator.
    ranluxcl_state_t ranluxclstate;

    //Download state into ranluxclstate struct.
    ranluxcl_download_seed(&ranluxclstate, ranluxcltab);

    // get indexing information 
    int lid             = get_local_id(0);
    int k               = get_global_id(0);
     
    
    // allocate for proposal 
    #ifdef USE_LOCAL_PROPOSAL
        // allocate local if size permits 
        __local float Y[NN * WORK_GROUP_SIZE];       
    
        // the first index owned by this work item in the local arrays
        const int start_idx = NN * lid;    
    #else  
        // allocate into private, likely spills into global 
        float Y[NN]; 
        const int start_idx = 0;
    #endif 
    
    // temps 
    float z, q, log_py, log_pxk;
    int j; 
    
    // random numbers go here 
    float4 xi; 
        
    #ifdef USE_LOCAL_DATA 
        // allocate more local for the observation arrays 
        __local float data_local[DATA_LEN]; 
        for(int i=lid; i<DATA_LEN; i += WORK_GROUP_SIZE)
            data_local[i] = data[i];  
    
        // make sure all the copies have gone through 
        barrier(CLK_LOCAL_MEM_FENCE);    
    #endif
    
    // if we somehow start more work items than there are walkers in each group, move on 
    if(k < K_OVER_TWO){
            
        // genearte the three needed random numbers
        xi = ranluxcl(&ranluxclstate);

        // draw the walker randomly
        j = (int) (xi.s0 * K_OVER_TWO); 

        // draw a sample from the g(Z) distribution
        z = A_COEFF_2 * xi.s1*xi.s1 + A_COEFF_1 * xi.s1 + A_COEFF_0;
        
        // compute the proposal 
        for(int i=0; i<NN; i++)
            Y[i + start_idx] = X_fixed[i + j*NN] + z * (X_moving[i + k*NN] - X_fixed[i + j*NN]); 

        // evaluate the likelihood function 
        #ifdef USE_LOCAL_DATA
            log_py  = log_pdf(Y + start_idx, data_st, data_local);
        #else
            log_py  = log_pdf(Y + start_idx, data_st, data);
        #endif 
        
        // always reject an inf sample 
        if(isinf(log_py)){ 
            q = 0.0f;              
        }
        else{
            log_pxk = log_prob_moving[k];  
            q   = pow(z, NN-1) * exp( data_st->beta * (log_py - log_pxk)) ; 
        }
         
        // accept and update
        if(xi.s2 <= q){
            accepted[k]++ ;  
            for(int i=0; i<NN; i++)
                X_moving[i + k*NN] = Y[i + start_idx];
            log_prob_moving[k] = log_py; 
        }
    }
    
    //Upload state again so that we don't get the same
    //numbers over again the next time we use ranluxcl.
    ranluxcl_upload_seed(&ranluxclstate, ranluxcltab);
    
    // update the components that need to be saved 
    if(data_st->save){
        int start_idx = k * data_st->num_to_save; 
        for(int i=0; i < (data_st->num_to_save); i++) 
            X_save[i + start_idx] = X_moving[ indices_to_save[i] + k*NN]; 
    }
 
}
