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


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cl-helper.h"


// General utilities


void read_arrays(cl_float *obs, cl_long N_obs, cl_long n_y, char *file_name){
    /*
     Read array of float data from file.
     Data file should be n_y columns and N_obs rows.

     Input:
         cl_float *obs          Preallocated array for observations.
         cl_long N_obs              Number of observations.
         cl_long n_y                Length of each observation.
         char *file_name        File to read from.

     Output
         cl_float *obs          Array is filled with data from file.
     */
    
    FILE *f = fopen(file_name, "r");
                
    for (cl_long j=0; j < N_obs; j++)
        for(cl_long i=0; i < n_y; i++)
            fscanf(f, "%f ", obs + i + j*n_y);

    fclose(f);
}


void output_array_to_matlab(cl_float *data, cl_long m, cl_long n, char *file_name){
    /*
    Output full data array to matlab file.
    Array is column major, printed to matlab in matrix order (not linear order)

    Input:
        cl_float *data        Data to output.
        cl_long m                 Number of rows.
        cl_long n                 Number of columns.
        char *file_name       File name.
    */

    FILE *f = fopen(file_name, "w");

    for(cl_long i=0; i<m; i++){
        for(cl_long j=0; j<n; j++){
            fprintf(f, "%f ", data[i + j*m]) ;
        }
        fprintf(f, " \n") ;
    }
    fclose(f);
}


void write_parameter_file_matlab(cl_long M, cl_long N, cl_long K, char *sampler_name, cl_long *indices_to_save, cl_long num_to_save, cl_long pdf_num){
    /*
     Write a parameter summary to a Matlab file for reading.
     Includes basic parameters of the sampler to include in plots.

     Input:
         cl_long M                    Length of chain
         cl_long N                    Dimension
         cl_long K                    Number of walkers
         char *sampler_name       Name of sampler (for title)
         cl_long num_to_save          Number of components
         cl_long pdf_num              PDF number, if zero will assume it is Gaussian debug problem and put extra information

     Output:
          File "load_parameters.m" is written in matlab format.
     */

    FILE *f = fopen("load_parameters.m", "w");
    fprintf(f, "M = %lld;\n", M);
    fprintf(f, "N = %lld;\n", N);
    fprintf(f, "K = %lld;\n", K);
    fprintf(f, "name = '%s';\n", sampler_name);
    fprintf(f, "pdf_num = %lld;\n", pdf_num);
    fprintf(f, "indicesToSave = [");
    for(cl_long j=0; j<num_to_save-1; j++)
        fprintf(f, "%lld; ", indices_to_save[j]+1);                // add one to index for matlab
    fprintf(f, "%lld];\n\n", indices_to_save[num_to_save-1]+1);
    fclose(f);
}


void histogram_data(cl_long n_bins, float *samples, cl_long n_samples, double tau, float *centers, float *f_hat){
    /*
     Compute a histogram from one dimensional data.
     Centers are computed dynamically to include all the data.

     Input:
         cl_long n_bins              Number of bins for the histogram
         float *samples          Samples to plot
         cl_long n_samples           Number of samples
         double tau              Autocorrelation time, for making error bars
         float *centers          Preallocated to length n_bins
         float *f_hat            Preallocated to length n_bins

     Output:
         float *centers          Bin centers
         float *f_hat            Estimate f_hat of pdf computed from data
     */

    double x_min = (double) samples[0];
    double x_max = (double) samples[0];

    for(cl_long k=0; k<n_samples; k++){
        if (samples[k] < x_min)
            x_min = (double) samples[k];
            
        if (samples[k] > x_max)
            x_max = (double) samples[k];
    }
    
    
    float dx = (x_max - x_min) / (double) n_bins;
    
    unsigned long *bin_counts = (unsigned long *) malloc(n_bins * sizeof(unsigned long)); 
    if(!bin_counts) { perror("Alloc host: histogram bins. "); abort(); }

    cl_long idx; 

    cl_long lost = 0; 

    for(cl_long k=0; k<n_bins; k++){
        bin_counts[k] = 0; // start the counters at zero 
        centers[k] = (float) 0.5*dx + k*dx + x_min;    // initialize the bin centers
    }

    for(cl_long k=0; k<n_samples; k++){
        // compute the bin number 

        idx = (int) ( (samples[k] - x_min)/dx );
        
        // include the last bdry too
        if(samples[k] == x_max)
            idx = n_bins - 1;

        if((idx < 0) || (idx>=n_bins)){
            lost++;                            // we're off to the side, this is bad
            printf("sample lost: %f, idx = %lld\n", samples[k], idx); 
        }
        else{
            bin_counts[idx]++;                 // increment the bin count 
        }
    }

    // turn the bins into estimates 
    // estimate sigma for the bin count as well
    // double effective_samples = ((double) n_samples) / tau;

    float pk;
    for(cl_long k=0; k<n_bins; k++){
        pk = ((float) bin_counts[k]) / (float) (n_samples);
        f_hat[k] = pk / (float) (dx);
    }
        
    if(lost > 1)
        fprintf(stderr, "Warning. Histogram lost %lld samples. Consider expanding histogram bounds.\n", lost); 
        
    free(bin_counts);     
}


void histogram_to_matlab(cl_long n_bins, float *centers, float *f_hat, cl_long var_number){
    /*
     Output a matlab file for histograms.
     Data must be precomputed.

     Input:
         cl_long n_bins              Number of bins.
         float *centers          Bin center.
         float *f_hat            Bin estimate.
         cl_long var_number          Component number for file title.

     Output:
         File "histogram_data_i.m" is written, where 'i' is variable number.
     */

    char title[100]; 
    sprintf(title, "histogram_data_%lld.m", var_number);     

    FILE *f = fopen(title, "w");

    fprintf(f, "centers = ["); 
    for(cl_long i=0; i<n_bins; i++){
        fprintf(f, "%f ", centers[i]); 
    }
    fprintf(f, "];\n\n"); 

    fprintf(f, "fhat = ["); 
    for(cl_long i=0; i<n_bins; i++){
        fprintf(f, "%f ", f_hat[i]); 
    }
    fprintf(f, "];\n"); 
    
    fclose(f); 
}


void histogram_to_gnuplot(cl_long n_bins, float *centers, float *f_hat, cl_long var_number){
    /*
     Output a gnuplot file for histograms.
     Data must be precomputed.

     Input:
         cl_long n_bins              Number of bins.
         float *centers          Bin center.
         float *f_hat            Bin estimate.
         cl_long var_number          Component number for file title.

     Output:
         File "histogram_data_gnuplot_i.m" is written, where 'i' is variable number.
     */

    char title[100]; 
    sprintf(title, "histogram_data_gnuplot_%lld.dat", var_number);

    FILE *f = fopen(title, "w");

    for(cl_long i=0; i<n_bins; i++){
        fprintf(f, "%f  %f \n", centers[i], f_hat[i]); 
    }

    fclose(f); 
}

void compute_mean_stddev(float *X, double *mean, double *sigma, cl_long total_samples){
    /*
     Compute mean and standard deviation of a one dimensional array.

     Input:
         float *X               Input array.
         double *mean           Mean, overwritten
         double *sigma          Standard deviation, overwritten
         cl_long total_samples      Length of timeseries

     Output:
         double *mean           Mean
         double *sigma          Standard deviation
     */

    *mean = 0.0;
    for(cl_long i=0; i<total_samples; i++)
        *mean += (double) X[i];
    *mean /= ((double) total_samples);

    *sigma = 0.0;
    for(cl_long i=0; i<total_samples; i++)
        *sigma += (double) (X[i] - *mean) * (X[i] - *mean);
    *sigma /= ((double) total_samples);
    *sigma = sqrt(*sigma) ;
}




/* acor module */ 

/*  The code that does the acor analysis of the time series.  See the README file for details.  */
#define TAUMAX  10                /*   Compute tau directly only if tau < TAUMAX.
                                       Otherwise compute tau using the pairwise sum series          */
#define WINMULT 5                 /*   Compute autocovariances up to lag s = WINMULT*TAU            */
#define MAXLAG  TAUMAX*WINMULT    /*   The autocovariance array is double C[MAXLAG+1] so that C[s]
                                       makes sense for s = MAXLAG.                                  */
#define MINFAC   5                /*   Stop and print an error message if the array is shorter
                                       than MINFAC * MAXLAG.                                        */


/*  Jonathan Goodman, March 2009, goodman@cims.nyu.edu  */

// Ported to C by Alex Kaiser, 12/2012


cl_long acor( double *mean, double *sigma, double *tau, double *X, cl_long L){

   cl_long pass = 1;
   
   *mean = 0.;                                   // Compute the mean of X ... 
   for ( cl_long i = 0; i < L; i++) *mean += X[i];
   *mean = *mean / L;
   for ( cl_long i = 0; i <  L; i++ ) X[i] -= *mean;    //  ... and subtract it away.
   
   if ( L < MINFAC*MAXLAG ) {
      fprintf(stderr, "Acor error 1: The autocorrelation time is too long relative to the variance.\n"); 
      return 0; }
   
   double C[MAXLAG+1];
   for ( cl_long s = 0; s <= MAXLAG; s++ )  C[s] = 0.;  // Here, s=0 is the variance, s = MAXLAG is the last one computed.
     
   cl_long iMax = L - MAXLAG;                                 // Compute the autocovariance function . . . 
   for ( cl_long i = 0; i < iMax; i++ ) 
      for ( cl_long s = 0; s <= MAXLAG; s++ )
         C[s] += X[i]*X[i+s];                              // ...  first the inner products ...
   for ( cl_long s = 0; s <= MAXLAG; s++ ) C[s] = C[s]/iMax;   // ...  then the normalization.
      
   double D = C[0];   // The "diffusion coefficient" is the sum of the autocovariances
   for ( cl_long s = 1; s <= MAXLAG; s++ ) D += 2*C[s];   // The rest of the C[s] are double counted since C[-s] = C[s].
   *sigma = sqrt( D / L );                            // The standard error bar formula, if D were the complete sum.
   *tau   = D / C[0];                                 // A provisional estimate, since D is only part of the complete sum.
   
   if ( *tau*WINMULT < MAXLAG ) return pass;             // Stop if the D sum includes the given multiple of tau.
                                                      // This is the self consistent window approach.
                                                      
   else {                                             // If the provisional tau is so large that we don't think tau
                                                      // is accurate, apply the acor procedure to the pairwase sums
                                                      // of X.
      cl_long Lh = L/2;                                   // The pairwise sequence is half the length (if L is even)
      double newMean;                                 // The mean of the new sequence, to throw away.
      cl_long j1 = 0;
      cl_long j2 = 1;
      for ( cl_long i = 0; i < Lh; i++ ) {
         X[i] = X[j1] + X[j2];
         j1  += 2;
         j2  += 2; }
      pass &= acor( &newMean, sigma, tau, X, Lh);
      D      = .25*(*sigma) * (*sigma) * L;    // Reconstruct the fine time series numbers from the coarse series numbers.
      *tau   = D/C[0];                         // As before, but with a corrected D.
      *sigma = sqrt( D/L );                    // As before, again.
    }
      
     
   return pass;
  }
