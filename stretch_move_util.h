
// header for utilities

void read_arrays(cl_float *obs, int N_obs, int n_y, char *file_name);

void output_array_to_matlab(cl_float *data, int m, int n, char *file_name);
void write_parameter_file_matlab(int M, int N, int K, char *sampler_name, int num_to_save, int pdf_num); 
void histogram_data(int n_bins, float *samples, int n_samples, double tau, float *centers, float *f_hat, float *sigma_f_hat);
void histogram_to_matlab(int n_bins, float *centers, float *f_hat, float *sigma_f_hat, int var_number);
void histogram_to_gnuplot(int n_bins, float *centers, float *f_hat, int var_number); 

void compute_mean_stddev(float *X, double *mean, double *sigma, int total_samples);

int acor( double *mean, double *sigma, double *tau, double *X, int L);


