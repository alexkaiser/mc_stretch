


typedef struct{

    cl_float beta;         // always include these
    cl_int num_to_save;
    cl_int save;

    // Add any other scalars here.
    // Small arrays of static length are also okay.
    // Pointers will cause errors, following OpenCL rules.

}data_struct;
