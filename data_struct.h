


typedef struct{

    cl_float beta;         // always include these
    cl_int num_to_save;
    cl_int save;

    // Add any other scalars here.
    // Small arrays of static length are also okay.
    // Pointers will cause errors, following OpenCL rules.

    #ifndef NX
        #define NX 1
    #endif

    // SDE solve specific extra array
    // this is a single scalar if NX is not defined prior to including this header
    cl_float x_initial[NX];

}data_struct;
