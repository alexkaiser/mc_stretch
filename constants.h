
// Place #define constants here.
// These values are included in main and pdf.h



// some general constants


// If true this will make initial components nonnegative
// This is generically fine
#define NONNEGATIVE_BOX 1


// Sometimes AMD requires a different include path to compile random number generator
// Set this to one to add it.
#define AMD 0


// Set the amount of output
// OUTPUT_LEVEL 0: Only fatal error messages
// OUTPUT_LEVEL 1: Print progress updates
// OUTPUT_LEVEL 2: Print information about selected device
#define OUTPUT_LEVEL 2


// SDE solve specific constants
#define NX              1
#define NY              1
#define N_TH            4
#define N_STEPS         40
#define OBS_FREQ        1
#define N_OBS           40
#define DYNAMICS_TYPE   1
#define H          0.1000000000f
#define H_SQRT     0.3162277660f
#define A_MIN          -10.0f
#define A_MAX         10.0f
#define SIG_MIN       0.0f
#define SIG_MAX       10.0f
#define B_MIN        -10.0f
#define B_MAX         10.0f
#define ZETA_MIN      0.0f
#define ZETA_MAX      10.0f
#define X_MIN         -200.0f
#define X_MAX          200.0f
#define A_TRUE_0  2.000000
#define SIG_TRUE_0  1.000000
#define B_TRUE_0  -2.000000
#define ZETA_TRUE_0  1.000000


