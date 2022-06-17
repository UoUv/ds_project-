
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <assert.h>
#include <athread.h>
#include "include/swwinogradconv.h"
#include "../include/swcommon.h"
#include "swblas.h"
#include <cblas.h>

//extern SLAVE_FUN(FJR_blas_sgemm)();
//extern SLAVE_FUN(FJR_blas_sgemm_smallB)();
extern int SLAVE_FUN(FJR_input_trans)();
extern int SLAVE_FUN(FJR_input_trans_Ni512)();
extern int SLAVE_FUN(FJR_filter_trans)();
extern int SLAVE_FUN(FJR_output_trans)();

const long MAX_TILES = (MAX_IROWS-2)*(MAX_IROWS-2)*0.25; 
// STRIDE is the max image*C*batch for image
//const long STRIDE = MAX_BATCH*(MAX_IMAGE_CHANNELS+18)*(MAX_TILES+13); 
//const long STRIDE = ((MAX_BATCH)*(MAX_IMAGE_CHANNELS+18)*(MAX_TILES+13)); 
#define STRIDE ((MAX_BATCH)*(MAX_IMAGE_CHANNELS+18)*(MAX_TILES+13))
// FSTRIDE is the max C*K for filter
const long FSTRIDE = (MAX_FILTER_CHANNELS+1)*(MAX_FILTERS+1); 

/*
float* t_filter;
float* t_image;
float* c_out;

// setup scratch memory used in the algorithm
void falcon_init_lib(int B, int Ni, int No, int Ci, int Ri){
    int T = (Ci-2)*(Ri-2)/4;
    t_filter = (float*)_aligned_malloc(16*Ni*No*sizeof(float), 128);    
    assert(t_filter != NULL);
    t_image = (float*)_aligned_malloc(16*Ni*T*B*sizeof(float), 128);    
    assert(t_image != NULL);
    c_out = (float*)_aligned_malloc(16*No*T*B*sizeof(float), 128);
    assert(c_out != NULL);
}

// free up the scratch pad
void falcon_free_lib(){
    //free(t_filter);
    //free(t_image);
    //free(c_out);
    _aligned_free(t_filter);
    _aligned_free(t_image);
    _aligned_free(c_out);
}
*/

//input layout (B, Ri, Ci, Ni)
//trans input layout (16, B, T, Ni)
static void fjr_get_tiles(const float* image, float* otile, int N, int C, int ntiles, int Ri, int Ci){
    int t, u;
    int cB, cNi;
    #pragma omp parallel for 
    for(cB = 0; cB < N; ++cB)
      for(cNi = 0; cNi < C; ++cNi) {
        int i, j, x; 
        float tmp[16] __attribute__((aligned(64))); 
        float s[16] __attribute__((aligned(64))); 

        // work on one image plane at a time, irrespective of the order
        int stride = N*C*ntiles;
        int tile_count = cNi + cB*ntiles*C;