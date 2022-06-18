
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
        int sizeI = Ri*Ci;
        for(i = 0; i < Ri-2; i += 2){
            #pragma unroll(4)
            for(j = 0; j < Ci-2; j += 2){
                //tmp[0 :4] =data[(i+0)*ldi+j:4]; 
                //tmp[4 :4] =data[(i+1)*ldi+j:4]; 
                //tmp[8 :4] =data[(i+2)*ldi+j:4]; 
                //tmp[12:4] =data[(i+3)*ldi+j:4]; 
                tmp[0] = image[cB*sizeI*C + i*Ci*C + j*C + cNi]; 
                tmp[1] = image[cB*sizeI*C + i*Ci*C + (j+1)*C + cNi]; 
                tmp[2] = image[cB*sizeI*C + i*Ci*C + (j+2)*C + cNi]; 
                tmp[3] = image[cB*sizeI*C + i*Ci*C + (j+3)*C + cNi]; 
                tmp[4] = image[cB*sizeI*C + (i+1)*Ci*C + j*C + cNi]; 
                tmp[5] = image[cB*sizeI*C + (i+1)*Ci*C + (j+1)*C + cNi]; 
                tmp[6] = image[cB*sizeI*C + (i+1)*Ci*C + (j+2)*C + cNi]; 
                tmp[7] = image[cB*sizeI*C + (i+1)*Ci*C + (j+3)*C + cNi]; 
                tmp[8] = image[cB*sizeI*C + (i+2)*Ci*C + j*C + cNi]; 
                tmp[9] = image[cB*sizeI*C + (i+2)*Ci*C + (j+1)*C + cNi]; 
                tmp[10] = image[cB*sizeI*C + (i+2)*Ci*C + (j+2)*C + cNi]; 
                tmp[11] = image[cB*sizeI*C + (i+2)*Ci*C + (j+3)*C + cNi]; 
                tmp[12] = image[cB*sizeI*C + (i+3)*Ci*C + j*C + cNi]; 
                tmp[13] = image[cB*sizeI*C + (i+3)*Ci*C + (j+1)*C + cNi]; 
                tmp[14] = image[cB*sizeI*C + (i+3)*Ci*C + (j+2)*C + cNi]; 
                tmp[15] = image[cB*sizeI*C + (i+3)*Ci*C + (j+3)*C + cNi]; 

                // The tranformation manually simplified
                s[0 ] =(tmp[0] - tmp[8 ]) - (tmp[2 ]- tmp[10]);   
                s[1 ] =(tmp[1] - tmp[9 ]) + (tmp[2 ]- tmp[10]); 
                s[2 ] =(tmp[2] - tmp[10]) - (tmp[1 ]- tmp[9 ]); 
                s[3 ] =(tmp[1] - tmp[9 ]) - (tmp[3 ]- tmp[11]); 
                s[4 ] =(tmp[4] + tmp[8 ]) - (tmp[6 ]+ tmp[10]); 
                s[5 ] =(tmp[5] + tmp[9 ]) + (tmp[6 ]+ tmp[10]); 
                s[6 ] =(tmp[6] + tmp[10]) - (tmp[5 ]+ tmp[9 ]); 
                s[7 ] =(tmp[5] + tmp[9 ]) - (tmp[7 ]+ tmp[11]); 
                s[8 ] =(tmp[8] - tmp[4 ]) - (tmp[10]- tmp[6 ]); 
                s[9 ] =(tmp[9] - tmp[5 ]) + (tmp[10]- tmp[6 ]); 
                s[10] =(tmp[10]- tmp[6 ]) - (tmp[9 ]- tmp[5 ]); 
                s[11] =(tmp[9] - tmp[5 ]) - (tmp[11]- tmp[7 ]); 
                s[12] =(tmp[4] - tmp[12]) - (tmp[6 ]- tmp[14]); 
                s[13] =(tmp[5] - tmp[13]) + (tmp[6 ]- tmp[14]); 
                s[14] =(tmp[6] - tmp[14]) - (tmp[5 ]- tmp[13]); 
                s[15] =(tmp[5] - tmp[13]) - (tmp[7 ]- tmp[15]); 

                // manually unrolled scatter to get max performance
                otile[tile_count+0*stride] = s[0 ]; 
                otile[tile_count+1*stride] = s[1 ]; 
                otile[tile_count+2*stride] = s[2 ]; 
                otile[tile_count+3*stride] = s[3 ]; 
                otile[tile_count+4*stride] = s[4 ]; 
                otile[tile_count+5*stride] = s[5 ]; 
                otile[tile_count+6*stride] = s[6 ]; 
                otile[tile_count+7*stride] = s[7 ]; 
                otile[tile_count+8*stride] = s[8 ]; 
                otile[tile_count+9*stride] = s[9 ]; 
                otile[tile_count+10*stride] = s[10]; 
                otile[tile_count+11*stride] = s[11]; 
                otile[tile_count+12*stride] = s[12]; 
                otile[tile_count+13*stride] = s[13]; 
                otile[tile_count+14*stride] = s[14]; 
                otile[tile_count+15*stride] = s[15]; 

                tile_count += C;
            }
        }
    }
}

// INTERNAL FUNCTION : FORM MATRIX A from input data, also includes transformation
// ldi == irows
// irows = irows
// sizeI = irows * icols
// C == input channel
// otile == results
// N == batch size
// ntiles == #tiles
// image (B, N, H, W) ?
static void get_tiles(const float* image, const int ldi, const int irows, const int sizeI, const int C, float* otile, const int N, const int ntiles){
    int t, u;
    #pragma omp parallel for 
    for(t = 0; t < N*C; t++){
        int i, j, x; 
        float tmp[16] __attribute__((aligned(64))); 
        float s[16] __attribute__((aligned(64))); 

        const float* data = image+t*sizeI; 
        int tile_count = t*ntiles; 
        // work on one image plane at a time, irrespective of the order
        for(i = 0; i < irows-2; i += 2){
            #pragma unroll(4)            
            for(j = 0; j < (irows-2); j += 2){
                //tmp[0 :4] =data[(i+0)*ldi+j:4]; 
                //tmp[4 :4] =data[(i+1)*ldi+j:4]; 
                //tmp[8 :4] =data[(i+2)*ldi+j:4]; 
                //tmp[12:4] =data[(i+3)*ldi+j:4]; 
                tmp[0] =data[(i+0)*ldi+j];
                tmp[1] =data[(i+0)*ldi+j+1];
                tmp[2] =data[(i+0)*ldi+j+2];
                tmp[3] =data[(i+0)*ldi+j+3];

                tmp[4] =data[(i+1)*ldi+j];
                tmp[5] =data[(i+1)*ldi+j+1];
                tmp[6] =data[(i+1)*ldi+j+2];
                tmp[7] =data[(i+1)*ldi+j+3];

                tmp[8] =data[(i+2)*ldi+j];
                tmp[9] =data[(i+2)*ldi+j+1];
                tmp[10] =data[(i+2)*ldi+j+2];
                tmp[11] =data[(i+2)*ldi+j+3];

                tmp[12] =data[(i+3)*ldi+j];
                tmp[13] =data[(i+3)*ldi+j+1];
                tmp[14] =data[(i+3)*ldi+j+2];
                tmp[15] =data[(i+3)*ldi+j+3];

                // The tranformation manually simplified
                s[0 ] =(tmp[0] - tmp[8 ]) - (tmp[2 ]- tmp[10]);
                s[1 ] =(tmp[1] - tmp[9 ]) + (tmp[2 ]- tmp[10]);
                s[2 ] =(tmp[2] - tmp[10]) - (tmp[1 ]- tmp[9 ]);
                s[3 ] =(tmp[1] - tmp[9 ]) - (tmp[3 ]- tmp[11]);
                s[4 ] =(tmp[4] + tmp[8 ]) - (tmp[6 ]+ tmp[10]);
                s[5 ] =(tmp[5] + tmp[9 ]) + (tmp[6 ]+ tmp[10]);
                s[6 ] =(tmp[6] + tmp[10]) - (tmp[5 ]+ tmp[9 ]);
                s[7 ] =(tmp[5] + tmp[9 ]) - (tmp[7 ]+ tmp[11]);
                s[8 ] =(tmp[8] - tmp[4 ]) - (tmp[10]- tmp[6 ]);
                s[9 ] =(tmp[9] - tmp[5 ]) + (tmp[10]- tmp[6 ]);
                s[10] =(tmp[10]- tmp[6 ]) - (tmp[9 ]- tmp[5 ]);
                s[11] =(tmp[9] - tmp[5 ]) - (tmp[11]- tmp[7 ]);
                s[12] =(tmp[4] - tmp[12]) - (tmp[6 ]- tmp[14]);
                s[13] =(tmp[5] - tmp[13]) + (tmp[6 ]- tmp[14]);
                s[14] =(tmp[6] - tmp[14]) - (tmp[5 ]- tmp[13]);
                s[15] =(tmp[5] - tmp[13]) - (tmp[7 ]- tmp[15]);

                // manually unrolled scatter to get max performance
                otile[tile_count+0*STRIDE ] = s[0 ];
                otile[tile_count+1*STRIDE ] = s[1 ];
                otile[tile_count+2*STRIDE ] = s[2 ];
                otile[tile_count+3*STRIDE ] = s[3 ];
                otile[tile_count+4*STRIDE ] = s[4 ];
                otile[tile_count+5*STRIDE ] = s[5 ];
                otile[tile_count+6*STRIDE ] = s[6 ];
                otile[tile_count+7*STRIDE ] = s[7 ];
                otile[tile_count+8*STRIDE ] = s[8 ];
                otile[tile_count+9*STRIDE ] = s[9 ];
                otile[tile_count+10*STRIDE] = s[10];
                otile[tile_count+11*STRIDE] = s[11];
                otile[tile_count+12*STRIDE] = s[12];
                otile[tile_count+13*STRIDE] = s[13];
                otile[tile_count+14*STRIDE] = s[14];
                otile[tile_count+15*STRIDE] = s[15];
