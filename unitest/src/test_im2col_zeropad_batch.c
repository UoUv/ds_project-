
#include "include/swim2col.h"
#include "include/swcommon.h"
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "swblas.h"


void test_im2col_zeropad_batch_swblas_float(int channels, int filters, int height, int width, int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w, int batch_size) {
  printf("begin test_im2col_zeropad_batch_swblas_float\n");
  int i, j, k;
#define Type float
  struct timeval t1, t2;

  int dilation_h, dilation_w, output_w, output_h;
  dilation_h = 1;
  dilation_w = 1;
  output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  output_w = (width  + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  int im2col_batch_size = 1;
  for(i = 1; i <= batch_size; i*=2) {
    if(batch_size%i == 0 && i*((width + 2*pad_w) + output_w)*sizeof(float) < 60*1024)
      im2col_batch_size = i;
  }
  //im2col_batch_size = 1;
  int im_size = channels*height*width;
  int col_size = output_w*output_h*channels*kernel_h*kernel_w;
  int zeropad_col_rowsize = (output_w * output_h + 127)/128*128;
  int zeropad_col_colsize = (kernel_h * kernel_w * channels + 7)/8*8;
  int pad_col_size = zeropad_col_rowsize * zeropad_col_colsize;
  int group_ = 1;

  printf("forward: channels %d, filters %d, height %d, width %d, kernel_h %d, kernel_w %d, \
      pad_h %d, pad_w %d, output_w %d, output_h %d, stride_h %d, stride_w %d, zeropad_col_rowsize %d (%d)\
      zeropad_col_colsize %d (%d), im2col_batch_size %d\n",
          channels, filters, height, width, \
          kernel_h,kernel_w,pad_h,pad_w,output_w,output_h,stride_h,stride_w, zeropad_col_rowsize, output_w*output_h,zeropad_col_colsize,
          kernel_h*kernel_w*channels,
          im2col_batch_size);

  //allocate memory
  Type* data_im = (Type*)malloc(sizeof(Type)*im_size*batch_size);
  for(i = 0; i < im_size*batch_size; ++i )
    data_im[i] = rand()/(Type)RAND_MAX;
#ifdef _MEM_ALIGN_
  float* data_col = (float*)_aligned_malloc(sizeof(Type)*col_size*batch_size, 128);
#elif
  float* data_col = (float*)malloc(sizeof(Type)*col_size*batch_size);
#endif

  if(!data_col)
    printf("allocate data_col failed!\n");
  memset(data_col,0.0, sizeof(Type)*col_size*batch_size);

  Type* zero_pad_data_col = (Type*)malloc(sizeof(Type)*pad_col_size*batch_size);
  if(!zero_pad_data_col)
    printf("allocate zero_pad_data_col failed!\n");
  memset(zero_pad_data_col, 0.0, sizeof(Type)*pad_col_size*batch_size);



  //params for GEMM
  int N = filters;
  int M = zeropad_col_rowsize;
  int K = zeropad_col_colsize;

  int blkK = 0;
  int blkM = 0;
  int blkN = 0;
  int cK, cM, cN;
  for(cK = 8; cK <= K && cK < 512; cK += 8)
    for(cM = 128; cM <= M; cM += 128) {
      for(cN = 32; cN <= N; cN += 32) {
        if(N%cN == 0 && K%cK == 0 && M%cM == 0 && (2*cK*cM + 2*cK*cN + cM*cN)*sizeof(double) < 56*1024*64) {
          blkM = cM;
          blkK = cK;
          blkN = cN;
        }
    }
  }