
#include "slave.h"
#include "simd.h"
#include "dma.h"
#include "include/swim2col.h"
#define LDM_MAX (64*1024)
// Precondition: no dilations. float data type
void sw_col2im_large_stride_f(Im2colPara *para) {
  dma_desc dma_put_im, dma_get_col;
#define Type float
#define SIMDType floatv4
#define SIMDSIZE 4
  int pad_h = para->pad_h;
  int pad_w = para->pad_w;
  int height= para->height;
  int width = para->width;
  int kernel_h = para->kernel_h;
  int kernel_w = para->kernel_w;
  int stride_h = para->stride_h;
  int stride_w = para->stride_w;
  int output_h = (height + 2 * pad_h - kernel_h)/stride_h + 1; // output height with stride
  int output_w = (width + 2 * pad_w - kernel_w)/stride_w + 1;  // output width with stride
  int channel_size = height*width;
  int out_channel_size = output_h*output_w*kernel_w*kernel_h;
  int id = athread_get_id(-1);
  // number of rows of <id> slave core.
  int local_row_size = (2*para->pad_h + para->height) * para->channels / 64
               + (id< ((2*para->pad_h + para->height) * para->channels % 64));