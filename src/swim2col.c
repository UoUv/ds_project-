#include <assert.h>
#include <athread.h>
#include "./include/swim2col.h"

#define LDM_MAX (64*1024)

extern void SLAVE_FUN(sw_im2col_large_stride_f)();
extern void SLAVE_FUN(sw_im2col_large_stride_zeropad_f)();
extern void SLAVE_FUN(sw_im2col_large_stride_zeropad_batch_f)();
extern void SLAVE_FUN(sw_im2col_large_stride_zeropad_batch_trans_f)();
extern void SLAVE_FUN(sw_im2col_large_stride_d)();
extern void SLAVE_FUN(sw_im2col_large_stride_d)();
extern void SLAVE_FUN(sw_col2im_large_stride_f)();
extern void SLAVE_FUN(sw_im2col_large_d)();
extern void SLAVE_FUN(sw_im2col_large_f)();
extern void SLAVE_FUN(sw_col2im_large_d)();
extern void SLAVE_FUN(sw_col2im_large_f)();


// float version
// TODO
void swim2col_zeropad_batch_trans_f(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    float* data_col, int batch_size) {
  Im2colPara* para = (Im2colPara*)malloc(sizeof(Im2colPara));
  para->data_im = data_im;
  para->data_col= data_col;
  para->channels= channels;
  para->height  = height;
  para->width   = width;
  para->kernel_h= kernel_h;
  para->kernel_w= kernel_w;
  para->pad_h   = pad_h;
  para->pad_w   = pad_w;
  para->stride_h= stride_h;
  para->stride_w= stride_w;
  para->dilation_h = dilation_h;
  para->dilation_w = dilation_w;
  int output_h = (height + 2 * pad_h 