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
    const int stride_h, c