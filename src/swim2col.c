#include <assert.h>
#include <athread.h>
#include "./include/swim2col.h"

#define LDM_MAX (64*1024)

extern void SLAVE_FUN(sw_im2col_large_stride_f)();
extern void SLAVE_FUN(sw_im2col_large_stride_zeropad_f)();
extern void SLAVE_FUN(sw_im2col_large_stride_zeropad_batch_f)();
extern void SLAVE_FUN(sw_im2col_large_stride_zeropad_batch_trans_f)();
extern void SLAVE_FUN(sw_im2