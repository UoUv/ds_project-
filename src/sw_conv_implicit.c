
/***
 * by Jerry Fang
 * fang_jiarui@163.com
 * For the benefit of the nation,, life and death
 ***/
#include <stdio.h>
#include <assert.h>
#include "athread.h"
#include <math.h>
#include "./include/sw_conv_implicit.h"
#include "./include/swtensortrans.h"

extern SLAVE_FUN(conv_valid)();
extern SLAVE_FUN(conv_full)();
extern SLAVE_FUN(conv_pad)();
extern SLAVE_FUN(conv_pad_float)();
extern SLAVE_FUN(conv_pad_float__)();
extern SLAVE_FUN(conv_full_pad)();
//extern SLAVE_FUN(conv_full_pad_float)();
extern SLAVE_FUN(conv_full_pad_float_v2)();

//#ifdef SW_TRANS
//#undef SW_TRANS
//#endif
//#define MPE_TRANS

// high -> low
// B, N, R, C
inline int image_caffe_offset(int n, int c, int h, int w, int N, int C, int H, int W) {
  return (((n*C + c)*H + h)*W + w);
}
// R, C, N, B
inline int image_swdnn_offset(int n, int c, int h, int w, int N, int C, int H, int W) {
  return (((h*W + w)*C + c)*N + n);
}
// R, C, B, N
inline int image_swdnn_offset_back(int n, int c, int h, int w, int N, int C, int H, int W) {
  return (((h*W + w)*N + n)*C + c);
}
// No, Ni, Kr, Kc
inline int weight_caffe_offset(int no, int ni, int kr, int kc, int No, int Ni, int K) {
  return (( no*Ni + ni )*K + kr)*K + kc;
}
// Kr, Kc, No, Ni
inline int weight_swdnn_offset(int no, int ni, int kr, int kc, int No, int Ni, int K) {
  return ((( kr*K + kc )*No + no) * Ni + ni );
}
// Kr, Kc, Ni, No
inline int weight_swdnn_offset_back(int no, int ni, int kr, int kc, int No, int Ni, int K) {
  return ((( kr*K + kc )*Ni + ni) * No + no );
}

//#define weight_swdnn_to_caffe(in,out,B,N,H,W) swapBN_HW(in,out,H,W,B,N)
//#define weight_caffe_to_swdnn(in,out,B,N,H,W) swapBN_HW(in,out,B,N,H,W)
//#define image_caffe_to_swdnn_back(in,out,B,N,H,W)  swapBN_HW(in,out,B,N,H,W)
static int init_flag = 0; 
//-----------------------------------
void sw_conv_forward_pad_impl_f_ori(
        const float* in,
        const float* weight,
        float* out,
        int Ci,
        int Ri,
        int K,
        int Ni,
        int No,
        int B,
        int pad)
{
#ifdef DEBUG_VERBOSE_SWDNN
    printf("forward : before swDNN conv float");
#endif
    int i;
    int cKr, cKc, cNo;
    int cRo, cCo, cB;
    int cRi, cCi, cNi;
    int Ro = Ri+2*pad-K+1 , Co = Ci+2*pad-K+1;
    float* my_in      = (float*)malloc(sizeof(float)*Ri*Ci*Ni*B);
    float* my_out     = (float*)malloc(sizeof(float)*Ro*Co*No*B);
    float* my_weight  = (float*)malloc(sizeof(float)*K*K*No*Ni);

#ifdef MPE_TRANS

#ifdef DEBUG_VERBOSE_SWDNN
    printf("in_trans before");
#endif
    for(cRi = 0; cRi < Ri; ++cRi)