/********************************************
 * Created by Xin You
 * Date: 2017/8/24
 * softmax layer interface for acc version.
 * *****************************************/
#include <stdio.h>
#include <assert.h>
#include <athread.h>
#include <math.h>
#include "include/swsoftmax.h"

//#define DEBUG_INFO
//#define MPE_TRANS

extern SLAVE_FUN(swsoftmax_trans_f)();
extern SLAVE_FUN(swsofmax_f)();
extern SLAVE_FUN(softmaxBackward)();
//extern SLAVE_FUN(swsofmax_d)();

typedef struct TransData_st {
  void* in;
  void* out;
  int tZ;
  int tX;
  int tY;
}TransData;

typedef struct SoftmaxData_st{
  void* bottom_data;
  void* sum_multiplier_;
  void* scale_data;
  void* top_data;
  int channels;
  int dim;
  int outer_num_;
  int inner_num_;
}SoftmaxData;

void sw_softmax_forward_impl_f(
    const float* bottom_data,
    const float* sum_multiplier_,
    float* scale_data,
    float* top_data,
    int channels,
    int dim,
    int outer_num_,
    int inner_num_) {
#ifdef DEBUG_INFO
  printf("channels = %d, dim = %d, outer_num_ = %d, inner_num_ = %d\n",channels,dim,outer_num_,inner_num_);
  /*int testArr[80];
  int testArr__[80];
  int tt, ii,ij,ik;
  for(tt = 0;tt<80;tt++) testArr[tt] = tt;
  TransData tdata;
  tdata.in=testArr;
  tdata.out=testArr__;
  tdata.tX = 4;
  tdata.tY = 5;
  tdata.tZ = 4;
  athread_spawn