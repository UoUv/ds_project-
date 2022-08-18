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
  void* sum_multip