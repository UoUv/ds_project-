#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <slave.h>
#include <math.h>
#include "simd.h"
#include "dma.h"
#include "./include/sw_conv_implicit.h"

/***************
 * GEMM PLAN 
 * Jerry Fang 
 * 2017 June 18
 *
 * input  is of dim(B, Ni)
 * weight is of dim(Ni, No)
 * ouput  is of dim(B, No)
 *
 * No overlap input DMA and weight DMA
 * for backward in_grad = conv(out_grad, weight, 'full');
 * pad_inv(out) = conv(in, weight, 'full')
 * ************/
#define Type double
#define SIMDSIZE 4
#define SIMDType doublev4

void conv_full_pad(ConvData* param)
{
  int cB, cNi, cRi, cCi, cKr, cKc, ccCore, crCore, cNo;
  int ii, jj, cRo, cCo;
  int CoStart;
  int id = athread_get_id(-1);
  int cid = id%8, rid = id/8;
  int input_calc_index=1, input_load_index=0;
  int weight_calc_index=1, weight_load_index=0;
  int i, j;
  int Ni, Ri, Ci, No, K, Ro, Co, B, pad;
  Ni = param->_Ni;
  Ri = param->_Ri;
  Ci = param->_Ci;
  No = param->_No;
  K  = param->_K;
  Ro = param->_Ro;
  Co = param->_Co;
  B  = param->_B;
  pad  = param->_pad;
  int CStride=param->_Costride;

//B, Ni, Ci, Ri
  SIMDType* local_input  = (SIMDType*)(long) ldm_malloc(sizeof(