#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <slave.h>
#include <math.h>
#include "simd.h"
#include "dma.h"
#include "../include/sw_conv_implicit.h"

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
#define SIMDSIZE 4
#define SIMDType floatv4
#define Type float

void conv_full_pad_float(ConvData* param)
{
  int cB, cNi, cRi, cCi, cKr, cKc, ccCore, crCore, cNo;
  int ii, jj, cRo, cCo