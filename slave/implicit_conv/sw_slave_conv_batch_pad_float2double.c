/***************
 * GEMM PLAN 
 * Jerry Fang 
 * 2017 June 15 
 *
 * input  is of dim(B, Ni)
 * weight is of dim(Ni, No)
 * ouput  is of dim(B, No)
 *
 * No overlap input DMA and weight DMA
 * ************/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <slave.h>
#include <math.h>
#include "simd.h"
#include "dma.h"
#include "./include/sw_conv_implicit.h"

#define SIMDSIZE  4
#define SIMDType  floatv4
#define Type      float
#define SIMDType