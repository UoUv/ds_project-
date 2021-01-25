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
 * input  i