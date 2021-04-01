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
#i