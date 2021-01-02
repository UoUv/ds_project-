#ifndef _DEF_WINOGRADCONV_H_
#define _DEF_WINOGRADCONV_H_
#include "simd.h"
#include "stdlib.h"
#include <string.h>
#include <malloc.h>

//#define Ni 64 
//#define Ri 32 
//#define Ci 32 
//#define No 64  //hard coded for now
//#define K 4 
//#define Ro (Ri-K+1)
//#define Co (Ci-K+1)
////#define vB 4
//#define B 32 

typedef struct InputData_st{
  void* input; //0
  void* transInput; //8
  int Ni, B, Ri, Ci;
} InputData;

typedef struct OutputData_st{
  void* output; //0
  void* transOutput; //8
  int No, B, Ro, Co;
} OutputData;

typedef struct FilterData_st{
  void* filter;
  void* transFilter;
  int Ni, No;
} FilterData;

#define MAX_BATCH           64
#define MAX_IMAGE_CHANNELS  64
#define M