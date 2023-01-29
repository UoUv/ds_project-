#include "include/swsoftmax.h"
#include "include/swcommon.h"
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define Dtype float
int test_softmax()
{
  int a[3],b[5],c[4];
  a[0]=32;a[1]=64;a[2]=128;
  b[0]=3;b[1]=64;b[2]=128;b[3]