#include "include/swscale.h"
#include "include/swcommon.h"
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
int test_scale() 
{
    int a[3],b[5],cc[4];
    a[0]=32;a[1]=64;a[2]=128;
    b[0]=3;b[1]=64;b[2]=128;b[3]=256;b[4]=512;
    cc[0]=6;cc[1]=14;cc[2]=28;cc[3]=56;
    int num,cha