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
    int num,channels_,width_,height_;
    int spatial_dim;
    int i,j,k,z,ph,pw,n,c,h,w;
    int blob_size;
    int ii,jj,kk;
    float eps_=1e-4;
    float factor;


    float * bbottom_data=(float*)malloc(sizeof(float)*128*512*56*56);
    float * sscale_data=(float*)malloc(sizeof(float)*128*512*56*56);

    //without origin data
    float * top_data=(float*)malloc(