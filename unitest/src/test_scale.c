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
    float * top_data=(float*)malloc(sizeof(float)*128*512*56*56);
    float * my_top_data=(float*)malloc(sizeof(float)*128*512*56*56);

    char out[20]="0 0 0 time";

    athread_init();
    printf("start\n");
    for(ii=0;ii<3;++ii)
    {
      for(jj=0;jj<5;++jj)
      {
        for(kk=0;kk<4;++kk)
        {
            //if(k==1) break;
            printf("i=%d j=%d k=%d\n",ii,jj,kk);
            num=a[ii];channels_=b[jj];width_=height_=cc[kk];
            if(ii==2&&jj==4&&kk==3)
            {
              num=1;channels_=128*512;
            }
            spatial_dim=width_*height_;
            blob_size=num*channels_*width_*height_;
            //with origin data
            //printf("here 1\n");
            //init
            for(z = 0; z < blob_size; ++z ) bbottom_data[z] = rand()/(float)RAND_MAX;
            for(z = 0; z < blob_size; ++z ) sscale_data[z] = rand()/(float)RAND_MAX;

            out[0]='0'+ii;
            out[2]='0'+jj;
            out[4]='0'+kk;


            int mode = 0;
            //mode==0 : max
  