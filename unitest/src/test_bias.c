
#include "include/swbias.h"
#include "include/swcommon.h"
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#define USE_SWDNN
int test_bias() 
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


    float * bbottom_data=(float*)malloc(sizeof(float)*128*512*56*56);
    float * bbias_data=(float*)malloc(sizeof(float)*512);
    //float * blob_0=(float*)malloc(sizeof(float)*512);
    //float * blob_1=(float*)malloc(sizeof(float)*512);
    //float * blob_2=(float*)malloc(sizeof(float));
    //float * spatial_sum_multiplier=(float*)malloc(sizeof(float)*56*56);
    //float * num_by_chans=(float*)malloc(sizeof(float)*128*512);
    //float * batch_sum_multiplier=(float*)malloc(sizeof(float)*128);

    //without origin data
    float * top_data=(float*)malloc(sizeof(float)*128*512*56*56);
    //float * top_1=(float*)malloc(sizeof(float)*128*512*56*56);
    //int * maxidx=(int*)malloc(sizeof(int)*128*512*56*56);
    //float * variance_origin=(float*)malloc(sizeof(float)*512);
    //float * xnorm=(float*)malloc(sizeof(float)*128*512*56*56);
    //float * temp=(float*)malloc(sizeof(float)*128*512*56*56);
    float * my_top_data=(float*)malloc(sizeof(float)*128*512*56*56);
    //float * my_top_1=(float*)malloc(sizeof(float)*128*512*56*56);
    float * multi=(int*)malloc(sizeof(float)*56*56);

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
            spatial_dim=width_*height_;
            blob_size=num*channels_*width_*height_;
            //with origin data
            //printf("here 1\n");
            //init
            for(z = 0; z < blob_size; ++z ) top_data[z] = 0.0;
            for(z = 0; z < blob_size; ++z ) my_top_data[z]=0.0;
            for(z = 0; z < blob_size; ++z ) bbottom_data[z] = rand()/(float)RAND_MAX;
            for(z = 0; z < channels_; ++z ) bbias_data[z] = rand()/(float)RAND_MAX;
            for(z = 0; z < spatial_dim; ++z ) multi[z] = 1.0;

            out[0]='0'+ii;
            out[2]='0'+jj;
            out[4]='0'+kk;


            int mode = 0;
            //mode==0 : max
            //mode==1 : avg

            int use_top_mask = 0;
            int global_pooling_ = 0;

            int kernel_h_= 2;
            int kernel_w_= 2;
            int pooled_height_;
            int pooled_width_;
            int stride_h_=2;
            int stride_w_=2;
            int pad_h_=0;