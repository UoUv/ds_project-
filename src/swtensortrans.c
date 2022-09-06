
#include <stdio.h>
#include <stdlib.h>
#include <athread.h>
#include <simd.h>
#include "./include/swtensortrans.h"

extern void SLAVE_FUN(swapBN)();
extern void SLAVE_FUN(swapNBHW)();
extern void SLAVE_FUN(swapNBHW_ROLL)();
extern void SLAVE_FUN(swapBN_f)();
extern void SLAVE_FUN(swapNBHW_f)();
extern void SLAVE_FUN(swapNBHW_ROLL_f)();

// high -> low
// B, N, W, H
inline int image_caffe_offset(int b, int n, int h, int w, int B, int N, int H, int W) {
  return (((b*N + n)*H + h)*W + w);
}

// W, H, N, B
inline int image_swdnn_offset(int b, int n, int h, int w, int B, int N, int H, int W) {
  return (((h*W + w)*N + n)*B + b);
}
inline int get_split_size(int nSize,int nMaxSize)
{
	int nVal = nSize/nMaxSize,nSplitSize = 0;
	if(nVal<1) 
	{
      nSplitSize = nSize - nSize%4;
	}
	else if(nVal>=nMaxSize) nSplitSize = nMaxSize;
	else{
		int nModHW = nSize - nSize%4,nTmp=0;

		nSplitSize = 0;
		for(;nVal<nMaxSize;nVal++)
		{
			nTmp = nModHW/nVal;
			if(nTmp <nMaxSize && (nTmp % 4 == 0))
			{
				nSplitSize = nTmp;
				break;
			}
		}
		if(nSplitSize<16){
			nSplitSize = (nModHW>>2);
			nSplitSize = nSplitSize - nSplitSize%4;
		}
	}
	return nSplitSize;
}

/********
 * float* in is a matrix of size (highDim, lowDim)
 * transpose the matrix to (lowDim, highDim) and store in out
 * ******/
void swap_lowdim_f(float*in, float*out, int highDim, int lowDim)
{
	int cRi, cCi, cNi, cB;
	int nHW = lowDim;
	int nNB = highDim;
	//process the problem of the (H,W) very small
	if(nHW < 4)
	{
	    for(cCi = 0; cCi < nHW; ++cCi)
			for(cRi = 0; cRi < nNB; ++cRi)
				out[cCi*nNB+cRi] = in[cRi*nHW+cCi]; 
        return;
	}
	SlaveParam_f param;
	param.B = highDim;
	param.N = 1;
	param.H = lowDim;
	param.W = 1;
	param.pIn = in;
	param.pOut = out;
	param.splitHW = get_split_size(nHW,HWSIZE);
	param.splitNB = get_split_size(nNB,BSIZE);
  int nTmp = NUM_THREADS*param.splitNB;
	//printf("N=%d B=%d H=%d W=%d splitNB=%d splitHW=%d\n",N,B,H,W,param.splitNB,param.splitHW);
	param.nCount = nNB/nTmp;
	nTmp = (nNB/param.splitNB)%NUM_THREADS;
	param.nNBHWThreadsNum = param.nCount >0 ? NUM_THREADS:nTmp;
	param.nNBHWLeftThreadsNum = param.nCount >0 ? nTmp:0;
	athread_spawn(swapNBHW_f,(void *)&param);
	//process the slave core left data
	int nHWLeft = nHW%(param.splitHW);
	int nBNLeft = nNB%(param.splitNB);
	if(nHWLeft >0 || nBNLeft >0)
	{
		int nC = nHW - nHWLeft;
		int nR = nNB - nBNLeft;
		for(cCi = nC; cCi < nHW; ++cCi)
			for(cRi = 0; cRi < nNB; ++cRi)
				out[cCi*nNB+cRi] = in[cRi*nHW+cCi];
		for(cRi = nR; cRi < nNB; ++cRi)
			for(cCi = 0; cCi < nC; ++cCi)
				out[cCi*nNB+cRi] = in[cRi*nHW+cCi];
    }
    athread_join();
}

/*******
 * tensor in is of shape (B, N, H, W)
 * out is of shape (N, B, H, W)
 * swap two high dimension of 4D tensor
 * *****/
inline void swapBN_d(double*in,double*out,int B,int N,int H, int W)
{
	int nNB = N*B;	
    SlaveParam param;
	
	param.B = B;
	param.N = N;
	param.H = H;
	param.W = W;
	param.pIn = in;
	param.pOut = out;	
	param.nCount = nNB/NUM_THREADS;
	int nTmp = nNB%NUM_THREADS;
	param.nBNThreadsNum = param.nCount >0 ? NUM_THREADS:nTmp;
	param.nBNLeftThreadsNum = param.nCount >0 ? nTmp:0;
	
	athread_spawn(swapBN,(void *)&param);
  athread_join();
}

inline void swapBN_f(float*in,float*out,int B,int N,int H, int W)
{
	int nNB = N*B;	
  SlaveParam_f param;
	
	param.B = B;
	param.N = N;
	param.H = H;
	param.W = W;
	param.pIn = in;
	param.pOut = out;	
	param.nCount = nNB/NUM_THREADS;
	int nTmp = nNB%NUM_THREADS;
	param.nBNThreadsNum = param.nCount >0 ? NUM_THREADS:nTmp;
	param.nBNLeftThreadsNum = param.nCount >0 ? nTmp:0;
	
	athread_spawn(swapBN_f,(void *)&param);
	athread_join();
}
inline void swapBN_HW_d(double*in,double*out,int B,int N,int H, int W)
{
	int cRi, cCi, cNi, cB;
	int nHW = H*W;
	int nNB = N*B;
	//process the problem of the (H,W) very small
	if(nHW < 4)
	{
	    for(cCi = 0; cCi < nHW; ++cCi)
			for(cRi = 0; cRi < nNB; ++cRi)
				out[cCi*nNB+cRi] = in[cRi*nHW+cCi]; 
        return;
	}
	
		
	SlaveParam param;
	
	param.B = B;
	param.N = N;
	param.H = H;
	param.W = W;
	param.pIn = in;
	param.pOut = out;	
	
	param.splitHW = get_split_size(nHW,HWSIZE);		
	param.splitNB = get_split_size(nNB,BSIZE);		
	int nTmp = NUM_THREADS*param.splitNB;	
	//printf("N=%d B=%d H=%d W=%d splitNB=%d splitHW=%d\n",N,B,H,W,param.splitNB,param.splitHW);
	param.nCount = nNB/nTmp;
	nTmp = (nNB/param.splitNB)%NUM_THREADS;
	param.nNBHWThreadsNum = param.nCount >0 ? NUM_THREADS:nTmp;
	param.nNBHWLeftThreadsNum = param.nCount >0 ? nTmp:0;
	
	athread_spawn(swapNBHW,(void *)&param);
	
	//process the slave core left data
	int nHWLeft = nHW%(param.splitHW);	
	int nBNLeft = nNB%(param.splitNB);	
	if(nHWLeft >0 || nBNLeft >0)
	{
		int nC = nHW - nHWLeft;
		int nR = nNB - nBNLeft;
		
		for(cCi = nC; cCi < nHW; ++cCi)
			for(cRi = 0; cRi < nNB; ++cRi)
				out[cCi*nNB+cRi] = in[cRi*nHW+cCi]; 
    
		for(cRi = nR; cRi < nNB; ++cRi)
			for(cCi = 0; cCi < nC; ++cCi)
				out[cCi*nNB+cRi] = in[cRi*nHW+cCi]; 
    }	
    athread_join();			
}

inline void swapBN_HW_f(float*in,float*out,int B,int N,int H, int W)
{
	int cRi, cCi, cNi, cB;
	int nHW = H*W;
	int nNB = N*B;
	//process the problem of the (H,W) very small
	if(nHW < 4)
	{
	    for(cCi = 0; cCi < nHW; ++cCi)
			for(cRi = 0; cRi < nNB; ++cRi)
				out[cCi*nNB+cRi] = in[cRi*nHW+cCi]; 
        return;
	}
	
		
	SlaveParam_f param;
	
	param.B = B;
	param.N = N;
	param.H = H;
	param.W = W;
	param.pIn = in;
	param.pOut = out;	
	
	param.splitHW = get_split_size(nHW,HWSIZE);		
	param.splitNB = get_split_size(nNB,BSIZE);		
	int nTmp = NUM_THREADS*param.splitNB;	
	//printf("N=%d B=%d H=%d W=%d splitNB=%d splitHW=%d\n",N,B,H,W,param.splitNB,param.splitHW);
	param.nCount = nNB/nTmp;
	nTmp = (nNB/param.splitNB)%NUM_THREADS;