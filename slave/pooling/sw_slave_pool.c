#include <stdio.h> 
#include <float.h>
#include <slave.h>
#include <math.h>
#include <dma.h>
#include <simd.h>
#include <string.h>
#include <limits.h>
#include <assert.h>

#define min(a,b) ((a)>(b)?(b):(a))
#define max(a,b) ((a)>(b)?(a):(b))

typedef double Type;
typedef struct _tagSlavePoolingParam
{
	int pooled_height_,pooled_width_,stride_h_,stride_w_,pad_h_,pad_w_,kernel_h_,kernel_w_,height_,width_;
	int nCount,nThreadsNum,nLeftThreadsNum;
	int nBottomOffset,nTopOffset,use_top_mask;
	int  *pMask;
	double *pTopData,*pBottomData,*pTopMask;
}SlavePoolingParam;



__thread_local_fix  dma_desc pool_dmaget2,dmaputmask,pool_dmaput2;
void poolingBackwardMax(SlavePoolingParam *pParam)
{
  const int nMaxBuffSize = 49152;//58KB 
	int pooled_height_,pooled_width_,stride_h_,stride_w_,pad_h_,pad_w_,kernel_h_,kernel_w_,height_,width_;
	int nCount,nMaxThreadsNum,nLeftMaxThreadsNum,nOffset,nOffset0,nOffset1;
	int nBottomOffset,nTopOffset,use_top_mask;
	int ph,pw,hstart,hend,wstart,wend,pool_index,h,w,index,bottom_index;
	Type *pTopData,*pBottomData,*pTopMask;	
	int  *pMask;	
  //dma_desc pool_dmaget2,pool_dmaput2;	
	volatile int getreply=0,putreply=0,putmaskreply=0;	
	int myid = athread_get_id(-1);
	
	pooled_height_ = pParam->pooled_height_;
	pooled_width_  = pParam->pooled_width_;
	stride_h_ = pParam->stride_h_;
	stride_w_ = pParam->stride_w_;
	pad_h_ = pParam->pad_h_;
	pad_w_ = pParam->pad_w_;
	kernel_h_ = pParam->kernel_h_;
	kernel_w_ = pParam->kernel_w_;
	height_ = pParam->height_;
	width_  = pParam->width_;	
	nCount = pParam->nCount;
	nMaxThreadsNum = pParam->nThreadsNum;
	nLeftMaxThreadsNum = pParam->nLeftThreadsNum;
	nBottomOffset = pParam->nBottomOffset;
	nTopOffset = pParam->nTopOffset;
	use_top_mask = pParam->use_top_mask;
	
	if(myid >= nMaxThreadsNum) return;	
	dma_set_op(&pool_dmaget2, DMA_GET);
	dma_set_mode(&pool_dmaget2, PE_MODE);
	dma_set_reply(&pool_dmaget2, &getreply);
	
	dma_set_op(&pool_dmaput2, DMA_PUT);
	dma_set_mode(&pool_dmaput2, PE_MODE);
	dma_set_reply(&pool_dmaput2, &putreply);	
	
	int nTopSize = pooled_height_ * pooled_width_*sizeof(Type),i=0,j=0;
	int nBottomSize = height_ * width_*sizeof(Type),nMaskSize=0;
	if(use_top_mask>0)
	  nMaskSize = nTopSize;
  else
	  nMaskSize = pooled_height_ * pooled_width_*sizeof(int);  	
	int nMaxSize = height_*width_-1;

	if((nTopSize+nBottomSize+nMaskSize) > nMaxBuffSize)
	{
	    nBottomSize = nMaxBuffSize - nTopSize - nMaskSize;
		int nSplitCount=0,nSplitRows =0,nLeftRows = 0;
		int nTopSize1 = 0,nMaskSize1=0,nBottomIndex;
		nSplitRows = pooled_height_;
        int nKernelSize = kernel_h_ *width_*sizeof(Type);
        int nStartAddr = 0;
		while(nBottomSize <nKernelSize)
		{
			nSplitRows = nSplitRows>>1;			
			nTopSize = nSplitRows*pooled_width_*sizeof(Type);
			if(use_top_mask >0) nMaskSize = nTopSize;				
			else nMaskSize = nSplitRows*pooled_width_*sizeof(int);
			nBottomSize = nMaxBuffSize - nTopSize - nMaskSize;
			nSplitCount++;
		}
		
		if(nSplitCount <1){
			nSplitRows = 0;
			nLeftRows = pooled_height_;
			nTopSize1 = pooled_height_ * pooled_width_*sizeof(Type);
			nMaskSize1 = pooled_height_ * pooled_width_*sizeof(int);
		}
	