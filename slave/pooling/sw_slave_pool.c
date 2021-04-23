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
	int pooled_hei