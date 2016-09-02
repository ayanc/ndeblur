/* 
   Very little error checking. Use with Caution.

   --Ayan Chakrabarti <ayanc@ttic.edu>
*/

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <stdint.h>

#define F float

#define NUMT 1024
#define tid threadIdx.x



void __global__ mConv(F * y, F * x, F * k,
		      int ysz, int ksz, int bsz) {


	extern __shared__ F myk[];
	F sum, *kp;
	int bid, n, i, j, xsz, n2; 

	xsz = ysz+ksz-1;

	/* Responsible for finding y(n,bid) = x(:,bid)*k(:,bid) */
	bid = blockIdx.x;
	n = bid % bsz; bid /= bsz;
	n = n * NUMT + tid;


	/* Make a copy of k(:,bid) into shared memory */
	for(i = 0; i < ksz*ksz; i+= NUMT)
		if(i + tid < ksz*ksz)
			myk[i+tid] = k[bid*ksz*ksz+i+tid];
	
	__syncthreads();

	if(n < ysz*ysz) {
		sum = 0; 
		kp = &myk[ksz*ksz-1];
		n2 = n + (n/ysz)*(xsz-ysz); 
		for(j = 0; j < ksz; j++)
			for(i = 0; i < ksz; i++)
				sum += (*(kp--)) *
					x[ bid*xsz*xsz +
					   j*xsz + i + n2 ];
		y[bid*ysz*ysz+n] = sum;
	}
}


F * getGPUmem(const char * name) {

	const mxGPUArray * tmp;
	F * dptr;

	if(!mxIsGPUArray(mexGetVariablePtr("caller",name)))
		mexPrintf("%s is not on gpu!\n",name);

	tmp = mxGPUCreateFromMxArray(mexGetVariablePtr("caller",name));
	dptr = (F*) mxGPUGetDataReadOnly(tmp);
	mxGPUDestroyGPUArray(tmp);

	return (F*) dptr;
}

/* function mConv */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	F * y, * x, * k;
	int ysz, ksz, bsz, nsamp, numB, sharesz;

	ysz = (int) mxGetScalar(mexGetVariablePtr("caller","ysz"));
	ksz = (int) mxGetScalar(mexGetVariablePtr("caller","ksz"));
	nsamp = mxGetScalar(mexGetVariablePtr("caller","nsamp"));

	bsz = (ysz*ysz+NUMT-1)/NUMT; numB = bsz*nsamp; sharesz = ksz*ksz*sizeof(F);

	y = getGPUmem("y"); x = getGPUmem("x"); k = getGPUmem("k");

	mConv<<<numB,NUMT,sharesz>>>(y,x,k,ysz,ksz,bsz);
}
