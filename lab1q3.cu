#include <stdio.h>
#include <stdlib.h>

#define ROWS 5
#define COLS 10
#define SIZE ROWS*COLS

/* Used the error checking snippet given in the feels */
#define checkCudaError() { gpuAssert(__FILE__, __LINE__); }

void gpuAssert(const char *file, int line){

	cudaError_t code = cudaGetLastError();
	if (code != cudaSuccess) {
        fprintf(stderr, "Cuda error: %s \n in file : %s line number : %d", cudaGetErrorString(code), file, line );
        exit(1);
   }
}

//kernel function to perform substraction
__global__ void subMatrix(int *ans_cuda,int *matA_cuda,int *matB_cuda);




int main(){

	//variabels to hold the matrices
	int matA[ROWS][COLS];
	int matB[ROWS][COLS];
	int ans[ROWS][COLS];
	
	//initialize the martices
	int i=0,j=0,k=0;
	for(i=0;i<ROWS;i++){
		for(j=0;j<COLS;j++){
			matA[i][j]=k;
			matB[i][j]=ROWS*COLS-k;
			k++;
		}
	}
	


	//variables for time measurements
	cudaEvent_t start,stop;
	
	//pointers for cuda memory locations
	int *matA_cuda;
	int *matB_cuda;
	int *ans_cuda;	

	//the moment at which we start measuring the time
	cudaEventCreate(&start);
	cudaEventRecord(start,0);
	float elapsedtime;
	
	//allocate memory in cuda
	cudaMalloc((void **)&matA_cuda,sizeof(int)*SIZE); checkCudaError();
	cudaMalloc((void **)&matB_cuda,sizeof(int)*SIZE); checkCudaError();	
	cudaMalloc((void **)&ans_cuda,sizeof(int)*SIZE); checkCudaError();
		
	//copy contents from ram to cuda
	cudaMemcpy(matA_cuda, matA, sizeof(int)*SIZE, cudaMemcpyHostToDevice); checkCudaError();
	cudaMemcpy(matB_cuda, matB, sizeof(int)*SIZE, cudaMemcpyHostToDevice); checkCudaError();	 

	//thread configuration 
	dim3 numBlocks(ceil(COLS/(float)16),ceil(ROWS/(float)16));
	dim3 threadsPerBlock(16,16);
	
	//do the matrix addition on CUDA
	subMatrix<<<numBlocks,threadsPerBlock>>>(ans_cuda,matA_cuda,matB_cuda);
	cudaDeviceSynchronize(); checkCudaError();

	//copy the answer back
	cudaMemcpy(ans, ans_cuda, sizeof(int)*SIZE, cudaMemcpyDeviceToHost);
	checkCudaError();

	//the moment at which we stop measuring time 
	cudaEventCreate(&stop);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);

	//free the memory we allocated on CUDA
	cudaFree(matA_cuda);
	cudaFree(matB_cuda);
	cudaFree(ans_cuda);

	//calculate and print the elapsed time
    cudaEventElapsedTime(&elapsedtime,start,stop);
    fprintf(stderr,"Time spent is %.10f seconds\n",elapsedtime/(float)1000);
	
/*************************CUDA STUFF ENDS HERE************************/

	for(i=0;i<ROWS;i++){
		for(j=0;j<COLS;j++){
			printf("%d ",ans[i][j]);
		}
		puts("");
	}
	
	return 0;

}


//kernel that does the matrix addition. Just add each element to the respective one
__global__ void subMatrix(int *ans_cuda,int *matA_cuda,int *matB_cuda){
		
	//calculate the row number based on block IDs and thread IDs
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	//calculate the column number based on block IDs and thread IDs
	int col = blockDim.x * blockIdx.x + threadIdx.x;

	//to remove any indices beyond the size of the array
	if (row<ROWS && col <COLS){
		
		//conversion of 2 dimensional indices to single dimension
		int position = row*COLS + col;
	
		//do the calculation
		ans_cuda[position]=matA_cuda[position]-matB_cuda[position];
	
	}
}
	