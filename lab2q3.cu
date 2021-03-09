#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>
#include <time.h>

#define BLOCK_SIZE 16

//inut 32 32
#define data_matrix_width 16
#define data_matrix_height 16

//kernel size 8*8
#define kernel_matrix_height 3 
#define kernel_matrix_width 3

//we can define the result matrix size of the GPU_Conv using above
#define conv_matrix_width (data_matrix_width - kernel_matrix_width + 1)
#define conv_matrix_height (data_matrix_height - kernel_matrix_height + 1)


__global__ void GPU_Conv(int* A, int* B, int* C, int numARows, int numACols, int numBRows, int numBCols, int numCRows, int numCCols)
{

	//2d kernel is writtem
	// in the persective of a thread allocated per a single element is ther result matrix
	//therfore  thread index are assingnefd to determine a single elemnt 
	int col = blockIdx.x * (BLOCK_SIZE - kernel_matrix_width + 1) + threadIdx.x;
	int row = blockIdx.y * (BLOCK_SIZE - kernel_matrix_width + 1) + threadIdx.y;
	int row_i = row - kernel_matrix_width + 1;
	int col_i = col - kernel_matrix_width + 1;


	//define a temp to hold the accumilated value

	int tmp = 0;

	//define a share memory buffer to immrove performance

	__shared__ int memShared[BLOCK_SIZE][BLOCK_SIZE];

	if (row_i < data_matrix_width && row_i >= 0 && col_i < data_matrix_width && col_i >= 0)
	{
		memShared[threadIdx.y][threadIdx.x] = A[col_i * data_matrix_width + row_i];
	}
	else
	{
		memShared[threadIdx.y][threadIdx.x] = 0;
	}


	//wait till all threads perfom the task
	__syncthreads();

	//assign the temp for the elements

	if (threadIdx.y < (BLOCK_SIZE - kernel_matrix_width + 1) && threadIdx.x < (BLOCK_SIZE - kernel_matrix_width + 1) && row < (conv_matrix_width - kernel_matrix_width + 1) && col < (conv_matrix_width - kernel_matrix_width + 1))
	{
		for (int i = 0; i< kernel_matrix_width;i++)
			for (int j = 0;j<kernel_matrix_width;j++)
				tmp += memShared[threadIdx.y + i][threadIdx.x + j] * C[j*kernel_matrix_width + i];
		B[col*conv_matrix_width + row] = tmp;
	}
}

//initialize the matrix with known values
void Matrix_Intialize(int* data, int size)
{
	for (int i = 0; i < size;i++)
		data[i]=i;
}

//initialize the kernel matrix with known values
void kernelInit(int* data, int size)
{
	for (int i = 0; i < size;i++)
		data[i]=i;
}

//we can have this in a seperate file 
//and import it as a header file if we need 
//so we can make the code
#define checkCudaError() { gpuAssert(__FILE__, __LINE__); }

static inline void gpuAssert(const char *file, int line){
	cudaError_t code = cudaGetLastError();
	if (code != cudaSuccess) {
        fprintf(stderr, "error of CUDA: %s \n in file : %s line number : %d", cudaGetErrorString(code), file, line);
        exit(1);
   }
}

int main(int argc, char** argv)
{
	


	cudaEvent_t start_G, stop_G;

	cudaEventCreate(&start_G);
	cudaEventCreate(&stop_G);

	unsigned int size_A = data_matrix_width * data_matrix_height;
	unsigned int mem_size_A = sizeof(int) * size_A;
	int* data_matrix_CPU = (int*)malloc(mem_size_A);

	unsigned int size_B = conv_matrix_width * conv_matrix_height;
	unsigned int mem_size_B = sizeof(int) * size_B;
	int* result_matrix_CPU = (int*)malloc(mem_size_B);

	unsigned int size_C = kernel_matrix_width * kernel_matrix_height;
	unsigned int mem_size_C = sizeof(int) * size_C;
	int* kernel_matrix_CPU = (int*)malloc(mem_size_C);


	Matrix_Intialize(data_matrix_CPU, size_A);
	//print the matrix
	printf("Matrix A Answer : \n");	
	for(int i=0;i<data_matrix_width;i++){
		for(int j=0;j<data_matrix_height;j++){
			printf("%d ",data_matrix_CPU[i*data_matrix_height+j]);
		}
		printf("\n");
	}	

	kernelInit(kernel_matrix_CPU, size_C);

	printf("Kernel is : \n");	
	for(int i=0;i<kernel_matrix_width;i++){
		for(int j=0;j<kernel_matrix_height;j++){
			printf("%d ",kernel_matrix_CPU[i*kernel_matrix_height+j]);
		}
		printf("\n");
	}	

	int* data_matrix;
	int* conv_matrix;
	int* kernel_matrix;

	cudaMalloc((void**)&data_matrix, mem_size_A);checkCudaError();

	cudaMalloc((void**)&conv_matrix, mem_size_B);checkCudaError();
	cudaMalloc((void**)&kernel_matrix, mem_size_C);checkCudaError();



	cudaMemcpy(data_matrix, data_matrix_CPU, mem_size_A, cudaMemcpyHostToDevice);checkCudaError();
	cudaMemcpy(kernel_matrix, kernel_matrix_CPU, mem_size_C, cudaMemcpyHostToDevice);checkCudaError();
	

	//new num of bloacks
	dim3 threadsPerBlock(BLOCK_SIZE,BLOCK_SIZE);
	dim3 numBlocks(ceil(data_matrix_height/(float)BLOCK_SIZE),ceil(data_matrix_width/(float)BLOCK_SIZE));

	//GPU_Conv << < numBlocks, threadsPerBlock >> >(data_matrix, conv_matrix, kernel_matrix, data_matrix_height, data_matrix_width, conv_matrix_height, conv_matrix_width, kernel_matrix_height, kernel_matrix_width);

	cudaEventRecord(start_G);

	//GPU_Conv << < grid, threads >> >(data_matrix, conv_matrix, kernel_matrix, data_matrix_height, data_matrix_width, conv_matrix_height, conv_matrix_width, kernel_matrix_height, kernel_matrix_width);
	GPU_Conv << < numBlocks, threadsPerBlock >> >(data_matrix, conv_matrix, kernel_matrix, data_matrix_height, data_matrix_width, conv_matrix_height, conv_matrix_width, kernel_matrix_height, kernel_matrix_width);
	cudaDeviceSynchronize();checkCudaError();

	cudaEventRecord(stop_G);

	cudaEventSynchronize(stop_G);

	cudaMemcpy(result_matrix_CPU, conv_matrix, mem_size_B, cudaMemcpyDeviceToHost);checkCudaError();

	float miliseconds = 0;
	cudaEventElapsedTime(&miliseconds, start_G, stop_G);

	printf("Time took to compute matrix A of dimensions %d x %d  on GPU is %f ms \n \n \n", data_matrix_width, data_matrix_height, miliseconds);

	//print the convolution matrix
	for (int i = 0;i < conv_matrix_height;i++)
	{
		for (int j = 0;j < conv_matrix_width;j++)
		{
			printf("%d ", result_matrix_CPU[i*conv_matrix_height + j]);
		}
		printf("\n");
	}

	free(data_matrix_CPU);
	free(result_matrix_CPU);
	free(kernel_matrix_CPU);
	cudaFree(data_matrix);
	cudaFree(conv_matrix);
	cudaFree(kernel_matrix);

	return EXIT_SUCCESS;
}