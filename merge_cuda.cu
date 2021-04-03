#include <stdlib.h>
#include <stdio.h>


//size of the list
#define SIZE 1048576

//time fr omp
double wtime;


//definitions of global host and device funstion needed 
void merge_On_CPU(int *list, int *sorted, int start, int mid, int end);
__device__ void mergeDevice(int *list, int *sorted, int start, int mid, int end);
__device__ void Merge_sort_on_GPU(int *list, int *sorted, int start, int end);
__global__ void Initial_Merge_sort(int *list, int *sorted, int Size_of_A_chunk, int N);




//function to initilse the array with known values
void Initialise_array(int N, int *array_pointer) {
   int i;
   for (i=0; i<N; i++) 
     array_pointer[i] = N-i;
   
}

int main() {

    //create the variables and pointers
    int *arr_h, *arrSorted_h, *arrSortedF_h;
    int *arr_d, *arrSorted_d, *arrSortedF_d;
    //the chunk size
    int Size_of_A_chunk;
    //total bytes size
    unsigned int byte_size;
    unsigned int Num_of_threads;
    unsigned int N; 
    unsigned int Num_of_blocks;
    
   
    N = SIZE;
    Num_of_threads = 256;
    Num_of_blocks = 16;
    Size_of_A_chunk = N/(Num_of_threads*Num_of_blocks);

    byte_size = N * sizeof(int);

    cudaEvent_t start, stop;
    float Execution_time;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    arr_h = (int*) malloc(byte_size);
    arrSorted_h = (int*) malloc(byte_size);
    arrSortedF_h = (int*) malloc(byte_size);



    //initilasie the array with given values
    Initialise_array(N, arr_h);
  
       
      

    cudaMalloc((int**)&arr_d, byte_size);
    cudaMalloc((int**)&arrSorted_d, byte_size);
    cudaMalloc((int**)&arrSortedF_d, byte_size);

    cudaMemcpy(arr_d, arr_h, byte_size, cudaMemcpyHostToDevice);
    
    printf("Given array is \n"); 
    //printArray(arr_h, N);

    for (int i=0; i < SIZE; i++) 
        printf("%d ", arr_h[i]); 
    printf("\n"); 
    
    cudaEventRecord(start, 0);
    Initial_Merge_sort<<<Num_of_blocks, Num_of_threads>>>(arr_d, arrSorted_d,Size_of_A_chunk, N);
    cudaMemcpy(arrSorted_h, arrSorted_d, byte_size, cudaMemcpyDeviceToHost);
    for (int i = Size_of_A_chunk*2; i < N + Size_of_A_chunk; i = i + Size_of_A_chunk) {
	int mid = i-Size_of_A_chunk;
	int end = i;
	if (end > N) {
	  end = N;
	}
	merge_On_CPU(arrSorted_h,arrSortedF_h, 0, mid, end);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaFree(arr_d);
    cudaFree(arrSorted_d);
    cudaFree(arrSortedF_d);

    cudaEventElapsedTime(&Execution_time,  start, stop);
    
    printf("\nSorted array is \n"); 
    //printArray(arrSortedF_h, N); 

    //print the sorted array
    for (int i=0; i < SIZE; i++) 
        printf("%d ", arrSortedF_h[i]); 
    printf("\n");
    
    printf("Num_of_threads: %d\n", Num_of_threads);
    printf("Num_of_blocks: %d\n", Num_of_blocks);
    printf("Total execution time %f secs\n", (Execution_time/1000));
   
    return 0; 
    
}

//merge part on host 
void merge_On_CPU(int *list, int *sorted, int start, int mid, int end)
{
    int ti=start, i=start, j=mid;
    while (i<mid || j<end)
    {
        if (j==end) sorted[ti] = list[i++];
        else if (i==mid) sorted[ti] = list[j++];
        else if (list[i]<list[j]) sorted[ti] = list[i++];
        else sorted[ti] = list[j++];
        ti++;
    }

    for (ti=start; ti<end; ti++)
        list[ti] = sorted[ti];
}



__device__ void mergeDevice(int *list, int *sorted, int start, int mid, int end)
{
    int ti=start, i=start, j=mid;
    while (i<mid || j<end)
    {
        if (j==end) sorted[ti] = list[i++];
        else if (i==mid) sorted[ti] = list[j++];
        else if (list[i]<list[j]) sorted[ti] = list[i++];
        else sorted[ti] = list[j++];
        ti++;
    }

    for (ti=start; ti<end; ti++)
        list[ti] = sorted[ti];
}

__device__ void Merge_sort_on_GPU(int *list, int *sorted, int start, int end)
{   
    if (end-start<2)
        return;
  
    Merge_sort_on_GPU(list, sorted, start, start + (end-start)/2);
    Merge_sort_on_GPU(list, sorted, start + (end-start)/2, end);
    mergeDevice(list, sorted, start, start + (end-start)/2, end);
    
}


__global__ void Initial_Merge_sort(int *list, int *sorted, int Size_of_A_chunk, int N) {
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	int start = tid*Size_of_A_chunk;
	int end = start + Size_of_A_chunk;
	if (end > N) {
		end = N;
	}
	Merge_sort_on_GPU(list, sorted, start, end);
}