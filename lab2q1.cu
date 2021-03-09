#include <stdio.h>
#include <assert.h>


//define the size of the matrix which is to be transposed
//
#define ROWS 8
#define COLS 8
#define N 16
#define k 16



//define a serial CPU version for transposr (For testing )
void transPose_CPU(int mat[],int ansCPU[]){
    for(int j=0;j<N;j++){
        for(int i=0;i<N;i++){
            ansCPU[j+i*N]=mat[i+j*N];
        }
    }

}

//define a serial GPU version for transposr (For testing )
__global__ void transPose_GPU(int ans[],int mat[]){
    for(int j=0;j<N;j++){
        for(int i=0;i<N;i++){
            ans[j+i*N]=mat[i+j*N];
        }
    }

}


//define a one thread per row to
//calculate a single row of outpu mattix use a thread 
// GPU version for transposr (For testing )
__global__ void transPose_GPU_row(int ans[],int mat[]){
    int i=threadIdx.x;// the inner loop value form the serial code

    //row is calculated by the threadId 
    // so we only need to run this for coumns
    //that is the only the outer loop should be run
    for(int j=0;j<N;j++){

        
        
            ans[j+i*N]=mat[i+j*N];//both mat i.j is inter changed with ans j ,i
        
    }

}



//define a parallel GPU version for transpose
//best is to use a single thread for a single element in the answer matrix
__global__ void transPose_GPU_element(int ans[],int mat[]){
   //from version 1 both i and j have to be implemented using thread indexes
   int i=blockIdx.x*k+threadIdx.x;
   int j=blockIdx.y*k+threadIdx.y;
   
       ans[j+i*N]=mat[i+j*N];//both mat i.j is inter changed with ans j ,i

 
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


//__global__  void transMat(int* ans_cuda, int * mat_cuda );

int main(){

    //arrays in main memory
    //2d arrays defined to hold the matrix 
    //int mat[ROWS][COLS];//the matrix
    //transose should be interchanged in num of rows and cols
	//int ans[COLS][ROWS];//the transposed matrix 

    //dynamic memory allocation in cpu RAM
    int numbytes=(N*N)*sizeof(int);

    int * mat=(int *)malloc(numbytes);
    int *ans=(int *)malloc(numbytes);
    int *ansCPU=(int *)malloc(numbytes);// to keep the CPU ansrwe for testing



	
    //fill the matrix with 
    int i,j;
	for(i=0;i<N;i++){
		for(j=0;j<N;j++){
			mat[j*N+i]=i*5+j;
		}
	}
    //test with the CPU transpose
    transPose_CPU(mat,ansCPU);

    //print the original matrix
    printf("Input matrix : \n");
	for(i=0;i<N;i++){
		for(j=0;j<N;j++){
			printf("%d ",mat[j*N+i]);
		}
		printf("\n");
	}


    //print the ans matrix
    printf("CPU Answer : \n");
	for(i=0;i<N;i++){
		for(j=0;j<N;j++){
			printf("%d ",ansCPU[j*N+i]);
		}
		printf("\n");
	}


    //here the cpu test case was passed
    //i have to parallelize it now

    //define pointers for d2 arrays to be put on cuda memory
    int *mat_cuda;
    int *ans_cuda;

    //for error checking
    cudaError_t code;
    
    //allocate memory in cuda device
    //size should be of ROWS* CLOS
    //that is basically SIZE macro define above
    //copy memory from ram to cuda
		//allocate memory in cuda
        cudaMalloc((void **)&mat_cuda,sizeof(int)*(N*N));
        checkCudaError();
        
        cudaMalloc((void **)&ans_cuda,sizeof(int)*(N*N));
        checkCudaError();
    
    
    //copy contents from main memory to cuda device memory
	cudaMemcpy(mat_cuda,mat,sizeof(int)*(N*N),cudaMemcpyHostToDevice);
	checkCudaError();
	
//.................................................Serial CUDA .........//////////


     //Start kernel time measure
	cudaEvent_t startkernel,stopkernel;
	float elapsedtimekernel;
	cudaEventCreate(&startkernel);
	cudaEventRecord(startkernel,0);


    //start the cuda kernel, which performs the matrix transpose
    //input is ans_cuda pointer while the inpt is mat_cuda
    transPose_GPU<<<1,1>>>(ans_cuda, mat_cuda);
    cudaDeviceSynchronize();
	code = cudaGetLastError();
    assert (code == cudaSuccess);

    //Stop time measure
	cudaEventCreate(&stopkernel);
	cudaEventRecord(stopkernel,0);
	cudaEventSynchronize(stopkernel);
	cudaEventElapsedTime(&elapsedtimekernel,startkernel,stopkernel);

    

    //Now the Device to Host 
    //GPU to CPU copy
    cudaMemcpy(ans,ans_cuda,sizeof(int)*(N*N),cudaMemcpyDeviceToHost);
    code = cudaGetLastError();
    assert (code == cudaSuccess);


//.....................Thread per raw implementation....  /////////////

//Start kernel time measure
cudaEvent_t startkernel1,stopkernel1;
float elapsedtimekernel1;
cudaEventCreate(&startkernel1);
cudaEventRecord(startkernel1,0);


//start the cuda kernel, which performs the matrix transpose
//input is ans_cuda pointer while the inpt is mat_cuda
//the single thread bloack consisting for N threads(Each thread fpr a rpw)
transPose_GPU_row<<<1,N>>>(ans_cuda, mat_cuda);
cudaDeviceSynchronize();
code = cudaGetLastError();
assert (code == cudaSuccess);

//Stop time measure
cudaEventCreate(&stopkernel1);
cudaEventRecord(stopkernel1,0);
cudaEventSynchronize(stopkernel1);
cudaEventElapsedTime(&elapsedtimekernel1,startkernel1,stopkernel1);

 
//.............////////////////
    
    //Now the Device to Host 
    //GPU to CPU copy
    cudaMemcpy(ans,ans_cuda,sizeof(int)*(N*N),cudaMemcpyDeviceToHost);
    code = cudaGetLastError();
    assert (code == cudaSuccess);
//.............//////////////// thread per element in the answer matrix///////////////////,,,,,,,,.......////////

//define num of blcks and therads per block
dim3 threadsPerBlock(k,k);
dim3 numBlocks(ceil(N/(float)k),ceil(N/(float)k));

//Start kernel time measure
cudaEvent_t startkernel2,stopkernel2;
float elapsedtimekernel2;
cudaEventCreate(&startkernel2);
cudaEventRecord(startkernel2,0);


//start the cuda kernel, which performs the matrix transpose
//input is ans_cuda pointer while the inpt is mat_cuda
//the single thread bloack consisting for N threads(Each thread fpr a rpw)
transPose_GPU_element<<<numBlocks,threadsPerBlock>>>(ans_cuda, mat_cuda);
cudaDeviceSynchronize();
code = cudaGetLastError();
assert (code == cudaSuccess);

//Stop time measure
cudaEventCreate(&stopkernel2);
cudaEventRecord(stopkernel2,0);
cudaEventSynchronize(stopkernel2);
cudaEventElapsedTime(&elapsedtimekernel2,startkernel2,stopkernel2);

 
//.............////////////////
    
    //Now the Device to Host 
    //GPU to CPU copy
    cudaMemcpy(ans,ans_cuda,sizeof(int)*(N*N),cudaMemcpyDeviceToHost);
    code = cudaGetLastError();
    assert (code == cudaSuccess);






    // ans matrix from GPU
    printf("GPU Answer : \n");
	for(i=0;i<N;i++){
		for(j=0;j<N;j++){
			printf("%d ",ans[j*N+i]);
		}
		printf("\n");
	}
  
      //print the timing reports
      fprintf(stderr,"Time spent for SERIAL version CUDA kernel is %1.5f seconds\n",elapsedtimekernel);
     fprintf(stderr,"Time spent for ThREAD PER ROW version CUDA kernel is %1.5f seconds\n",elapsedtimekernel1);
     fprintf(stderr,"Time spent for ThREAD PER ELEMENT version CUDA kernel is %1.5f seconds\n",elapsedtimekernel2);


    //free memory
    cudaFree(ans_cuda);
    cudaFree(mat_cuda);

    
	
	return 0;


}


