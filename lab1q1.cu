#include <stdio.h>
#include <assert.h>

#define SIZE 10

__global__  void addVector(float *vectorAns_cuda, float *vectorA_cuda );

int main(){

    //arrays in main memory
    float vectorA[SIZE];
    float vectorAns[SIZE/2];

    //generate some values (use randn())
    /*for(int i=0;i<SIZE;i++){
        vectorA[i]=rand();
    }*/
    //known vector for test answer 
    for(i=0;i<SIZE;i++){
		vectorA[i]=i;
	}

    //pointers for arrays to be put on cuda memory
    float *vectorA_cuda;
    float *vectorAns_cuda;

    //for error checking
    cudaError_t code;
    
    //allocate memory in cuda device
    cudaMalloc((void **)&vectorA_cuda,sizeof(float)*SIZE);
	code = cudaGetLastError();
    assert (code == cudaSuccess);
    
    cudaMalloc((void **)&vectorAns_cuda,sizeof(float)*(SIZE/2));
	code = cudaGetLastError();
    assert (code == cudaSuccess);
    
    //copy contents from main memory to cuda device memory
    cudaMemcpy(vectorA_cuda,vectorA,sizeof(float)*SIZE,cudaMemcpyHostToDevice);
	code = cudaGetLastError();
    assert (code == cudaSuccess);
    
    //call the cuda kernel
    addVector<<<1,SIZE>>>(vectorAns_cuda, vectorA_cuda);
    cudaDeviceSynchronize();
	code = cudaGetLastError();
    assert (code == cudaSuccess);
    
    //copy back the results from cuda memory to main memory
    cudaMemcpy(vectorAns,vectorAns_cuda,sizeof(float)*(SIZE/2),cudaMemcpyDeviceToHost);
    code = cudaGetLastError();
    assert (code == cudaSuccess);
    

    //free memory
    cudaFree(vectorA_cuda);
    cudaFree(vectorAns_cuda);


    printf("Answer is : ");

    for(i=0;i<(SIZE/2);i++){
		printf("%d ",vectorAns[i]);
	}
	
	return 0;


}


__global__  void addVector(float *vectorAns_cuda, float *vectorA_cuda ){
	
    int tid=threadIdx.x;
    /*if(tid==0){
        //the first element
        vectorAns_cuda[tid/2]=vectorA_cuda[tid]+vectorA_cuda[tid+1];
    }
    else if(tid%2==0){
        //other indexes than 0
    vectorAns_cuda[tid-1]=vectorA_cuda[tid]+vectorA_cuda[tid+1];
}*/
    if(tid%2==0){
        vectorAns_cuda[tid/2]=(vectorA_cuda[tid]+vectorA_cuda[tid+1])/2;
    }

  

}