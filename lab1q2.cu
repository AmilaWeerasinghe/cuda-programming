#include <stdio.h>
#include <assert.h>
#include <math.h>

#define SIZE 10

__global__  void Rad(double *rad_cuda, double *vectorA_cuda );
__global__  void Cos(double *cos_cuda, double *vectorA_cuda );


int main(){

    //arrays in main memory
    double vectorA[SIZE]; //holds the angles degrees
    double rad[SIZE];//angles in radian
    double cos[SIZE];//angles in radian

    //generate some values for angles
	int i;
	/*for(i=0;i<SIZE;i++){
		vectorA[i]=rand();
    }*/
    //testing purpose(known input)
    for(i=0;i<SIZE;i++){
		vectorA[i]=i*90;
    }
    

    /////////////////////////////////////////////////////// PART 1 DEGREE TO RADIAN////////////////////////////////////////
    //pointers for arrays to be put on cuda memory
    double *vectorA_cuda;
    double *rad_cuda;
    double *cos_cuda;


    //for error checking
    cudaError_t code;

    //allocate memory in cuda device
    cudaMalloc((void **)&vectorA_cuda,sizeof(double)*SIZE); 
    code = cudaGetLastError();
    assert (code == cudaSuccess);

    cudaMalloc((void **)&rad_cuda,sizeof(double)*SIZE);
    code = cudaGetLastError();
    assert (code == cudaSuccess);
    
    //copy contents from main memory to cuda device memory
    cudaMemcpy(vectorA_cuda,vectorA,sizeof(double)*SIZE,cudaMemcpyHostToDevice);
    code = cudaGetLastError();
    assert (code == cudaSuccess);

    //call the cuda kernel
    Rad<<<1,SIZE>>>(rad_cuda, vectorA_cuda);
    cudaDeviceSynchronize();
	  code = cudaGetLastError();
    assert (code == cudaSuccess);

    //copy back the results from cuda memory to main memory
    cudaMemcpy(rad,rad_cuda,sizeof(double)*(SIZE),cudaMemcpyDeviceToHost);
    code = cudaGetLastError();
    assert (code == cudaSuccess);

    //free memory
    cudaFree(rad_cuda);
	


	printf("Radian is : ");

	for(i=0;i<SIZE;i++){
		printf("%lf ",rad[i]);
  }
  
  printf("\n");
  ////////////////////////////////////PART B COS ////////////////////////////////////////////////
  //pointers for arrays to be put on cuda memory
  //double *cos_cuda;


  //allocate memory in cuda device
  cudaMalloc((void **)&cos_cuda,sizeof(double)*SIZE);
  code = cudaGetLastError();
  assert (code == cudaSuccess);
  
  //copy contents from main memory to cuda device memory
  cudaMemcpy(vectorA_cuda,vectorA,sizeof(double)*SIZE,cudaMemcpyHostToDevice);
  code = cudaGetLastError();
  assert (code == cudaSuccess);

  //call the cuda kernel
  Cos<<<1,SIZE>>>(cos_cuda, vectorA_cuda);
  cudaDeviceSynchronize();
  code = cudaGetLastError();
  assert (code == cudaSuccess);

  //copy back the results from cuda memory to main memory
  cudaMemcpy(cos,cos_cuda,sizeof(double)*(SIZE),cudaMemcpyDeviceToHost);
  code = cudaGetLastError();
  assert (code == cudaSuccess);

  //free memory
  cudaFree(cos_cuda);



printf("Cos is : ");

for(i=0;i<SIZE;i++){
  printf("%lf ",rad[i]);
}


return 0;

}

//degree to rad
__global__  void Rad(double *rad_cuda, double *vectorA_cuda ){
	
	int tid=threadIdx.x;
	rad_cuda[tid]=vectorA_cuda[tid]*(M_PI/180);

}


//
__global__  void Cos(double *cos_cuda, double *vectorA_cuda ){
	
	int tid=threadIdx.x;
	cos_cuda[tid]=cosf(vectorA_cuda[tid]);

}