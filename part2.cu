#include <stdio.h>
#include <assert.h>
#include <math.h>

#define SIZE 10

__global__  void Rad(double *rad_cuda, double *vectorA_cuda );
__global__  void Cos(double *cos_cuda, double *rad_cuda );
__global__  void Sin(double *sin_cuda, double *rad_cuda );

__global__  void sqrtVector(double *sqrt_cuda, double *firstVector_cuda, double *secondVector_cuda );


int main(){

    //arrays in main memory
    double vectorA[SIZE]; //holds the angles degrees
    double rad[SIZE];//angles in radian
    double cos[SIZE];//cos values for radian inputs
    double sin[SIZE];//sin values for radian inputs

    //two arrays to square and add(For testing)
    double firstVector[SIZE];
    double secondVector[SIZE];

    //hold the squared 
    double srqt[SIZE];

    //generate some values for angles
	int i;
	/*for(i=0;i<SIZE;i++){
		vectorA[i]=rand();
    }*/
    //testing purpose(known input)
    for(i=0;i<SIZE;i++){
		vectorA[i]=i*90;
    }
      // fill the second vector with known values
    for(i=0;i<SIZE;i++){
      firstVector[i]=i;
      secondVector[i]=i;
      
      }
    

    /////////////////////////////////////////////////////// PART 1 DEGREE TO RADIAN////////////////////////////////////////
    //pointers for arrays to be put on cuda memory
    double *vectorA_cuda;
    double *rad_cuda;
    double *cos_cuda;
    double *sin_cuda;
    double *sqrt_cuda;


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
	


	printf("a) Radian is : ");

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
  //cudaMemcpy(vectorA_cuda,vectorA,sizeof(double)*SIZE,cudaMemcpyHostToDevice);
  cudaMemcpy(rad_cuda,rad,sizeof(double)*SIZE,cudaMemcpyHostToDevice);
  code = cudaGetLastError();
  assert (code == cudaSuccess);

  //call the cuda kernel
  //Cos<<<1,SIZE>>>(cos_cuda, vectorA_cuda);
  Cos<<<1,SIZE>>>(cos_cuda, rad_cuda);
  cudaDeviceSynchronize();
  code = cudaGetLastError();
  assert (code == cudaSuccess);

  //copy back the results from cuda memory to main memory
  cudaMemcpy(cos,cos_cuda,sizeof(double)*(SIZE),cudaMemcpyDeviceToHost);
  code = cudaGetLastError();
  assert (code == cudaSuccess);

  //free memory
  cudaFree(cos_cuda);



printf("b)  Cos is : ");

for(i=0;i<SIZE;i++){
  printf("%lf ",cos[i]);
}

printf("\n");


 ////////////////////////////////////PART C SIN ////////////////////////////////////////////////
  //pointers for arrays to be put on cuda memory
  //double *sin_cuda;


  //allocate memory in cuda device
  cudaMalloc((void **)&sin_cuda,sizeof(double)*SIZE);
  code = cudaGetLastError();
  assert (code == cudaSuccess);
  
  //copy contents from main memory to cuda device memory
  cudaMemcpy(rad_cuda,rad,sizeof(double)*SIZE,cudaMemcpyHostToDevice);
  code = cudaGetLastError();
  assert (code == cudaSuccess);

  //call the cuda kernel
  Sin<<<1,SIZE>>>(sin_cuda, rad_cuda);
  cudaDeviceSynchronize();
  code = cudaGetLastError();
  assert (code == cudaSuccess);

  //copy back the results from cuda memory to main memory
  cudaMemcpy(sin,sin_cuda,sizeof(double)*(SIZE),cudaMemcpyDeviceToHost);
  code = cudaGetLastError();
  assert (code == cudaSuccess);

  //free memory
  cudaFree(sin_cuda);



printf("c) Sin is : ");

for(i=0;i<SIZE;i++){
  printf("%lf ",sin[i]);
}

printf("\n");

///////////////////////////Part D Square Sum/////////////////////////////////////////////////////////////////////////////
//pointers for arrays to be put on cuda memory
  //double *sin_cuda;
  double *firstVector_cuda;
  double *secondVector_cuda;


  //allocate memory in cuda device
  cudaMalloc((void **)&firstVector_cuda,sizeof(double)*SIZE);
  code = cudaGetLastError();
  assert (code == cudaSuccess);

  cudaMalloc((void **)&secondVector_cuda,sizeof(double)*SIZE);
  code = cudaGetLastError();
  assert (code == cudaSuccess);

  cudaMalloc((void **)&sqrt_cuda,sizeof(double)*SIZE);
  code = cudaGetLastError();
  assert (code == cudaSuccess);
  
  //copy contents from main memory to cuda device memory
  cudaMemcpy(firstVector_cuda,firstVector,sizeof(double)*SIZE,cudaMemcpyHostToDevice);
  code = cudaGetLastError();
  assert (code == cudaSuccess);

  cudaMemcpy(secondVector_cuda,secondVector,sizeof(double)*SIZE,cudaMemcpyHostToDevice);
  code = cudaGetLastError();
  assert (code == cudaSuccess);


  cudaMemcpy(sqrt_cuda,srqt,sizeof(double)*SIZE,cudaMemcpyHostToDevice);
  code = cudaGetLastError();
  assert (code == cudaSuccess);

  //call the cuda kernel
  sqrtVector<<<1,SIZE>>>(sqrt_cuda, firstVector_cuda,secondVector_cuda);
  cudaDeviceSynchronize();
  code = cudaGetLastError();
  assert (code == cudaSuccess);

  //copy back the results from cuda memory to main memory
  cudaMemcpy(srqt,sqrt_cuda,sizeof(double)*(SIZE),cudaMemcpyDeviceToHost);
  code = cudaGetLastError();
  assert (code == cudaSuccess);

  //free memory
  cudaFree(sqrt_cuda);



printf("d) Squared values is : ");

for(i=0;i<SIZE;i++){
  printf("%lf ",srqt[i]);
}

printf("\n");




return 0;

}

//degree to rad
__global__  void Rad(double *rad_cuda, double *vectorA_cuda ){
	
	int tid=threadIdx.x;
	rad_cuda[tid]=vectorA_cuda[tid]*(M_PI/180);

}


//Cude Math library cos function used which is for double precision according to the documentation 
//Here the cos is calculated for radians in the cos function
//So i used the converted radian vector in part a), allocated to GPU took the cos of each element 
__global__  void Cos(double *cos_cuda, double *rad_cuda ){
	
	int tid=threadIdx.x;
  //cos_cuda[tid]=cos(vectorA_cuda[tid]);
  cos_cuda[tid]=cos(rad_cuda[tid]);

}

//Cude Math library cos function used which is for double precision according to the documentation 
//Here the cos is calculated for radians in the cos function
__global__  void Sin(double *sin_cuda, double *rad_cuda ){
	
	int tid=threadIdx.x;
	sin_cuda[tid]=sin(rad_cuda[tid]);

}

//squared addition
__global__  void sqrtVector(double *sqrt_cuda, double *firstVector_cuda, double *secondVector_cuda ){
	
	int tid=threadIdx.x;
	sqrt_cuda[tid]=(firstVector_cuda[tid]*firstVector_cuda[tid]) +(secondVector_cuda[tid]*secondVector_cuda[tid]);

}


