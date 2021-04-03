//E 15 385
//Weerasinghe S.P.A.P.E

#include <sys/time.h>
#include <stdio.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <fstream>

//to calculate total parallel executions

using namespace std;

//error checking function given by Dr Hasindu
#define checkCudaError() { gpuAssert(__FILE__, __LINE__); }

static inline void gpuAssert(const char *file, int line){
	cudaError_t code = cudaGetLastError();
	if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s \n in file : %s line number : %d", cudaGetErrorString(code), file, line);
        exit(1);
   }
}



//define the macros
#define WIDTH 1000 /* Width of mandelbrot set Matrix Image */
#define HEIGHT 800 /* Height of mandelbrot set Matrix Image */
//The maximum number of times  that is tried to check whether infinity is reached.
#define MAXN 3000
//The value on the real axis which maps to the left most pixel of the image
#define XMIN -2.0
//The value of the real axis which maps to the right most pixel of the image
#define XMAX 1
//The value in the imaginary axis which maps to the top pixels of the image
#define YMIN -1.25
//The value in the imaginary axis which maps to the bottom pixels of the image
#define YMAX 1.25
//The value that we consider as infinity.
#define INF 4


#define MaxRGB 256 //Max RGB value


//bloack size from occupancy calculator results
#define Block_Size 32

//It it better the pass the RGB values into the 
//kernel as a data structure  than just passing arrays
//basically a data typr fpr RGB
typedef struct {
	unsigned int red;
	unsigned int green;
	unsigned int blue;
} pixel;

//What is searching for is a mandelBrot 
//here I defined a DS so it can be used in without re sizing
//basically  a data typr fpr mandel brot
typedef struct {
	pixel* image;
	unsigned int width;
	unsigned int height;
} Mandelbrot;

//Kenerl function 
__global__ void mandelbrotKernel(Mandelbrot mandelbrot, double* cr, double* ci) {
	//Here the shared memory is defines
	//reduces the global mem access
	__shared__ double imagi_axis[HEIGHT];
	__shared__ double real_axis[WIDTH];
	// Row major convention bases
	//Row index to access to the rows  
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	// Col index to access to the cols
	int col = blockIdx.x * blockDim.x + threadIdx.x;


	//check the dimension of the matrix
	if (row > mandelbrot.height || col > mandelbrot.width) return;
    //index of thread , each thread is assingned to compute a single pixels RGB
	int index = row * mandelbrot.width + col;

	//Store  to the shared memeory
	imagi_axis[row] = ci[row];
	real_axis[col] = cr[col];

	int i = 0;
	//Create variables to store the z Real values and Z imaginari values
	double zr = 0.0;
	double zi = 0.0;

	//How many times loop to find if number is increasing to infinite
	const int maxIterations = 3000;

	//check till maximum iterations
	while (i < maxIterations && zr * zr + zi * zi < 4.0) {
		double fz = zr * zr - zi * zi + real_axis[col];
		zi = 2.0 * zr * zi + imagi_axis[row];
		zr = fz;
		i++;
	}//upto this line sample codes isInMandelbrot is included

	//depend on the i value assign colors(same colors given in the sample code)
	int r, g, b;
	if(i == maxIterations){
		r=0;
		g=0;
		b=0;
	}else{
		//Colouring the mandelbrot set image
	b=((i + 234) % 7 * (255/7));
	r=((i+10)%256);
	g=((i+100) % 9 * (255/9));
	}

	//Update the array of pixel
	mandelbrot.image[index].red = r;
	mandelbrot.image[index].green = g;
	mandelbrot.image[index].blue = b;
}

//Fill the c values
int FillTheCVals(double* c, int state, double beginRange, double endRange, double minVal, double maxVal) {
	if (state < endRange) {
		c[state] = ((state - beginRange) / (endRange - beginRange))*(maxVal - minVal) + minVal;
		return FillTheCVals(c, state + 1, beginRange, endRange, minVal, maxVal);
	}
	else {
		return 0;
	}
}

//later defined
void mandelbrotSetCUDA(Mandelbrot mandelbrot, double* cr, double* ci);

int main(int argc, char* argv[])
{
	

	//timing 
	// Start measuring time
    struct timeval begin, end;
    gettimeofday(&begin, 0);

	//Set default values to 0
	unsigned int width = 0;
	unsigned int height = 0;
	unsigned int maxRGB = 0;
		width = WIDTH; //Set Width value
		height = HEIGHT; //Set Height value
		maxRGB = MaxRGB; //Set MaxRGB value


	// mandlerbort set image an array of rgb values
	Mandelbrot mandelbrot;

	//Create arrays that we will store the range values of real numbers and imaginari
	double* cr;
	double* ci;

	//Set the range of the mandelbrot set for c Real number and Imaginari values (Zoom in, Zoom out)
	double minValR = XMIN;
	double maxValR = XMAX;
	double minValI = YMIN;
	double maxValI = YMAX;

	size_t size;

	//Set width and height to out mandelbrot set
	mandelbrot.width = width;
	mandelbrot.height = height;

	//Dynamic allocate memory space for the size of the image on host
	size = width * height * sizeof(pixel);
	mandelbrot.image = (pixel*)malloc(size);

	//c allocate memory space 
	size = width * sizeof(double);
	cr = (double*)malloc(size);
	// allocate memory space 
	size = height * sizeof(double);
	ci = (double*)malloc(size);

	//Fill the c Values
	FillTheCVals(cr, 0, 0, width, minValR, maxValR);
	FillTheCVals(ci, 0, 0, height, minValI, maxValI);
	
	//for easiness and clear code 

    //all cuda stuffs are allocated im the mandelbrotWith is called
	mandelbrotSetCUDA(mandelbrot, cr, ci);

	// Stop measuring time and calculate the elapsed time
    gettimeofday(&end, 0);
    long seconds = end.tv_sec - begin.tv_sec;
    long microseconds = end.tv_usec - begin.tv_usec;
    double elapsed = seconds + microseconds*1e-6;
    

		printf("Output PPM file\n");

		//Create a PPM image file
		ofstream fout("cuda_image.ppm");
		//Set it to be a PPM file
		fout << "P3" << endl;
		//Set the Dimensions
		fout << mandelbrot.width << " " << mandelbrot.height << endl;
		//Max RGB Value
		fout << maxRGB << endl;

		//ColorImage
		for (int h = 0; h < height; h++) {
			for (int w = 0; w < width; w += 2) {
				int index = h * width + w;
				fout << mandelbrot.image[index].red << " " << mandelbrot.image[index].green << " " << mandelbrot.image[index].blue << " ";
				fout << mandelbrot.image[index + 1].red << " " << mandelbrot.image[index + 1].green << " " << mandelbrot.image[index + 1].blue << " ";
			}
			fout << endl;
		}
		fout.close();
		
	printf("Done!!CODE ran successfully\n");
	printf("Time measured: %.3f seconds.\n", elapsed);
	return 0;
}



void mandelbrotSetCUDA(Mandelbrot mandelbrot, double* cr, double* ci)
{

	//Store width and height
	unsigned int width = mandelbrot.width;
	unsigned int height = mandelbrot.height;

	//Create  mandelbrot which is gloabl
	Mandelbrot mandelbrot_d;
	//Set width and height to the mandelboer instance
	mandelbrot_d.width = width;
	mandelbrot_d.height = height;
	//madelbrot set size memory needed to alocate memory of the device
	size_t  mandlebortSize = width * height * sizeof(pixel);
	//Dynamic allocate memory space for the mandelbrot instance on device
	cudaMalloc((void **)&mandelbrot_d.image, mandlebortSize);checkCudaError();

	//Create cr vector on device
	double* cr_d;
	//space neeeded
	size_t CRealSize = width * sizeof(double);
	//allocate memory 
	cudaMalloc((void**)&cr_d, CRealSize);checkCudaError();

	//Create cr and ci to store the c value of our mandelrbort set on device
	double* ci_d;
	//Space neeeded
	size_t  CImagSize = height * sizeof(double);
	//allocate memory space 
	cudaMalloc((void**)&ci_d, CImagSize);checkCudaError();

	//copy from CPU to GPU
	cudaMemcpy(cr_d, cr, CRealSize, cudaMemcpyHostToDevice);checkCudaError();
	cudaMemcpy(ci_d, ci, CImagSize, cudaMemcpyHostToDevice);checkCudaError();
	

	
	//dimensions
	dim3 threadsPerBlock(Block_Size,Block_Size,1);
	dim3 numBlocks(ceil(WIDTH/(float)Block_Size),ceil(HEIGHT/(float)Block_Size),1);
	
	//call kernel
	mandelbrotKernel << <numBlocks, threadsPerBlock >> > (mandelbrot_d, cr_d, ci_d);
	cudaDeviceSynchronize();
	checkCudaError();

	printf("Kernel success\n");

	//Copy results from GPU to CPU
	cudaMemcpy(mandelbrot.image, mandelbrot_d.image, mandlebortSize, cudaMemcpyDeviceToHost);checkCudaError();
   //free mem 
	cudaFree(mandelbrot_d.image);
	cudaFree(cr_d);
	cudaFree(ci_d);

	
}