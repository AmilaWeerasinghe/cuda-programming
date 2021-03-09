#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <time.h>
#include <stdio.h>

#define b 16

//used to define the boz around the pixel
#define MASKSIZE 15

//extracted from given sapmple
#define INPUTFILE "image.jpg"
#define OUTPUTFILE "out.jpg"

using namespace cv;

#ifndef CV_LOAD_IMAGE_COLOR
        #define CV_LOAD_IMAGE_COLOR IMREAD_COLOR
#endif


//image blurring function (CU version left untouched )
void blur(unsigned char *outImage,unsigned char *inImage,int rows,int cols){
	int i,j,k,m;
	
	//go through each row
	for(i=0;i<rows;i++){
		//go through each column
		for(j=0;j<cols;j++){
				
				//average the color values of nearby pixels that falls in the mask to calculate the blurred pixel
				
				int sum_blue=0;
				int sum_green=0;
				int sum_red=0;
				
				//go through each pixel inside the mask
				for(k=i-MASKSIZE/2; k<i+MASKSIZE/2+1; k++){
					for(m=j-MASKSIZE/2; m<j+MASKSIZE/2+1; m++){
						
						//prevent accessing out of bound pixels
						if(k>=0 && k<rows && m>=0 && m<cols){
							//get the sum of  corresponding pixels
							sum_blue+=inImage[3*(k*cols+m)];
							sum_green+=inImage[3*(k*cols+m)+1];
							sum_red+=inImage[3*(k*cols+m)+2];
						}
					}
				}
				
				//colour value of output image's pixel
				outImage[3*(i*cols+j)]=(unsigned char)(sum_blue/(MASKSIZE*MASKSIZE));
				outImage[3*(i*cols+j)+1]=(unsigned char)(sum_green/(MASKSIZE*MASKSIZE));
				outImage[3*(i*cols+j)+2]=(unsigned char)(sum_red/(MASKSIZE*MASKSIZE));

		}
	}
}

/*//kernel function
__global__ void blurGPU(unsigned char *outImage,unsigned char *inImage,int rows,int cols){
	int k,m;
	
	//here instread of iterating though loop
	//assign tread for reach element in  the output matrix

	//write the kernel in point of a single pixel

	//int i=blockIdx.x*cols+threadIdx.x;
	//int j=blockIdx.y*rows+threadIdx.y;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;


				
				//average the color values of nearby pixels that falls in the mask to calculate the blurred pixel
				
				int sum_blue=0;
				int sum_green=0;
				int sum_red=0;
				
				//go through each pixel inside the mask
				for(k=i-MASKSIZE/2; k<i+MASKSIZE/2+1; k++){
					for(m=j-MASKSIZE/2; m<j+MASKSIZE/2+1; m++){
						
						//prevent accessing out of bound pixels
						if(k>=0 && k<rows && m>=0 && m<cols){
							//get the sum of  corresponding pixels
							sum_blue+=inImage[3*(k*cols+m)];
							sum_green+=inImage[3*(k*cols+m)+1];
							sum_red+=inImage[3*(k*cols+m)+2];
						}
					}
				}
				
				//colour value of output image's pixel
				outImage[3*(i*cols+j)]=(unsigned char)(sum_blue/(MASKSIZE*MASKSIZE));
				outImage[3*(i*cols+j)+1]=(unsigned char)(sum_green/(MASKSIZE*MASKSIZE));
				outImage[3*(i*cols+j)+2]=(unsigned char)(sum_red/(MASKSIZE*MASKSIZE));

	
}*/


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

//new kernel for gpu
__global__
void blurKernel_GPU(unsigned char * in, unsigned char * out, int w, int h) {
	//here instread of iterating though loop
	//assign tread for reach element in  the output matrix

	//write the kernel in point of a single pixel

	//int i=blockIdx.x*cols+threadIdx.x;
	//int j=blockIdx.y*rows+threadIdx.y;
           int Col = blockIdx.x * blockDim.x + threadIdx.x;
		   int Row = blockIdx.y * blockDim.y + threadIdx.y;
		   if (Col < w && Row < h) {
			   int pixVal = 0;
			   int pixels = 0;
			   // Get the average of the surrounding 2xMASKSIZE x 2xMASKSIZE box
			   for(int blurRow = -MASKSIZE; blurRow < MASKSIZE+1; ++blurRow) {
				   for(int blurCol = -MASKSIZE; blurCol < MASKSIZE+1; ++blurCol) {
					   int curRow = Row + blurRow;
					   int curCol = Col + blurCol;
					   // Verify we have a valid image pixel
					   if(curRow > -1 && curRow < h && curCol > -1 && curCol < w) {
						   pixVal += in[curRow * w + curCol];
						   pixels++; // accumilated total
						}
					}
				}
				// Write our new pixel value out
				out[Row * w + Col] = (unsigned char)(pixVal / pixels);
			}
		}


int main(){
	
	//space for input image
    Mat inImage; 
	// Read the input file	
    inImage = imread(INPUTFILE, CV_LOAD_IMAGE_COLOR);   	
	// Check for invalid input
    if(!inImage.data ){
        fprintf(stderr,"Could not open or find the image");
        return -1;
    }

	//space for output image
	Mat outImage(inImage.rows, inImage.cols, CV_8UC3, Scalar(0, 0, 0));	
	/*The 8U means the 8-bit Unsigned integer, C3 means 3 Channels for RGB color, and Scalar(0, 0, 0) is the initial value for each pixel. */
	
	/*clock_t start = clock();
	//call blurring function
	blur(outImage.data,inImage.data,inImage.rows,inImage.cols);
	clock_t stop = clock();
	
	//write the output image
    imwrite(OUTPUTFILE, outImage );
	
	//calculate the time taken and print to stderr
	double elapsedtime = (stop-start)/(double)CLOCKS_PER_SEC;
	fprintf(stderr,"Elapsed time for operation on CPU is %1.5f seconds \n",elapsedtime);	*/


	//Read teh imge first
	Mat Input_Image=imread(INPUTFILE);
    //check the size of image
	fprintf(stderr,"Image Height is %d, Width is %d and Channels is %d\n",Input_Image.rows,Input_Image.cols,Input_Image.channels());

	//now i have the imge

	//space for output image
	Mat Output_Image(Input_Image.rows, Input_Image.cols, CV_8UC3, Scalar(0, 0, 0));	
    
   
	//gpu varaible
	unsigned char *Dev_Input_Image;//
	unsigned char* Dev_Output_Image;
    int width = inImage.rows;
    int height = inImage.cols;
	int NumChannels=inImage.channels();//threee channels avaiable
    //Mat outImage(width, height, CV_8UC3, Scalar(0, 0, 0));
    const size_t size = sizeof(unsigned char)*width*height*NumChannels;

	//allocate mem size of input img
	//cudaMalloc((void**)&Dev_Input_Image,Input_Image.rows*Input_Image.cols*Input_Image.channels());
    

	//allocate mem size of input img

	// Allocate memory and copy the source image to CUDA memory
    cudaMalloc((void **)&Dev_Input_Image, size); checkCudaError();
    cudaMalloc((void **)& Dev_Output_Image, size); checkCudaError();
    cudaMemcpy(Dev_Input_Image, inImage.data, size, cudaMemcpyHostToDevice); checkCudaError();

	//cpoy data from cpu to GPU
	//cudaMemcpy(Dev_Input_Image,Input_Image,Input_Image.rows*Input_Image.cols*Input_Image.channels(),cudaMemcpyHostToDevice);
    

	//define dimnsions
	//define num of blcks and therads per block
	dim3 threadsPerBlock(b,b);
	//dim3 grid_Image(width,height);
	dim3 numBlocks(ceil(height/(float)b),ceil(width/(float)b));

    //start time
	cudaEvent_t startkernel2,stopkernel2;
	float elapsedtimekernel2;
	cudaEventCreate(&startkernel2);
	cudaEventRecord(startkernel2,0);



	//pass the imge to gpu function
	blurKernel_GPU<<<numBlocks,threadsPerBlock>>>( Dev_Output_Image,Dev_Input_Image,width,height);
	//dim3 blockDims(512,1,1);
    //dim3 gridDims((unsigned int) ceil((double)(width*height*3/blockDims.x)), 1, 1 );
	//blurGPU<<<gridDims,blockDims>>>( Dev_Output_Image,Dev_Input_Image,width,height);

	cudaDeviceSynchronize();
	checkCudaError();

	//end measuring time for cuda kernel
	cudaEventCreate(&stopkernel2);
	cudaEventRecord(stopkernel2,0);
	cudaEventSynchronize(stopkernel2);
	cudaEventElapsedTime(&elapsedtimekernel2,startkernel2,stopkernel2);
	
     

	//copy the results from gpu to cpu
	cudaMemcpy(outImage.data,  Dev_Output_Image, size, cudaMemcpyDeviceToHost); checkCudaError();

	imwrite(OUTPUTFILE, outImage );

	//freemem
	cudaFree( Dev_Output_Image);
    cudaFree(Dev_Input_Image);

	fprintf(stderr,"Time spent for CUDA kernel is %1.5f seconds\n",elapsedtimekernel2/(float)1000);

	//..........given CPU code......................................................................////
	clock_t start = clock();
	//call blurring function
	blur(outImage.data,inImage.data,inImage.rows,inImage.cols);
	clock_t stop = clock();
	
	//write the output image
    imwrite(OUTPUTFILE, outImage );
	
	//calculate the time taken and print to stderr
	double elapsedtime = (stop-start)/(double)CLOCKS_PER_SEC;
	fprintf(stderr,"Elapsed time for operation on CPU is %1.5f seconds \n",elapsedtime);
	
    return 0;
}