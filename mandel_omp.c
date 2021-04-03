
//include the header 
# include <omp.h>

#include <math.h>
#include <stdio.h>
#include <time.h>



#define numOfThreadsUse 8

#define WIDTH 1000

#define HEIGHT 1000

#define XMIN -2.0

#define XMAX 1

#define YMIN -1.25

#define YMAX 1.25

#define INF 4

#define MAXN 3000

#define max(a,b) (((a)>(b))?(a):(b))

#define FILENAME "image.ppm"

double wtime;


int mandel_set[HEIGHT][WIDTH]; 

unsigned char image[WIDTH *HEIGHT * 3];

unsigned char red(int i){
	if (i==0 )
		return 0 ;
	else 
	return ((i+10)%256);
}


unsigned char blue(int i){
	if (i==0)
	return  0;
	else
	return ((i + 234) % 7 * (255/7));
}


unsigned char green(int i){
	if (i==0)
		return  0 ;
	else
		return ((i+100) % 9 * (255/9));
}
	


float transform_to_x(int x){ 
	return XMIN+x*(XMAX-XMIN)/(float)WIDTH;
}

float transform_to_y(int y){
	return YMAX-y*(YMAX-YMIN)/(float)HEIGHT;
}


int isin_mandelbrot(float realc,float imagc){
	

	int i=0;
	float realz_next=0,imagz_next=0;
	float abs=0;
	float realz=0;
	float imagz=0;
	

	while(i<MAXN && abs<INF){
		
		
		realz_next=realz*realz-imagz*imagz+realc;
		imagz_next=2*realz*imagz+imagc;
		
		
		abs=realz*realz+imagz*imagz;
		
		
		realz=realz_next;
		imagz=imagz_next;
		i++;
	}
	
	
	if (i==MAXN)
		return 0;
	
	else
		return i;
}



void plot(int blank[HEIGHT][WIDTH]){
	int x,y;
    int NumOfThreads=0;
    int MaxNumberOfThreads=0;
    //define a variable 

//create the parallel region
   //#pragma omp parallel
   {

       omp_set_num_threads(numOfThreadsUse);
        NumOfThreads= omp_get_num_threads();
        MaxNumberOfThreads=omp_get_max_threads(); 
       //apply omp for two the loops
       #pragma omp parallel for collapse(2) num_threads(numOfThreadsUse) schedule(dynamic)
	for (y=0;y<HEIGHT;y++){
		for (x=0;x<WIDTH;x++){
			blank[y][x]=isin_mandelbrot(transform_to_x(x),transform_to_y(y));
		}
	}	

   }//end of parallel region
	
   // printf ( " Number of threads used is %d Out of %d Maximum num od threads.\n", numOfThreadsUse,MaxNumberOfThreads );
    printf ( " Number of threads used is %d .\n", numOfThreadsUse );
}


void createimage() {

  	int x=0,y=0,n=0;int color;
	
	
	for (y=0;y<HEIGHT;y++){
		for(x=0;x<WIDTH;x++){
			color=mandel_set[y][x];
			image[n]=red(color);
			image[n+1]=green(color);
			image[n+2]=blue(color);
			n=n+3;
		}
	}	

}




int main(int argc, char** argv) {

//change the timing with omp get time
    wtime = omp_get_wtime ( );

	//create the mandelbrot matrix
	plot(mandel_set);
	
	createimage();
  
    wtime = omp_get_wtime ( ) - wtime;
  printf ( "  Time = %g seconds.\n", wtime );

	
  
	
    const int MaxColorComponentValue=255; 
    char *comment="# ";
        
        
    FILE * fp=fopen(FILENAME,"wb"); 
    
    
    fprintf(fp,"P6\n %s\n %d\n %d\n %d\n",comment,WIDTH,HEIGHT,MaxColorComponentValue);
    
    
    fwrite(image,1,WIDTH *HEIGHT * 3,fp);
			
    
    fclose(fp);
		
	return 0;
}
