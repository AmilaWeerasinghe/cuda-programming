#include <stdio.h>
#include <stdlib.h>

#include "omp.h"
//size of the list
#define SIZE 10000
//define number of threads
#define numOfThreads 8



//time fr omp
double wtime;



//mergesort baiscally parallise the left and right subarrys into section

void merge(int a[],int i1,int j1,int i2,int j2)
{
    int temp[SIZE];    //array used for merging

    int i,j,k;
    i=i1;   
    j=i2;   
    k=0;
    
    while(i<=j1 && j<=j2)    //while elements in both lists
    {
        if(a[i]<a[j])
            temp[k++]=a[i++];
        else
            temp[k++]=a[j++];
    }
    
    while(i<=j1)    //copy remaining elements of the first list
        temp[k++]=a[i++];
        
    while(j<=j2)    //copy remaining elements of the second list
        temp[k++]=a[j++];
        
    //Transfer elements from temp[] back to a[]
    for(i=i1,j=0;i<=j2;i++,j++)
        a[i]=temp[j];
}
//merge sort function
void mergesort(int a[],int i,int j)
{
    int mid;
        
    if(i<j)
    {
        mid=(i+j)/2;
        
        #pragma omp parallel sections num_threads(numOfThreads)
        {

            #pragma omp section
            {
                mergesort(a,i,mid);        //left sub array to be done in one parallel section
            }

            #pragma omp section
            {
                mergesort(a,mid+1,j);    //right sub array to be done in one parallel section
            }
        }

        merge(a,i,mid,mid+1,j);    //merge the sorted sub arrays together
    }
}
 



int main()
{
    int *a;//array to be soreted
    int  Size_of_array;//size of the array
    int i;
    Size_of_array=SIZE;
     
     //set the num of threads
    int num_of_threads=numOfThreads;
    //omp_set_num_threads(num_of_threads);
    


   a = (int *)malloc(sizeof(int) * Size_of_array);
    for(i=0;i<Size_of_array;i++){
         a[i]=SIZE-i;
        
    }
       
     //change the timing with omp get time
    wtime = omp_get_wtime ( );
	    
    mergesort(a, 0, Size_of_array-1);

     wtime = omp_get_wtime ( ) - wtime;
    
    printf("\nSorted array :\n");
    for(i=0;i<Size_of_array;i++){
         printf("%d ",a[i]);
    }
       


    	   //print timr
            printf("\n\n");
	 printf ( "  Time = %g seconds.\n", wtime );    
    
    return 0;
}
 
