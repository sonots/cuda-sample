#include <stdio.h>
#include "cublas_v2.h"

#define m 6 // a - mxk matrix
#define n 4 // b - kxn matrix
#define k 5 // c - mxn matrix

#define SWAP(a,b,tmp) { (tmp)=(a); (a)=(b); (b)=(tmp); }

// https://www.christophlassner.de/using-blas-from-c-with-row-major-data.html
void cublasRowMajorSgemm(float *a, float *b, float *c) {
    int i,j; // i-row valex, j-column valex
    cublasHandle_t handle; // CUBLAS context
    cublasCreate(&handle); // initialize CUBLAS context
    float al=1.0f; // al =1
    float bet=1.0f; // bet =1

    // b^T = nxk matrix
    // a^T = kxm matrix
    // c^T = nxm matrix
    // c^T = b^T * a^T

    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,n,m,k,&al,b,n,a,k,&bet,c,n);
    cudaDeviceSynchronize();
    printf ("c after Sgemm :\n");
    for(i=0;i<m;i++){
        for(j=0;j<n;j++){
            printf("%7.0f",c[i*n+j]);
        }
        printf("\n");
    }
    cublasDestroy(handle); // destroy CUBLAS context
}
int main(void) {
    int i,j, ind; // i-row valex, j-column valex
    float *a; // mxk matrix
    float *b; // kxn matrix
    float *c; // mxn matrix
    // unified memory for a,b,c
    cudaMallocManaged(&a, m*k*sizeof(cuComplex));
    cudaMallocManaged(&b, k*n*sizeof(cuComplex));
    cudaMallocManaged(&c, m*n*sizeof(cuComplex));
    // define an mxk matrix a column by column
    int val=0; // a:
    for(i=0;i<m*k;i++){ a[i] = (float)val++; }
    printf ("a:\n");
    ind=0;
    for (i=0;i<m;i++){
        for (j=0;j<k;j++){
            printf("%5.0f",a[ind++]);
        }
        printf ("\n");
    }
    // define a kxn matrix b column by column
    val=0; // b:
    for(i=0;i<k*n;i++){ b[i] = (float)val++; }
    printf ("b:\n");
    ind=0;
    for (i=0;i<k;i++){
        for (j=0;j<n;j++){
            printf("%5.0f",b[ind++]);
        }
        printf ("\n");
    }
    // define an mxn matrix c column by column
    val=0; // c:
    for(i=0;i<m*n;i++){ c[i] = (float)0; }
    printf ("c:\n");
    ind=0;
    for (i=0;i<m;i++){
        for (j=0;j<n;j++){
            printf("%5.0f",c[ind++]);
        }
        printf ("\n");
    }
    cublasRowMajorSgemm(a, b, c);
    cudaFree(a); // free memory
    cudaFree(b); // free memory
    cudaFree(c); // free memory
    return EXIT_SUCCESS ;
}
