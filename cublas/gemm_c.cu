#include <stdio.h>
#include "cublas_v2.h"

#define m 6 // a - mxk matrix
#define n 4 // b - kxn matrix
#define k 5 // c - mxn matrix

int main(void) {
    cublasHandle_t handle; // CUBLAS context
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
    for(i=0;i<m*n;i++){ c[i] = (float)val++; }
    printf ("c:\n");
    ind=0;
    for (i=0;i<m;i++){
        for (j=0;j<n;j++){
            printf("%5.0f",c[ind++]);
        }
        printf ("\n");
    }
    cublasCreate(&handle); // initialize CUBLAS context
    float al=1.0f; // al =1
    float bet=1.0f; // bet =1
    // matrix - matrix multiplication : c = al*a*b + bet *c
    // a -mxk matrix , b -kxn matrix , c -mxn matrix ;
    // al, bet - scalars
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&al,a,m,b,k,&bet,c,m);
    cudaDeviceSynchronize();
    printf ("c after Sgemm :\n");
    for(i=0;i<m;i++){
        for(j=0;j<n;j++){
            printf("%7.0f",c[i*n+j]);
        }
        printf("\n");
    }
    cudaFree(a); // free memory
    cudaFree(b); // free memory
    cudaFree(c); // free memory
    cublasDestroy(handle); // destroy CUBLAS context
    return EXIT_SUCCESS ;
}
