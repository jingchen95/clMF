#include <CL/cl.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>
#include <math.h>
#include "cj.h"
#include "tools.h"
#include "pmf.h"
#include "util.h"
#include <time.h>
#include <sys/time.h>
using namespace std;

double gettime()
{
    struct timeval t;
    gettimeofday(&t,NULL);
    return t.tv_sec+t.tv_usec*1e-6;
    return 1;
}

void choldc1(int n, float** a, float* p) {
	unsigned i, j;
	int k;
	float sum;
	for (i = 0; i < n; ++i) {
		for (j = i; j < n; ++j) {
			sum = a[i][j];
			for (k = i - 1; k >= 0; --k) {
				sum -= a[i][k] * a[j][k];
			}
			if (i == j) {
				if (sum <= 0) {
					printf(" a is not positive definite!\n");
				}
				p[i] = sqrtf(sum);//float square root
			}
			else {
				a[j][i] = sum / p[i];
			}
		}
	}
}

void choldcsl(int n, float** A) {
	unsigned i, j, k; double sum;
	float* p;
	p = (float *)malloc(n * sizeof(float));
	choldc1(n, A, p);
	for (i = 0; i < n; ++i) {
		A[i][i] = 1 / p[i];
		for (j = i + 1; j < n; ++j) {
			sum = 0;
			for (k = i; k < j; ++k) {
				sum -= A[j][k] * A[k][i];
			}
			A[j][i] = sum / p[j];
		}
	}
	free(p);
}

void inverseMatrix_CholeskyMethod(int n, float** A) {
	unsigned i, j, k;
	choldcsl(n, A);
	for (i = 0; i < n; ++i) {
		for (j = i + 1; j < n; ++j) {
			A[i][j] = 0.0;
		}
	}
	for (i = 0; i < n; i++) {
		A[i][i] *= A[i][i];
		for (k = i + 1; k < n; ++k) {
			A[i][i] += A[k][i] * A[k][i];
		}
		for (j = i + 1; j < n; ++j) {
			for (k = j; k < n; ++k) {
				A[i][j] += A[k][i] * A[k][j];
			}
		}
	}
	for (i = 0; i < n; ++i) {
		for (j = 0; j < i; ++j) {
			A[i][j] = A[j][i];
		}
	}
}
//Multiply matrix M transpose by M
void Mt_byM_multiply(int i, int j, float**M, float**Result){
	float SUM;
	for (unsigned I = 0; I < j; ++I){
		for (unsigned J = I; J < j; ++J) {
			SUM = 0.0f;
			for (unsigned K = 0; K < i; ++K){
				//printf("%.3f %.3f\n", M[K][I], M[K][J]);
				SUM += M[K][I] * M[K][J];
			}
			Result[J][I] = SUM;
			Result[I][J] = SUM;
		}
	}
}
//Multiply matrix M by M transpose
void M_byMt_multiply(int i, int j, float**M, float**Result){
	float SUM;
	for (unsigned I = 0; I < i; ++I){
		for (unsigned J = 0; J < i; ++J) {
			SUM = 0.0;
			for (unsigned K = 0; K < j; ++K)
				SUM += M[I][K] * M[J][K];
			Result[I][J] = SUM;
		}
	}
}

int main(int argc, char* argv[])
{
    double t11=gettime();

	cl_int    status;
	cl_uint NumDevice;
	cl_platform_id platform;
	getPlatform(platform);
	cl_device_id *devices=getCl_device_id(platform);
	cl_context context = clCreateContext(NULL,1, devices,NULL,NULL,NULL);
	status=clGetContextInfo(context,CL_CONTEXT_NUM_DEVICES,sizeof(cl_uint),&NumDevice,NULL);
	cl_command_queue commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);

	const char *filename = "ALS.cl";
	string sourceStr;
	status = convertToString(filename, sourceStr);
	const char *source = sourceStr.c_str();
	size_t sourceSize[] = {strlen(source)};
	cl_program program = clCreateProgramWithSource(context, 1, &source, sourceSize, NULL);

	status=clBuildProgram(program, 1,devices,NULL,NULL,NULL);
	if(status!=CL_SUCCESS)
	{
		size_t length;
		clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0,NULL,&length);
		char* buffer = (char*)malloc(length+1);
		clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, length,buffer,NULL);
		printf("build info: %s\n", buffer);
	}

	puts("ALS-OpenCL-Parallel Programming: starts!");
	char input_file_name[1024]="../dataset/movielens";
    parameter param;
	smat_t R;
	mat_t W_c, H_c;
	bool with_weights=false;

    double t3 = gettime();
	load(input_file_name, R, true, with_weights);
	double t4 = gettime();
    double deltaT1 = t4 - t3;
    cout<<"Load R Time:"<<deltaT1<<" s.\n";

    FILE *test_fp=fopen("../dataset/movielens/test.ratings", "r");
	if(test_fp == NULL)
    {
        	printf("can't open output file.\n");
        	exit(1);
    }
	initial_col(W_c, R.rows, param.k);
	initial_col(H_c, R.cols, param.k);
	int k = param.k;
	float lambda = param.lambda;
	long rows = R.rows;
	long cols = R.cols;
	int nBlocks = param.nBlocks;
	int nThreadsPerBlock = param.nThreadsPerBlock;
	int maxiter = param.maxiter;
	long *col_ptr =R.col_ptr, *row_ptr = R.row_ptr;
	unsigned *row_idx = R.row_idx, *col_idx = R.col_idx;
	unsigned *colMajored_sparse_idx = R.colMajored_sparse_idx;
	float *val = R.val;

    float *submatrix;
    submatrix=(float *)malloc(k*k*sizeof(float));
    for(int i=0;i<k;i++)
    {
        for(int j=0;j<k;j++)
        {
            submatrix[i*k+j]=0.0f;
        }
    }

	float *W,*H;
    W=(float *)malloc(k*R.rows*sizeof(float));
	H=(float *)malloc(k*R.cols*sizeof(float));

	size_t nbits_W_ = R.rows*k*sizeof(float);
	size_t nbits_H_ = R.cols*k*sizeof(float);
	int indexPosition = 0;
	for (int i = 0; i <rows; ++i){
		for (int j = 0; j < k; ++j){
			W[indexPosition] = 0.0;
			++indexPosition;
		}

	}

	int indexPosition1 = 0;
	for (int i = 0; i <R.cols; ++i){
		for (int j = 0; j < k; ++j){
			H[indexPosition1] = H_c[i][j];
			++indexPosition1;
		}
	}
	cl_mem    row_ptrBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, R.nbits_row_ptr,(void *)row_ptr, NULL);
	cl_mem    col_idxBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR ,R.nbits_col_idx, (void *)col_idx, NULL);
	cl_mem    col_ptrBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, R.nbits_col_ptr,(void *)col_ptr, NULL);
	cl_mem    row_idxBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR ,R.nbits_row_idx, (void *)row_idx, NULL);
	cl_mem    colMajored_sparse_idxBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR ,R.nbits_colMajored_sparse_idx, (void *)colMajored_sparse_idx, NULL);
	cl_mem    valBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR ,R.nbits_val, (void *)val, NULL);
	cl_mem    WBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, nbits_W_,(void *)W, NULL);
	cl_mem    HBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, nbits_H_,(void *)H, NULL);

	cl_mem    pBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, nBlocks*nThreadsPerBlock*k*sizeof(float),NULL, NULL);
	cl_mem    subVecBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, nBlocks*nThreadsPerBlock*k*sizeof(float),NULL, NULL);
	cl_mem    subMatBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, nBlocks*nThreadsPerBlock*k*k*sizeof(float),NULL, NULL);
	cl_mem    subMatrixBuffer=clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, k*k*sizeof(float),(void *)submatrix, NULL);
	cl_mem    p_Buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, nBlocks*nThreadsPerBlock*k*sizeof(float),NULL, NULL);
	cl_mem    subVec_Buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, nBlocks*nThreadsPerBlock*k*sizeof(float),NULL, NULL);
	cl_mem    subMat_Buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, nBlocks*nThreadsPerBlock*k*k*sizeof(float),NULL, NULL);

	cl_int err;
	cl_kernel updateWOverH_kernel = clCreateKernel(program,"updateW_overH_kernel", &err);
	cl_kernel updateHOverW_kernel = clCreateKernel(program,"updateH_overW_kernel", &err);
	if(err!=CL_SUCCESS)
    {
            printf("err: %s\n", get_error_string(err));
            return 1;
    }

	status = clSetKernelArg(updateWOverH_kernel, 0, sizeof(long), &rows);
	status = clSetKernelArg(updateWOverH_kernel, 1, sizeof(cl_mem), (void *)&row_ptrBuffer);
	status = clSetKernelArg(updateWOverH_kernel, 2, sizeof(cl_mem), (void *)&col_idxBuffer);
	status = clSetKernelArg(updateWOverH_kernel, 3, sizeof(cl_mem), (void *)&colMajored_sparse_idxBuffer);
	status = clSetKernelArg(updateWOverH_kernel, 4, sizeof(cl_mem), (void *)&valBuffer);
	status = clSetKernelArg(updateWOverH_kernel, 5, sizeof(float), &lambda);
	status = clSetKernelArg(updateWOverH_kernel, 6, sizeof(int), &k);
	status = clSetKernelArg(updateWOverH_kernel, 7, sizeof(cl_mem), (void *)&WBuffer);
	status = clSetKernelArg(updateWOverH_kernel, 8, sizeof(cl_mem), (void *)&HBuffer);
	status = clSetKernelArg(updateWOverH_kernel, 9, sizeof(cl_mem), (void *)&pBuffer);
	status = clSetKernelArg(updateWOverH_kernel, 10, sizeof(cl_mem), (void *)&subVecBuffer);
	status = clSetKernelArg(updateWOverH_kernel, 11, sizeof(cl_mem), (void *)&subMatBuffer);
	status = clSetKernelArg(updateWOverH_kernel, 12, sizeof(cl_mem), (void *)&subMatrixBuffer);

	status = clSetKernelArg(updateHOverW_kernel, 0, sizeof(long), &cols);
	status = clSetKernelArg(updateHOverW_kernel, 1, sizeof(cl_mem), (void *)&col_ptrBuffer);
	status = clSetKernelArg(updateHOverW_kernel, 2, sizeof(cl_mem), (void *)&row_idxBuffer);
	status = clSetKernelArg(updateHOverW_kernel, 3, sizeof(cl_mem), (void *)&valBuffer);
	status = clSetKernelArg(updateHOverW_kernel, 4, sizeof(float), &lambda);
	status = clSetKernelArg(updateHOverW_kernel, 5, sizeof(int), &k);
	status = clSetKernelArg(updateHOverW_kernel, 6, sizeof(cl_mem), (void *)&WBuffer);
	status = clSetKernelArg(updateHOverW_kernel, 7, sizeof(cl_mem), (void *)&HBuffer);
	status = clSetKernelArg(updateHOverW_kernel, 8, sizeof(cl_mem), (void *)&p_Buffer);
	status = clSetKernelArg(updateHOverW_kernel, 9, sizeof(cl_mem), (void *)&subVec_Buffer);
	status = clSetKernelArg(updateHOverW_kernel, 10, sizeof(cl_mem), (void *)&subMat_Buffer);

	double t1 = gettime();
	for(unsigned int ite=0;ite<5;ite++)
    {
        size_t global_work_size[1] = {nBlocks*nThreadsPerBlock};
        size_t local_work_size[1] = {nThreadsPerBlock};

        /** update_W_Over_H */
        cl_event enentPoint;
        status = clEnqueueNDRangeKernel(commandQueue, updateWOverH_kernel, 1, NULL, global_work_size ,local_work_size, 0, NULL, &enentPoint);
        clWaitForEvents(1,&enentPoint);
        clReleaseEvent(enentPoint);
/*
        status=clEnqueueReadBuffer(commandQueue, WBuffer, CL_TRUE, 0, nbits_W_, W, 0, NULL, NULL);
        status=clEnqueueReadBuffer(commandQueue, HBuffer, CL_TRUE, 0, nbits_H_, H, 0, NULL, NULL);
        status=clEnqueueReadBuffer(commandQueue, subMatrixBuffer, CL_TRUE, 0, k*k*sizeof(float), submatrix, 0, NULL, NULL);
        cout<<"update_W_over_H   W:\n";
        for(int df=0;df<10;df++)
        {
            for(int fd=0;fd<5;fd++)
            {
                cout<<W[df*k+fd]<<"  ";
            }
            cout<<"\n";
        }

        cout<<"update_W_over_H   H:\n";
        for(int df=0;df<10;df++)
        {
            for(int fd=0;fd<5;fd++)
            {
                cout<<H[df*k+fd]<<"  ";
            }
            cout<<"\n";
        }

        cout<<"update_W_over_H   subMatrix:\n";
        for(int df=0;df<10;df++)
        {
            for(int fd=0;fd<10;fd++)
            {
                cout<<submatrix[df*k+fd]<<"  ";
            }
            cout<<"\n";
        }
*/
        /** update_H_Over_W */
        cl_event enentPoint1;
        status = clEnqueueNDRangeKernel(commandQueue, updateHOverW_kernel, 1, NULL, global_work_size , local_work_size, 0, NULL, &enentPoint1);
        clWaitForEvents(1,&enentPoint1);
        clReleaseEvent(enentPoint1);
/*	printf("ddd.\n");

        status=clEnqueueReadBuffer(commandQueue, WBuffer, CL_TRUE, 0, nbits_W_, W, 0, NULL, NULL);
        status=clEnqueueReadBuffer(commandQueue, HBuffer, CL_TRUE, 0, nbits_H_, H, 0, NULL, NULL);
        cout<<"update_H_over_W   W:\n";
        for(int df=0;df<10;df++)
        {
            for(int fd=0;fd<5;fd++)
            {
                cout<<W[df*k+fd]<<"  ";
            }
            cout<<"\n";
        }
        cout<<"update_H_over_W   H:\n";
        for(int df=0;df<10;df++)
        {
            for(int fd=0;fd<5;fd++)
            {
                cout<<H[df*k+fd]<<"  ";
            }
            cout<<"\n";
        }
*/
    }
    double t2 = gettime();
    double deltaT = t2 - t1;
    cout<<"Training Time:"<<deltaT<<" s.\n";

    status=clEnqueueReadBuffer(commandQueue, WBuffer, CL_TRUE, 0, nbits_W_, W, 0, NULL, NULL);
    status=clEnqueueReadBuffer(commandQueue, HBuffer, CL_TRUE, 0, nbits_H_, H, 0, NULL, NULL);

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            W_c[i][j] = W[i*k+j];
        }
    }
    for (int i = 0; i < cols; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            H_c[i][j] = H[i*k+j];
        }
    }

    double t5 = gettime();
	int i, j;
	double v, rmse = 0;
	size_t num_insts = 0;
	while(fscanf(test_fp, "%d %d %lf", &i, &j, &v) != EOF)
    	{
	 	double pred_v = 0;
		for(int t = 0; t < k; t++)
			pred_v += W_c[i-1][t] * H_c[j-1][t];
		num_insts ++;
		rmse += (pred_v - v)*(pred_v - v);
	}
	rmse = sqrt(rmse/num_insts);
	printf("test RMSE = %lf.\n", rmse);
	double t6 = gettime();
    double deltaT2 = t6 - t5;
    cout<<"Predict Time:"<<deltaT2<<" s.\n";
/* Release */
	status = clReleaseKernel(updateHOverW_kernel);
	status = clReleaseKernel(updateWOverH_kernel);//*Release kernel.
	status = clReleaseProgram(program);    //Release the program object.
	status = clReleaseMemObject(row_ptrBuffer);//Release mem object.
	status = clReleaseMemObject(col_idxBuffer);//Release mem object.
	status = clReleaseMemObject(col_ptrBuffer);//Release mem object.
	status = clReleaseMemObject(row_idxBuffer);
	status = clReleaseMemObject(colMajored_sparse_idxBuffer);//Release mem object.
	status = clReleaseMemObject(valBuffer);//Release mem object.
	status = clReleaseMemObject(WBuffer);
	status = clReleaseMemObject(HBuffer);
	status = clReleaseMemObject(subMatrixBuffer);
	status = clReleaseCommandQueue(commandQueue);
	status = clReleaseContext(context);

	free(devices);
	double t12=gettime();
	double ss=t12-t11;
	cout<<"total time is "<<ss<<" s.\n";
	return 0;
}
