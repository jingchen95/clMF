#include <CL/cl.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include "cj.h"
#include "tools.h"
#include "util.h"

using namespace std;

double gettime()
{
    struct timeval t;
    gettimeofday(&t,NULL);
    return t.tv_sec+t.tv_usec*1e-6;
    return 1;
}

int main(int argc, char* argv[])
{
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

	puts("ALS-OpenCL-Parallel Programming Starts!");

	// Input file
	char input_file_name[1024]="../dataset/ml-10M100K";
    	parameter param;
	smat_t R;
	mat_t W_c, H_c;
	bool with_weights=false;

	// Load matrix => CSR
	load(input_file_name, R, true, with_weights);

	// Initial matrix W and H
	initial(W_c, R.rows, param.k);
	initial(H_c, R.cols, param.k);
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
	float *val_t = R.val_t;

	float *submatrix;
	submatrix=(float *)malloc(nBlocks*k*k*sizeof(float));
	for(int i=0; i < nBlocks * k * k ;i++)
	{
		submatrix[i] = 0.0;
	}

	float *subvector;
        subvector=(float *)malloc(nBlocks * k * sizeof(float));
        for(int i=0; i < nBlocks * k ;i++)
        {
                subvector[i] = 0.0;
        }

	float *W,*H;
    	W=(float *)malloc(k*R.rows*sizeof(float));
	H=(float *)malloc(k*R.cols*sizeof(float));

	size_t nbits_W_ = R.rows*k*sizeof(float);
	size_t nbits_H_ = R.cols*k*sizeof(float);
	int indexPosition = 0;
	for (int i = 0; i <rows; ++i){
		for (int j = 0; j < k; ++j){
			W[indexPosition] = W_c[i][j];
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
	cl_mem    valBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR ,R.nbits_val, (void *)val_t, NULL);
	cl_mem    WBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, nbits_W_,(void *)W, NULL);
	cl_mem    HBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, nbits_H_,(void *)H, NULL);
	cl_mem    pBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, nBlocks*nThreadsPerBlock*k*sizeof(float),NULL, NULL);
	cl_mem    subVecBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, nBlocks * k * sizeof(float), (void *)subvector, NULL);
	cl_mem    subMatBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, nBlocks * k * k * sizeof(float), (void *)submatrix, NULL);

	cl_int err;
	cl_kernel Mt_byM_multiply_k_kernel = clCreateKernel(program,"Mt_byM_multiply_k", &err);
	cl_kernel updateWOverH_kernel = clCreateKernel(program,"updateW_overH_kernel", &err);
	if(err!=CL_SUCCESS)
    	{
            printf("err: %s\n", get_error_string(err));
            return 1;
    	}

	status = clSetKernelArg(Mt_byM_multiply_k_kernel, 0, sizeof(long), &rows);
	status = clSetKernelArg(Mt_byM_multiply_k_kernel, 1, sizeof(cl_mem), (void *)&row_ptrBuffer);
	status = clSetKernelArg(Mt_byM_multiply_k_kernel, 2, sizeof(int), &k);
	status = clSetKernelArg(Mt_byM_multiply_k_kernel, 3, sizeof(cl_mem), (void *)&HBuffer);
	status = clSetKernelArg(Mt_byM_multiply_k_kernel, 4, sizeof(cl_mem), (void *)&subMatBuffer);
	status = clSetKernelArg(Mt_byM_multiply_k_kernel, 5, sizeof(float), &lambda);
	status = clSetKernelArg(Mt_byM_multiply_k_kernel, 6, sizeof(cl_mem), (void *)&col_idxBuffer);
	status = clSetKernelArg(Mt_byM_multiply_k_kernel, 7, sizeof(cl_mem), (void *)&colMajored_sparse_idxBuffer);
	status = clSetKernelArg(Mt_byM_multiply_k_kernel, 8, sizeof(cl_mem), (void *)&valBuffer);
	status = clSetKernelArg(Mt_byM_multiply_k_kernel, 9, sizeof(cl_mem), (void *)&subVecBuffer);

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

	double t1 = gettime();
	for(unsigned int ite=0;ite<1;ite++)
    	{
		size_t global_work_size[3] = {30, 30, nBlocks};
		size_t local_work_size[3] = {nThreadsPerBlock, nThreadsPerBlock, 1};
		
		cl_event enentPoint2;
		status = clEnqueueNDRangeKernel(commandQueue, Mt_byM_multiply_k_kernel, 3, NULL, global_work_size ,local_work_size, 0, NULL, &enentPoint2);
		clWaitForEvents(1,&enentPoint2);
		clReleaseEvent(enentPoint2);
	
		size_t global_size[1] = {32 * nBlocks};
		size_t local_size[1] = {32};
		cl_event enentPoint;
		status = clEnqueueNDRangeKernel(commandQueue, updateWOverH_kernel, 1, NULL, global_size ,local_size, 0, NULL, &enentPoint);
		clWaitForEvents(1,&enentPoint);
		clReleaseEvent(enentPoint);
    	}
    	double t2 = gettime();
    	double deltaT = t2 - t1;
    	cout<<"CLMF Training Time:"<<deltaT<<" s.\n";

	/* Release */
	status = clReleaseKernel(Mt_byM_multiply_k_kernel);
	status = clReleaseKernel(updateWOverH_kernel);
	status = clReleaseProgram(program); 
	status = clReleaseMemObject(row_ptrBuffer);
	status = clReleaseMemObject(col_idxBuffer);
	status = clReleaseMemObject(col_ptrBuffer);
	status = clReleaseMemObject(row_idxBuffer);
	status = clReleaseMemObject(colMajored_sparse_idxBuffer);
	status = clReleaseMemObject(valBuffer);
	status = clReleaseMemObject(WBuffer);
	status = clReleaseMemObject(HBuffer);
	status = clReleaseMemObject(pBuffer);
	status = clReleaseMemObject(subMatBuffer);
	status = clReleaseMemObject(subVecBuffer);
	status = clReleaseCommandQueue(commandQueue);
	status = clReleaseContext(context);
	free(devices);
	return 0;
}
