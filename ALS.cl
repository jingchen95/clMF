void choldc1(int n, __global float* a, __global float* p) {
	int base = get_group_id(0) * n * n;
	unsigned i, j;
	int k;
	float sum;
	for (i = 0; i < n; ++i) {
		for (j = i; j < n; ++j) {
			//sum = a[i][j];
			sum =a[base + i * n + j];
			for (k = i - 1; k >= 0; --k) {
				//sum -= a[i][k] * a[j][k];
				sum -= a[base + i * n + k] * a[base + j * n + k];
			}
			if (i == j) {
				if (sum <= 0) {
					printf(" a is not positive definite!\n");
				}
				p[i] = sqrt(sum);
			}
			else {
				//a[j][i] = sum / p[i];
				a[base + j * n + i] = sum / p[i];
			}
		}
	}
}

void choldcsl(int n, __global float* A, __global float *tp) {
	unsigned i, j, k; double sum;
	int base = get_group_id(0) * n * n;
	__global float* p;
	//p = (float *)malloc(n * sizeof(float));
	int gid = get_group_id(0);
	p = &(tp[gid*n]);
	choldc1(n, A, p);
	for (i = 0; i < n; ++i) {
		//A[i][i] = 1 / p[i];
		A[base + i * n + i] = 1 / p[i];
		for (j = i + 1; j < n; ++j) {
			sum = 0;
			for (k = i; k < j; ++k){
				//sum -= A[j][k] * A[k][i];
				sum -= A[base + j * n + k] * A[base + k * n + i];
			}
			//A[j][i] = sum / p[j];
			A[base + j * n + i] = sum / p[j];
		}
	}
	//free(p);
}

void inverseMatrix_CholeskyMethod(int n, __global float* A, __global float *p) {
	int base = get_group_id(0) * n * n;
	unsigned i, j, k;
	choldcsl(n, A, p);
	//vecIndex = (i * 3) + j; to ontain index from vector if needed.
	for (i = 0; i < n; ++i) {
		for (j = i + 1; j < n; ++j) {
			//A[i][j] = 0.0;
			A[base + i * n + j] = 0.0;
		}
	}
	for (i = 0; i < n; i++) {
		//A[i][i] *= A[i][i];
		A[base + i * n + i] *= A[base + i * n + i];
		for (k = i + 1; k < n; ++k) {
			//A[i][i] += A[k][i] * A[k][i];
			A[base + i * n + i] += A[base + k * n + i]* A[base + k * n + i];
		}
		for (j = i + 1; j < n; ++j) {
			for (k = j; k < n; ++k) {
				//A[i][j] += A[k][i] * A[k][j];
				A[base + i * n + j] += A[base + k * n + i]* A[base + k * n + j];
			}
		}
	}
	for (i = 0; i < n; ++i) {
		for (j = 0; j < i; ++j) {
			//A[i][j] = A[j][i];
			A[base + i * n + j] = A[base + j * n + i];
		}
	}
}
/*
void Mt_byM_multiply_k(int i, int j,  __global float *H, __global float *Result, const long ptr, __global const unsigned *idx){
	int base = get_group_id(0)*j*j;
	int ss = get_local_id(0);
	int gg = get_local_size(0);
	float SUM[100] = {0};
	for (unsigned I = ss; I < j; I+=gg){
		for (unsigned J = I; J < j; ++J) {
			for (unsigned K = 0; K < i; ++K){
				unsigned offset = idx[ptr + K] * j;
				SUM[I*j+J] += H[offset + I] * H[offset + J];
			}
			Result[base+(J*j)+I] = SUM[I*j+J];
			Result[base+(I*j)+J] = SUM[I*j+J];
		}
	}
}
*/
void Mt_byM_multiply_k(int i, int j,  __global float *H, __global float *Result, const long ptr, __global const unsigned *idx){
	int base = get_group_id(0)*j*j;
	int ss = get_local_id(0);
	int gg = get_local_size(0);
	float SUM[100] = {0};
	int n=(i-1)/8+1;
	int8 offset;
	float8 cj;
	for (unsigned I = ss; I < j; I+=gg){
		for (unsigned J = I; J < j; ++J) {
            cj=(float8)(0);
			for (unsigned K = 0; K < n-1; ++K){
				offset = (int8)(idx[ptr + 0 + K * 8] * j,idx[ptr + 1 + K * 8] * j,idx[ptr + 2 + K * 8] * j,idx[ptr + 3 + K * 8] * j,idx[ptr + 4 + K * 8] * j,idx[ptr + 5 + K * 8] * j
                     ,idx[ptr + 6 + K * 8] * j,idx[ptr + 7 + K * 8] * j);
                cj.s0 += H[offset.s0+I]*H[offset.s0+J];
                cj.s1 += H[offset.s1+I]*H[offset.s1+J];
                cj.s2 += H[offset.s2+I]*H[offset.s2+J];
                cj.s3 += H[offset.s3+I]*H[offset.s3+J];
                cj.s4 += H[offset.s4+I]*H[offset.s4+J];
                cj.s5 += H[offset.s5+I]*H[offset.s5+J];
                cj.s6 += H[offset.s6+I]*H[offset.s6+J];
                cj.s7 += H[offset.s7+I]*H[offset.s7+J];

				SUM[I*j+J] = cj.s0+cj.s1+cj.s2+cj.s3+cj.s4+cj.s5+cj.s6+cj.s7;
			}
			for (unsigned K = 8*(n-1); K < i; ++K){
				unsigned offset = idx[ptr + K] * j;
				SUM[I*j+J] += H[offset + I] * H[offset + J];
			}
			Result[base+(J*j)+I] = SUM[I*j+J];
			Result[base+(I*j)+J] = SUM[I*j+J];
		}
	}
}

__kernel void updateW_overH_kernel( const ulong rows,
                                   __global const long *row_ptr,
                                   __global const unsigned *col_idx,
                      __global const unsigned *colMajored_sparse_idx,
                                   __global const float *val,
                                   const float lambda,
                                   const uint k,
                                   __global float *W,
                                   __global float *H,
                                   __global float *p,
                                   __global float *subVector,
                                   __global float *subMatrix)
{
   int i = get_global_id(0);
   int j = get_global_size(0);
   int s = get_local_id(0);
   int g = get_local_size(0);
   int a = get_group_id(0);
   int v = get_num_groups(0);
   int base = a * k * k;
   int baseV = a * k;
   for (int Rw = a; Rw < rows; Rw += v){
		__global float *Wr = &W[Rw*k];
		unsigned omegaSize = row_ptr[Rw + 1] - row_ptr[Rw];
		if (omegaSize>0){
            		Mt_byM_multiply_k(omegaSize, k, H, subMatrix, row_ptr[Rw], col_idx);
            barrier(CLK_GLOBAL_MEM_FENCE);
            for (unsigned c = s; c < k; c+=g){
                subMatrix[base + c * k + c] += lambda;
            }
          barrier(CLK_GLOBAL_MEM_FENCE);
	    if(s==0)
	    {
	           inverseMatrix_CholeskyMethod(k, subMatrix, p);
	     }
	    barrier(CLK_GLOBAL_MEM_FENCE);
	    for (unsigned c = s; c < k; c+=g){
                subVector[baseV + c] = 0;
                for (unsigned idx = row_ptr[Rw]; idx < row_ptr[Rw + 1]; ++idx){
                    unsigned idx2 = colMajored_sparse_idx[idx];
                    subVector[baseV + c] += val[idx2] * H[(col_idx[idx] * k) + c];
                }
            }
            barrier(CLK_GLOBAL_MEM_FENCE);

            for (unsigned c = s; c < k; c+=g){
            	     Wr[c]=0.0f;
	             for(unsigned subVid=0;subVid<k;++subVid){
			     Wr[c] +=subVector[baseV+subVid]*subMatrix[base + c * k+subVid];
                }
            }

            barrier(CLK_GLOBAL_MEM_FENCE);

		}
		else{
			for (unsigned c = 0; c < k; ++c){
				Wr[c] = 0.0f;
			}
		}
   }
}

__kernel void updateH_overW_kernel( const ulong cols,
                                   __global const long *col_ptr,
                                   __global const unsigned *row_idx,
                                   __global const float *val,
                                   const float lambda,
                                   const uint k,
                                   __global float *W,
                                   __global float *H,
                                   __global float *p,
                                   __global float *subVector,
                                   __global float *subMatrix)
{
   int i = get_global_id(0);
   int j = get_global_size(0);
   int s = get_local_id(0);
   int g = get_local_size(0);
   int a = get_group_id(0);
   int v = get_num_groups(0);
   int base = a * k * k;
   int baseV = a * k;
   for (int Rh = a; Rh < cols; Rh +=v){
		__global float *Hr = &H[Rh*k];
		unsigned omegaSize = col_ptr[Rh + 1] - col_ptr[Rh];
		if (omegaSize>0){
            Mt_byM_multiply_k(omegaSize, k, W, subMatrix, col_ptr[Rh], row_idx);
            barrier(CLK_GLOBAL_MEM_FENCE);

            for (unsigned c = s; c < k; c+=g){
                subMatrix[base + c * k + c] += lambda;
            }
            barrier(CLK_GLOBAL_MEM_FENCE);
            if(s==0){
                inverseMatrix_CholeskyMethod(k, subMatrix, p);
            }
  	    barrier(CLK_GLOBAL_MEM_FENCE);
            for (unsigned c = s; c < k; c+=g){
                subVector[baseV + c] = 0;
				for (unsigned idx = col_ptr[Rh]; idx < col_ptr[Rh + 1]; ++idx){
                    subVector[baseV + c] += val[idx] * W[(row_idx[idx] * k) + c];
				}
			}
			barrier(CLK_GLOBAL_MEM_FENCE);
            for (unsigned c = s; c < k; c+=g){
				Hr[c] = 0;
				for (unsigned subVid = 0; subVid < k; ++subVid){
					Hr[c] += subVector[baseV + subVid] * subMatrix[base + c * k + subVid];
				}
			}
			barrier(CLK_GLOBAL_MEM_FENCE);
		}
		else{
			for (unsigned c = 0; c < k; ++c){
				Hr[c] = 0.0f;
			}
		}
      }
}

