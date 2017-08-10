#define N 90
#define th_blk 25
#define mini 50
#define block_nnz 30

__kernel void Mt_byM_multiply_k(const ulong rows,
                                __global const long *row_ptr,
				const uint k,
				__global float *H,
				__global float *subMatrix,
				const float lambda,
				__global const unsigned *col_idx,
				__global const unsigned *colMajored_sparse_idx,
                                __global const float *val,
				__global float *subVector)
{
	int block_x = get_group_id(0);
	int block_y = get_group_id(1);
	int block_z = get_group_id(2);
	int blocksize_z = get_num_groups(2);
	int thread_x = get_local_id(0);
	int thread_y = get_local_id(1);
	int dim_x = get_local_size(0);
	int dim_y = get_local_size(1);
	int tid = thread_y * dim_y + thread_x;

	int base = block_z * k * k;
        int baseV = block_z * k;
	if(block_x > block_y){ return; }
	for(int Rw = block_z; Rw < rows; Rw += blocksize_z)
	{
		int omegasize = row_ptr[Rw+1] - row_ptr[Rw];
		if(omegasize > 0)
		{
			float SUM[ (mini/th_blk) * (mini/th_blk) ] = { 0 };
			float subvector[ mini/th_blk ] = { 0  };
			int it = (omegasize - 1) / block_nnz + 1;
			__local float sH[ block_nnz * mini ];
			__local float b[block_nnz];
	
			if(block_x == block_y)
			{
				for(int p = it; p > 1; p--)
				{
					// Load H => sH
					for(int K = thread_y; K < block_nnz ; K += dim_y)
					{
						unsigned offset = col_idx[row_ptr[Rw] + K + (it-p) * block_nnz] * k;
						for(int I = thread_x; I < mini; I += dim_x)
						{
							sH[I + K * mini] = H[offset + block_x * mini + I];
						}
					}
					
					// Calculation 
					for (unsigned S = 0; S < block_nnz; S++)
				    	{
						for(int a = 0; a < mini/th_blk; a++)
						{
							for(int c = 0; c < mini/th_blk; c++)
							{
								SUM[a*th_blk+c] += sH[ S * mini + a * th_blk + thread_x ] * sH[ S * mini + c * th_blk + thread_y ];
							}
						}
					}
				}
				for(int K = thread_y; K < omegasize - (it - 1) * block_nnz ; K += dim_y)
				{
					int offset = col_idx[row_ptr[Rw] + K + (it-1) * block_nnz] * k;
					for(int I = thread_x; I < mini; I += dim_x)
					{
						sH[I + K * mini] = H[offset + block_x * mini + I];
					}
				}

				for (unsigned S = 0; S < omegasize - (it - 1) * block_nnz ; S++)
				{
					for(int a = 0; a < mini/th_blk; a++)
					{
						for(int c = 0; c < mini/th_blk; c++)
						{
							SUM[a * th_blk +c] += sH[ S * mini + a * th_blk + thread_x ] * sH[ S * mini + c * th_blk + thread_y ];
						}
					}	
				}

				// Save results to global memory
				for(int a = 0; a < mini/th_blk; a++)
				{
					for(int c = 0; c < mini/th_blk; c++)
					{
						if( a * th_blk + thread_x == c * th_blk + thread_y)
						{
							SUM[a * th_blk + c] += lambda;
						}
						subMatrix[base + (block_y * mini + a * th_blk + thread_y) * k + block_x * mini + c * th_blk + thread_x] = SUM[a * th_blk + c];
					}
				}
			}
			else
			{
				__local float sHT[ block_nnz * mini ];
				for(int p = it; p > 1; p--)
				{
					// Load H => sH,sHT
					for(int K = thread_y; K < block_nnz; K += dim_y)
					{
						int offset = col_idx[row_ptr[Rw] + K + (it-p) * block_nnz] * k;
						for(int I = thread_x; I < mini; I += dim_x)
						{
							sH[I + K * mini] = H[offset + block_x * mini + I];
						}
						for(int I = thread_x; I < mini; I += dim_x)
						{
							sHT[I + K * mini] = H[offset + block_y * mini + I];
						}
					}
			
					// Calculation 
					for (unsigned S = 0; S < block_nnz; S++)
				    	{
						for(int a = 0; a < mini/th_blk; a++)
						{
							for(int c = 0; c < mini/th_blk; c++)
							{
								SUM[a * th_blk + c] += sH[ S * mini + a * th_blk + thread_x ] * sHT[ S * mini + c * th_blk + thread_y ];
							}
						}
					}		
				}
				for(int K = thread_y; K < omegasize - (it - 1) * block_nnz ; K += dim_y)
				{
					int offset = col_idx[row_ptr[Rw] + K + (it-1) * block_nnz] * k;
					for(int I = thread_x; I < mini; I += dim_x)
					{
						sH[I + K * mini] = H[offset + block_x * mini + I];
					}
					for(int I = thread_x; I < mini; I += dim_x)
					{
						sHT[I + K * mini] = H[offset + block_y * mini + I];
					}
				}

				for (unsigned S = 0; S < omegasize - (it - 1) * block_nnz ; S++)
				{
					for(int a = 0; a < mini/th_blk; a++)
					{
						for(int c = 0; c < mini/th_blk; c++)
						{
							SUM[a * th_blk + c] += sH[ S * mini + a * th_blk + thread_x ] * sHT[ S * mini + c * th_blk + thread_y ];
						}
					}
				}
			
				// Save results to global memory
				for(int a = 0; a < mini/th_blk; a++)
				{
					for(int c = 0; c < mini/th_blk; c++)
					{
						subMatrix[base + (block_y * mini + a * th_blk + thread_y) * k + block_x * mini + c * th_blk + thread_x] = SUM[a * th_blk + c];
					}
				}
			}
		}
	}
}

void batchsolve(int i, unsigned omegasize, int j, __global float *H, __global float *val, __global float *result,__global unsigned *colMajored_sparse_idx,__global long *row_ptr,__global unsigned *col_idx, int baseV)
{
    	int ss = get_local_id(0);
	int gg = get_local_size(0);
	__local float a[block_nnz * N];
	__local float b[block_nnz];
	float subvector0 = 0;
	long nn = omegasize/block_nnz;
	if(nn>0)
    	{
		for(unsigned nm=0;nm<nn;nm++)
		{
		    	for (unsigned idx = row_ptr[i]+nm*block_nnz+ss; idx < (nm+1)*block_nnz+row_ptr[i]; idx+=gg)
		    	{
				unsigned idx2 = colMajored_sparse_idx[idx];
				b[idx-(nm*block_nnz)-row_ptr[i]] = val[idx2];
				for(int ii=0;ii<j;ii++)
				{
				    	a[(idx-(nm*block_nnz)-row_ptr[i])*j+ii]=H[(col_idx[idx] * j) + ii];
				}
		    	}
		    	for(int gh=0;gh<block_nnz;gh++)
		    	{
				if(ss<j)
				{
				    subvector0 += b[gh]*a[gh*j+ss];
				}
		    	}
		}
		for (unsigned idx = row_ptr[i]+nn*block_nnz+ss; idx < row_ptr[i+1]; idx+=gg)
		{
		    	unsigned idx2 = colMajored_sparse_idx[idx];
		    	b[idx-(nn*block_nnz)-row_ptr[i]] = val[idx2];
		    	for(int ii=0;ii<j;ii++)
		    	{
		        	a[(idx-(nn*block_nnz)-row_ptr[i])*j+ii]=H[(col_idx[idx] * j) + ii];
		    	}
		}
		for(unsigned gh=0;gh<row_ptr[i+1]-row_ptr[i]-nn*block_nnz;gh++)
		{
		    	if(ss<j)
		    	{
		        	subvector0 += b[gh]*a[gh*j+ss];
		    	}
		}
    	}
	else
    	{
		for (unsigned idx = row_ptr[i]+ss; idx < row_ptr[i+1]; idx+=gg)
		{
		    	unsigned idx2 = colMajored_sparse_idx[idx];
		    	b[idx-row_ptr[i]] = val[idx2];
		    	for(int ii=0;ii<j;ii++)
		    	{
		        	a[(idx-row_ptr[i])*j+ii]=H[(col_idx[idx] * j) + ii];
		    	}
		}
		for(unsigned gh=0;gh<omegasize;gh++)
		{
		    	if(ss<j)
		    	{
		        	subvector0 += b[gh]*a[gh*j+ss];
		    	}
		}
    	}
    	if(ss < j)
    	{
        	result[baseV+ss]=subvector0;
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
                                   __global float *subMatrix,
				   __global float *subMatrix_f)
{
	int s = get_local_id(0);
   	int g = get_local_size(0);
	int group_id = get_group_id(0);
   	int num_group = get_num_groups(0);
	for (int Rw = group_id; Rw < rows; Rw += num_group)
   	{
   		int baseV = Rw * k;
		__global float *Wr = &W[Rw * k];
		unsigned omegaSize = row_ptr[Rw + 1] - row_ptr[Rw];
		if (omegaSize>0)
		{
			batchsolve(Rw, omegaSize, k, H, val, subVector, colMajored_sparse_idx, row_ptr, col_idx, baseV);
		    	barrier(CLK_GLOBAL_MEM_FENCE);	
		    	for (unsigned c = s; c < k; c += g)
		    	{
				Wr[c] = subVector[Rw * k + c];
		    	}
		}
		else
		{
			for (unsigned c = s; c < k; c += g)
			{
				Wr[c] = 0.0f;
			}
		}
	}
}
