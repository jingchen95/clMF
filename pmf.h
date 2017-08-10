#ifndef _PMF_H_
#define _PMF_H_

#include "util.h"

class parameter {
	public:
		int solver_type;
		int k;
		int maxiter, maxinneriter;
		float lambda;
		float rho;
		int lrate_method, num_blocks;
		int do_predict, verbose;
		int do_nmf;  // non-negative matrix factorization
		int nBlocks;
		int nThreadsPerBlock;
		parameter() {
			k =10;
			maxiter = 5;
			lambda = 0.1f; 
			do_predict = 0;
			verbose = 0;
			do_nmf = 0;
			nBlocks = 8192;
			nThreadsPerBlock =32;
		}
};


#endif
