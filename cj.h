#include <CL/cl.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>
#include "util.h"
using namespace std;

class parameter 
{
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
		parameter() 
		{
			k = 90;
			maxiter = 5;
			lambda = 0.1f; 
			do_predict = 0;
			verbose = 0;
			do_nmf = 0;
			nBlocks = 16384;
			nThreadsPerBlock = 15;
		}
};


/** convert the kernel file into a string */
int convertToString(const char *filename,string& s);

/**Getting platforms and choose an available one.*/
int getPlatform(cl_platform_id &platform);

/**Step 2:Query the platform and choose the first GPU device if has one.*/
cl_device_id *getCl_device_id(cl_platform_id &platform);

