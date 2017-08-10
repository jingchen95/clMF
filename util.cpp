#include "stdafx.h"
#include "util.h"

void load(const char* srcdir, smat_t &R, bool ifALS, bool with_weights){
	char filename[1024], buf[1024];
	sprintf(filename,"%s/meta",srcdir);
	FILE *fp = fopen(filename,"r");
	long m, n, nnz;
	fscanf(fp, "%ld %ld", &m, &n);

	fscanf(fp, "%ld %s", &nnz, buf);
	sprintf(filename,"%s/%s", srcdir, buf);
	R.load(m, n, nnz, filename, ifALS, with_weights);
	fclose(fp);
	return ;
}
void initial_col(mat_t &X, long k, long n){
	X = mat_t(k, vec_t(n));
	srand(0L);//srand48(0L);//########################################################################################################################################
	long i,j;
	for( i = 0; i < n; ++i)
		for(j = 0; j < k; ++j)
			X[j][i] = 0.1*(float(rand()) / RAND_MAX)+0.001;//X[j][i] = 0.1*drand48();
}
