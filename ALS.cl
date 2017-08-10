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
	int gid = get_group_id(0);
	p = &(tp[gid*n]);
	choldc1(n, A, p);
	for (i = 0; i < n; ++i) {
		A[base + i * n + i] = 1 / p[i];
		for (j = i + 1; j < n; ++j) {
			sum = 0;
			for (k = i; k < j; ++k){
				sum -= A[base + j * n + k] * A[base + k * n + i];
			}
			A[base + j * n + i] = sum / p[j];
		}
	}
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

void Mt_byM_multiply_k(int i, int j,  __global float *H,__global float *Result, const long ptr,__global const unsigned *idx){
	int base = get_group_id(0)*j*j;
	int ss = get_local_id(0);
	int gg = get_local_size(0);
	//__local float SUM[100];
    float SUM0=0,SUM1=0,SUM2=0,SUM3=0,SUM4=0,SUM5=0,SUM6=0,SUM7=0,SUM8=0,SUM9=0,
    SUM11=0,SUM12=0,SUM13=0,SUM14=0,SUM15=0,SUM16=0,SUM17=0,SUM18=0,SUM19=0,
    SUM22=0,SUM23=0,SUM24=0,SUM25=0,SUM26=0,SUM27=0,SUM28=0,SUM29=0,
    SUM33=0,SUM34=0,SUM35=0,SUM36=0,SUM37=0,SUM38=0,SUM39=0,
    SUM44=0,SUM45=0,SUM46=0,SUM47=0,SUM48=0,SUM49=0,
    SUM55=0,SUM56=0,SUM57=0,SUM58=0,SUM59=0,
    SUM66=0,SUM67=0,SUM68=0,SUM69=0,
    SUM77=0,SUM78=0,SUM79=0,
    SUM88=0,SUM89=0,
    SUM99=0;
    __local float a[300];
	__local int offset[30];
	int f=300/j;
	int nh=(i/f)+1;
    int p=nh;
    if(i>f)
    {
        for(p;p>1;p--)
        {
            for (int K = ss; K < f; K+=gg)
            {
                offset[K] = idx[ ptr + K + (nh-p)*f ] * j;
                for(unsigned I=0;I<j;++I)
                {
                    a[K*j+I] = H[offset[K]+I];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            for (unsigned S = 0; S < f; S++)
            {
                SUM0 += a[S*j]*a[S*j];
                SUM1 += a[S*j]*a[S*j+1];
                SUM2 += a[S*j]*a[S*j+2];
                SUM3 += a[S*j]*a[S*j+3];
                SUM4 += a[S*j]*a[S*j+4];
                SUM5 += a[S*j]*a[S*j+5];
                SUM6 += a[S*j]*a[S*j+6];
                SUM7 += a[S*j]*a[S*j+7];
                SUM8 += a[S*j]*a[S*j+8];
                SUM9 += a[S*j]*a[S*j+9];

                SUM11 += a[S*j+1]*a[S*j+1];
                SUM12 += a[S*j+1]*a[S*j+2];
                SUM13 += a[S*j+1]*a[S*j+3];
                SUM14 += a[S*j+1]*a[S*j+4];
                SUM15 += a[S*j+1]*a[S*j+5];
                SUM16 += a[S*j+1]*a[S*j+6];
                SUM17 += a[S*j+1]*a[S*j+7];
                SUM18 += a[S*j+1]*a[S*j+8];
                SUM19 += a[S*j+1]*a[S*j+9];

                SUM22 += a[S*j+2]*a[S*j+2];
                SUM23 += a[S*j+2]*a[S*j+3];
                SUM24 += a[S*j+2]*a[S*j+4];
                SUM25 += a[S*j+2]*a[S*j+5];
                SUM26 += a[S*j+2]*a[S*j+6];
                SUM27 += a[S*j+2]*a[S*j+7];
                SUM28 += a[S*j+2]*a[S*j+8];
                SUM29 += a[S*j+2]*a[S*j+9];

                SUM33 += a[S*j+3]*a[S*j+3];
                SUM34 += a[S*j+3]*a[S*j+4];
                SUM35 += a[S*j+3]*a[S*j+5];
                SUM36 += a[S*j+3]*a[S*j+6];
                SUM37 += a[S*j+3]*a[S*j+7];
                SUM38 += a[S*j+3]*a[S*j+8];
                SUM39 += a[S*j+3]*a[S*j+9];

                SUM44 += a[S*j+4]*a[S*j+4];
                SUM45 += a[S*j+4]*a[S*j+5];
                SUM46 += a[S*j+4]*a[S*j+6];
                SUM47 += a[S*j+4]*a[S*j+7];
                SUM48 += a[S*j+4]*a[S*j+8];
                SUM49 += a[S*j+4]*a[S*j+9];

                SUM55 += a[S*j+5]*a[S*j+5];
                SUM56 += a[S*j+5]*a[S*j+6];
                SUM57 += a[S*j+5]*a[S*j+7];
                SUM58 += a[S*j+5]*a[S*j+8];
                SUM59 += a[S*j+5]*a[S*j+9];

                SUM66 += a[S*j+6]*a[S*j+6];
                SUM67 += a[S*j+6]*a[S*j+7];
                SUM68 += a[S*j+6]*a[S*j+8];
                SUM69 += a[S*j+6]*a[S*j+9];

                SUM77 += a[S*j+7]*a[S*j+7];
                SUM78 += a[S*j+7]*a[S*j+8];
                SUM79 += a[S*j+7]*a[S*j+9];

                SUM88 += a[S*j+8]*a[S*j+8];
                SUM89 += a[S*j+8]*a[S*j+9];

                SUM99 += a[S*j+9]*a[S*j+9];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        for(int K = ss;K < i-(nh-1)*f ; K+=gg)
        {
            offset[K] = idx[ptr + K+ (nh-1)*f ] * j;
        	for(unsigned I=0;I<j;++I)
        	{
                a[K*j+I] = H[offset[K]+I];
        	}
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned S = 0; S < i-(nh-1)*f ; S++)
        {
                SUM0 += a[S*j]*a[S*j];
                SUM1 += a[S*j]*a[S*j+1];
                SUM2 += a[S*j]*a[S*j+2];
                SUM3 += a[S*j]*a[S*j+3];
                SUM4 += a[S*j]*a[S*j+4];
                SUM5 += a[S*j]*a[S*j+5];
                SUM6 += a[S*j]*a[S*j+6];
                SUM7 += a[S*j]*a[S*j+7];
                SUM8 += a[S*j]*a[S*j+8];
                SUM9 += a[S*j]*a[S*j+9];

                SUM11 += a[S*j+1]*a[S*j+1];
                SUM12 += a[S*j+1]*a[S*j+2];
                SUM13 += a[S*j+1]*a[S*j+3];
                SUM14 += a[S*j+1]*a[S*j+4];
                SUM15 += a[S*j+1]*a[S*j+5];
                SUM16 += a[S*j+1]*a[S*j+6];
                SUM17 += a[S*j+1]*a[S*j+7];
                SUM18 += a[S*j+1]*a[S*j+8];
                SUM19 += a[S*j+1]*a[S*j+9];

                SUM22 += a[S*j+2]*a[S*j+2];
                SUM23 += a[S*j+2]*a[S*j+3];
                SUM24 += a[S*j+2]*a[S*j+4];
                SUM25 += a[S*j+2]*a[S*j+5];
                SUM26 += a[S*j+2]*a[S*j+6];
                SUM27 += a[S*j+2]*a[S*j+7];
                SUM28 += a[S*j+2]*a[S*j+8];
                SUM29 += a[S*j+2]*a[S*j+9];

                SUM33 += a[S*j+3]*a[S*j+3];
                SUM34 += a[S*j+3]*a[S*j+4];
                SUM35 += a[S*j+3]*a[S*j+5];
                SUM36 += a[S*j+3]*a[S*j+6];
                SUM37 += a[S*j+3]*a[S*j+7];
                SUM38 += a[S*j+3]*a[S*j+8];
                SUM39 += a[S*j+3]*a[S*j+9];

                SUM44 += a[S*j+4]*a[S*j+4];
                SUM45 += a[S*j+4]*a[S*j+5];
                SUM46 += a[S*j+4]*a[S*j+6];
                SUM47 += a[S*j+4]*a[S*j+7];
                SUM48 += a[S*j+4]*a[S*j+8];
                SUM49 += a[S*j+4]*a[S*j+9];

                SUM55 += a[S*j+5]*a[S*j+5];
                SUM56 += a[S*j+5]*a[S*j+6];
                SUM57 += a[S*j+5]*a[S*j+7];
                SUM58 += a[S*j+5]*a[S*j+8];
                SUM59 += a[S*j+5]*a[S*j+9];

                SUM66 += a[S*j+6]*a[S*j+6];
                SUM67 += a[S*j+6]*a[S*j+7];
                SUM68 += a[S*j+6]*a[S*j+8];
                SUM69 += a[S*j+6]*a[S*j+9];

                SUM77 += a[S*j+7]*a[S*j+7];
                SUM78 += a[S*j+7]*a[S*j+8];
                SUM79 += a[S*j+7]*a[S*j+9];

                SUM88 += a[S*j+8]*a[S*j+8];
                SUM89 += a[S*j+8]*a[S*j+9];

                SUM99 += a[S*j+9]*a[S*j+9];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        Result[base+0]=SUM0;
        Result[base+1]=SUM1;
        Result[base+2]=SUM2;
        Result[base+3]=SUM3;
        Result[base+4]=SUM4;
        Result[base+5]=SUM5;
        Result[base+6]=SUM6;
        Result[base+7]=SUM7;
        Result[base+8]=SUM8;
        Result[base+9]=SUM9;
        Result[base+10]=Result[base+1];
        Result[base+11]=SUM11;
        Result[base+12]=SUM12;
        Result[base+13]=SUM13;
        Result[base+14]=SUM14;
        Result[base+15]=SUM15;
        Result[base+16]=SUM16;
        Result[base+17]=SUM17;
        Result[base+18]=SUM18;
        Result[base+19]=SUM19;
        Result[base+20]=Result[base+2];
        Result[base+21]=Result[base+12];
        Result[base+22]=SUM22;
        Result[base+23]=SUM23;
        Result[base+24]=SUM24;
        Result[base+25]=SUM25;
        Result[base+26]=SUM26;
        Result[base+27]=SUM27;
        Result[base+28]=SUM28;
        Result[base+29]=SUM29;
        Result[base+30]=Result[base+3];
        Result[base+31]=Result[base+13];
        Result[base+32]=Result[base+23];
        Result[base+33]=SUM33;
        Result[base+34]=SUM34;
        Result[base+35]=SUM35;
        Result[base+36]=SUM36;
        Result[base+37]=SUM37;
        Result[base+38]=SUM38;
        Result[base+39]=SUM39;
        Result[base+40]=Result[base+4];
        Result[base+41]=Result[base+14];
        Result[base+42]=Result[base+24];
        Result[base+43]=Result[base+34];
        Result[base+44]=SUM44;
        Result[base+45]=SUM45;
        Result[base+46]=SUM46;
        Result[base+47]=SUM47;
        Result[base+48]=SUM48;
        Result[base+49]=SUM49;
        Result[base+50]=Result[base+5];
        Result[base+51]=Result[base+15];
        Result[base+52]=Result[base+25];
        Result[base+53]=Result[base+35];
        Result[base+54]=Result[base+45];
        Result[base+55]=SUM55;
        Result[base+56]=SUM56;
        Result[base+57]=SUM57;
        Result[base+58]=SUM58;
        Result[base+59]=SUM59;
        Result[base+60]=Result[base+6];
        Result[base+61]=Result[base+16];
        Result[base+62]=Result[base+26];
        Result[base+63]=Result[base+36];
        Result[base+64]=Result[base+46];
        Result[base+65]=Result[base+56];
        Result[base+66]=SUM66;
        Result[base+67]=SUM67;
        Result[base+68]=SUM68;
        Result[base+69]=SUM69;
        Result[base+70]=Result[base+7];
        Result[base+71]=Result[base+17];
        Result[base+72]=Result[base+27];
        Result[base+73]=Result[base+37];
        Result[base+74]=Result[base+47];
        Result[base+75]=Result[base+57];
        Result[base+76]=Result[base+67];
        Result[base+77]=SUM77;
        Result[base+78]=SUM78;
        Result[base+79]=SUM79;
        Result[base+80]=Result[base+8];
        Result[base+81]=Result[base+18];
        Result[base+82]=Result[base+28];
        Result[base+83]=Result[base+38];
        Result[base+84]=Result[base+48];
        Result[base+85]=Result[base+58];
        Result[base+86]=Result[base+68];
        Result[base+87]=Result[base+78];
        Result[base+88]=SUM88;
        Result[base+89]=SUM89;
        Result[base+90]=Result[base+9];
        Result[base+91]=Result[base+19];
        Result[base+92]=Result[base+29];
        Result[base+93]=Result[base+39];
        Result[base+94]=Result[base+49];
        Result[base+95]=Result[base+59];
        Result[base+96]=Result[base+69];
        Result[base+97]=Result[base+79];
        Result[base+98]=Result[base+89];
        Result[base+99]=SUM99;
    }
    else
    {
        for (int K = ss; K < i; K+=gg)
    	{
            offset[K] = idx[ptr + K] * j;
        	for(unsigned I=0;I<j;++I)
        	{
                a[K*j+I] = H[offset[K]+I];
        	}
    	}
    	barrier(CLK_LOCAL_MEM_FENCE);
        for (unsigned S = 0; S < i; S++)
        {
                SUM0 += a[S*j]*a[S*j];
                SUM1 += a[S*j]*a[S*j+1];
                SUM2 += a[S*j]*a[S*j+2];
                SUM3 += a[S*j]*a[S*j+3];
                SUM4 += a[S*j]*a[S*j+4];
                SUM5 += a[S*j]*a[S*j+5];
                SUM6 += a[S*j]*a[S*j+6];
                SUM7 += a[S*j]*a[S*j+7];
                SUM8 += a[S*j]*a[S*j+8];
                SUM9 += a[S*j]*a[S*j+9];

                SUM11 += a[S*j+1]*a[S*j+1];
                SUM12 += a[S*j+1]*a[S*j+2];
                SUM13 += a[S*j+1]*a[S*j+3];
                SUM14 += a[S*j+1]*a[S*j+4];
                SUM15 += a[S*j+1]*a[S*j+5];
                SUM16 += a[S*j+1]*a[S*j+6];
                SUM17 += a[S*j+1]*a[S*j+7];
                SUM18 += a[S*j+1]*a[S*j+8];
                SUM19 += a[S*j+1]*a[S*j+9];

                SUM22 += a[S*j+2]*a[S*j+2];
                SUM23 += a[S*j+2]*a[S*j+3];
                SUM24 += a[S*j+2]*a[S*j+4];
                SUM25 += a[S*j+2]*a[S*j+5];
                SUM26 += a[S*j+2]*a[S*j+6];
                SUM27 += a[S*j+2]*a[S*j+7];
                SUM28 += a[S*j+2]*a[S*j+8];
                SUM29 += a[S*j+2]*a[S*j+9];

                SUM33 += a[S*j+3]*a[S*j+3];
                SUM34 += a[S*j+3]*a[S*j+4];
                SUM35 += a[S*j+3]*a[S*j+5];
                SUM36 += a[S*j+3]*a[S*j+6];
                SUM37 += a[S*j+3]*a[S*j+7];
                SUM38 += a[S*j+3]*a[S*j+8];
                SUM39 += a[S*j+3]*a[S*j+9];

                SUM44 += a[S*j+4]*a[S*j+4];
                SUM45 += a[S*j+4]*a[S*j+5];
                SUM46 += a[S*j+4]*a[S*j+6];
                SUM47 += a[S*j+4]*a[S*j+7];
                SUM48 += a[S*j+4]*a[S*j+8];
                SUM49 += a[S*j+4]*a[S*j+9];

                SUM55 += a[S*j+5]*a[S*j+5];
                SUM56 += a[S*j+5]*a[S*j+6];
                SUM57 += a[S*j+5]*a[S*j+7];
                SUM58 += a[S*j+5]*a[S*j+8];
                SUM59 += a[S*j+5]*a[S*j+9];

                SUM66 += a[S*j+6]*a[S*j+6];
                SUM67 += a[S*j+6]*a[S*j+7];
                SUM68 += a[S*j+6]*a[S*j+8];
                SUM69 += a[S*j+6]*a[S*j+9];

                SUM77 += a[S*j+7]*a[S*j+7];
                SUM78 += a[S*j+7]*a[S*j+8];
                SUM79 += a[S*j+7]*a[S*j+9];

                SUM88 += a[S*j+8]*a[S*j+8];
                SUM89 += a[S*j+8]*a[S*j+9];

                SUM99 += a[S*j+9]*a[S*j+9];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        Result[base+0]=SUM0;
        Result[base+1]=SUM1;
        Result[base+2]=SUM2;
        Result[base+3]=SUM3;
        Result[base+4]=SUM4;
        Result[base+5]=SUM5;
        Result[base+6]=SUM6;
        Result[base+7]=SUM7;
        Result[base+8]=SUM8;
        Result[base+9]=SUM9;
        Result[base+10]=Result[base+1];
        Result[base+11]=SUM11;
        Result[base+12]=SUM12;
        Result[base+13]=SUM13;
        Result[base+14]=SUM14;
        Result[base+15]=SUM15;
        Result[base+16]=SUM16;
        Result[base+17]=SUM17;
        Result[base+18]=SUM18;
        Result[base+19]=SUM19;
        Result[base+20]=Result[base+2];
        Result[base+21]=Result[base+12];
        Result[base+22]=SUM22;
        Result[base+23]=SUM23;
        Result[base+24]=SUM24;
        Result[base+25]=SUM25;
        Result[base+26]=SUM26;
        Result[base+27]=SUM27;
        Result[base+28]=SUM28;
        Result[base+29]=SUM29;
        Result[base+30]=Result[base+3];
        Result[base+31]=Result[base+13];
        Result[base+32]=Result[base+23];
        Result[base+33]=SUM33;
        Result[base+34]=SUM34;
        Result[base+35]=SUM35;
        Result[base+36]=SUM36;
        Result[base+37]=SUM37;
        Result[base+38]=SUM38;
        Result[base+39]=SUM39;
        Result[base+40]=Result[base+4];
        Result[base+41]=Result[base+14];
        Result[base+42]=Result[base+24];
        Result[base+43]=Result[base+34];
        Result[base+44]=SUM44;
        Result[base+45]=SUM45;
        Result[base+46]=SUM46;
        Result[base+47]=SUM47;
        Result[base+48]=SUM48;
        Result[base+49]=SUM49;
        Result[base+50]=Result[base+5];
        Result[base+51]=Result[base+15];
        Result[base+52]=Result[base+25];
        Result[base+53]=Result[base+35];
        Result[base+54]=Result[base+45];
        Result[base+55]=SUM55;
        Result[base+56]=SUM56;
        Result[base+57]=SUM57;
        Result[base+58]=SUM58;
        Result[base+59]=SUM59;
        Result[base+60]=Result[base+6];
        Result[base+61]=Result[base+16];
        Result[base+62]=Result[base+26];
        Result[base+63]=Result[base+36];
        Result[base+64]=Result[base+46];
        Result[base+65]=Result[base+56];
        Result[base+66]=SUM66;
        Result[base+67]=SUM67;
        Result[base+68]=SUM68;
        Result[base+69]=SUM69;
        Result[base+70]=Result[base+7];
        Result[base+71]=Result[base+17];
        Result[base+72]=Result[base+27];
        Result[base+73]=Result[base+37];
        Result[base+74]=Result[base+47];
        Result[base+75]=Result[base+57];
        Result[base+76]=Result[base+67];
        Result[base+77]=SUM77;
        Result[base+78]=SUM78;
        Result[base+79]=SUM79;
        Result[base+80]=Result[base+8];
        Result[base+81]=Result[base+18];
        Result[base+82]=Result[base+28];
        Result[base+83]=Result[base+38];
        Result[base+84]=Result[base+48];
        Result[base+85]=Result[base+58];
        Result[base+86]=Result[base+68];
        Result[base+87]=Result[base+78];
        Result[base+88]=SUM88;
        Result[base+89]=SUM89;
        Result[base+90]=Result[base+9];
        Result[base+91]=Result[base+19];
        Result[base+92]=Result[base+29];
        Result[base+93]=Result[base+39];
        Result[base+94]=Result[base+49];
        Result[base+95]=Result[base+59];
        Result[base+96]=Result[base+69];
        Result[base+97]=Result[base+79];
        Result[base+98]=Result[base+89];
        Result[base+99]=SUM99;
    }
}

void batchsolve(int i, int j, __global float *H, __global float *val, __global float *result,__global unsigned *colMajored_sparse_idx,__global long *row_ptr,__global unsigned *col_idx){
    int basev = get_group_id(0) * j;
    int ss = get_local_id(0);
	int gg = get_local_size(0);
	__local float a[300];
	__local float b[30];
	float subvector0 = 0, subvector1 = 0, subvector2 = 0,subvector3 = 0,subvector4 = 0, subvector5 = 0,subvector6 = 0,subvector7 = 0,subvector8 = 0,subvector9 = 0;
	unsigned n = row_ptr[i+1]-row_ptr[i];
	long nn = n/30;
	if(nn>0)
    {
        for(unsigned nm=0;nm<nn;nm++)
        {
            for (unsigned idx = row_ptr[i]+nm*30+ss; idx < (nm+1)*30+row_ptr[i]; idx+=gg)
            {
                unsigned idx2 = colMajored_sparse_idx[idx];
                b[idx-(nm*30)-row_ptr[i]] = val[idx2];
                for(int ii=0;ii<j;ii++)
                {
                    a[(idx-(nm*30)-row_ptr[i])*j+ii]=H[(col_idx[idx] * j) + ii];
                }
            }
            for(int gh=0;gh<30;gh++)
            {
                subvector0 += b[gh]*a[gh*j];
                subvector1 += b[gh]*a[gh*j+1];
                subvector2 += b[gh]*a[gh*j+2];
                subvector3 += b[gh]*a[gh*j+3];
                subvector4 += b[gh]*a[gh*j+4];
                subvector5 += b[gh]*a[gh*j+5];
                subvector6 += b[gh]*a[gh*j+6];
                subvector7 += b[gh]*a[gh*j+7];
                subvector8 += b[gh]*a[gh*j+8];
                subvector9 += b[gh]*a[gh*j+9];
            }
        }
        for (unsigned idx = row_ptr[i]+nn*30+ss; idx < row_ptr[i+1]; idx+=gg)
        {
            unsigned idx2 = colMajored_sparse_idx[idx];
            b[idx-(nn*30)-row_ptr[i]] = val[idx2];
            for(int ii=0;ii<j;ii++)
            {
                a[(idx-(nn*30)-row_ptr[i])*j+ii]=H[(col_idx[idx] * j) + ii];
            }
        }
        for(unsigned gh=0;gh<row_ptr[i+1]-row_ptr[i]-nn*30;gh++)
        {
            subvector0 += b[gh]*a[gh*j];
            subvector1 += b[gh]*a[gh*j+1];
            subvector2 += b[gh]*a[gh*j+2];
            subvector3 += b[gh]*a[gh*j+3];
            subvector4 += b[gh]*a[gh*j+4];
            subvector5 += b[gh]*a[gh*j+5];
            subvector6 += b[gh]*a[gh*j+6];
            subvector7 += b[gh]*a[gh*j+7];
            subvector8 += b[gh]*a[gh*j+8];
            subvector9 += b[gh]*a[gh*j+9];
        }
    }
	else
    {
	//printf("else enter.\n");
        for (unsigned idx = row_ptr[i]+ss; idx < row_ptr[i+1]; idx+=gg)
        {
            unsigned idx2 = colMajored_sparse_idx[idx];
            b[idx-row_ptr[i]] = val[idx2];
            for(int ii=0;ii<j;ii++)
            {
                a[(idx-row_ptr[i])*j+ii]=H[(col_idx[idx] * j) + ii];
            }
        }
        for(unsigned gh=0;gh<n;gh++)
        {
            subvector0 += b[gh]*a[gh*j];
            subvector1 += b[gh]*a[gh*j+1];
            subvector2 += b[gh]*a[gh*j+2];
            subvector3 += b[gh]*a[gh*j+3];
            subvector4 += b[gh]*a[gh*j+4];
            subvector5 += b[gh]*a[gh*j+5];
            subvector6 += b[gh]*a[gh*j+6];
            subvector7 += b[gh]*a[gh*j+7];
            subvector8 += b[gh]*a[gh*j+8];
            subvector9 += b[gh]*a[gh*j+9];
        }
    }
    result[basev+0]=subvector0;
    result[basev+1]=subvector1;
    result[basev+2]=subvector2;
    result[basev+3]=subvector3;
    result[basev+4]=subvector4;
    result[basev+5]=subvector5;
    result[basev+6]=subvector6;
    result[basev+7]=subvector7;
    result[basev+8]=subvector8;
    result[basev+9]=subvector9;
}

void batchsolve1(int i, int j, __global float *W, __global float *val, __global float *result,__global long *col_ptr,__global unsigned *row_idx){
    int basev = get_group_id(0) * j;
    int ss = get_local_id(0);
	int gg = get_local_size(0);
	__local float a[300];
	__local float b[30];
	float subvector0 = 0, subvector1 = 0, subvector2 = 0,subvector3 = 0,subvector4 = 0, subvector5 = 0,subvector6 = 0,subvector7 = 0,subvector8 = 0,subvector9 = 0;
	unsigned n = col_ptr[i+1]-col_ptr[i];
	long nn = n/30;
	if(nn>0)
    {
        for(unsigned nm=0;nm<nn;nm++)
        {
            for (unsigned idx = col_ptr[i]+nm*30+ss; idx < (nm+1)*30+col_ptr[i]; idx+=gg)
            {
                b[idx-(nm*30)-col_ptr[i]] = val[idx];
                for(int ii=0;ii<j;ii++)
                {
                    a[(idx-(nm*30)-col_ptr[i])*j+ii]=W[(row_idx[idx] * j) + ii];
                }
            }
            for(int gh=0;gh<30;gh++)
            {
                subvector0 += b[gh]*a[gh*j];
                subvector1 += b[gh]*a[gh*j+1];
                subvector2 += b[gh]*a[gh*j+2];
                subvector3 += b[gh]*a[gh*j+3];
                subvector4 += b[gh]*a[gh*j+4];
                subvector5 += b[gh]*a[gh*j+5];
                subvector6 += b[gh]*a[gh*j+6];
                subvector7 += b[gh]*a[gh*j+7];
                subvector8 += b[gh]*a[gh*j+8];
                subvector9 += b[gh]*a[gh*j+9];
            }
        }
        for (unsigned idx = col_ptr[i]+nn*30+ss; idx < col_ptr[i+1]; idx+=gg)
        {
            b[idx-(nn*30)-col_ptr[i]] = val[idx];
            for(int ii=0;ii<j;ii++)
            {
                a[(idx-(nn*30)-col_ptr[i])*j+ii]=W[(row_idx[idx] * j) + ii];
            }
        }
        for(unsigned gh=0;gh<col_ptr[i+1]-col_ptr[i]-nn*30;gh++)
        {
            subvector0 += b[gh]*a[gh*j];
            subvector1 += b[gh]*a[gh*j+1];
            subvector2 += b[gh]*a[gh*j+2];
            subvector3 += b[gh]*a[gh*j+3];
            subvector4 += b[gh]*a[gh*j+4];
            subvector5 += b[gh]*a[gh*j+5];
            subvector6 += b[gh]*a[gh*j+6];
            subvector7 += b[gh]*a[gh*j+7];
            subvector8 += b[gh]*a[gh*j+8];
            subvector9 += b[gh]*a[gh*j+9];
        }
    }
	else
    {
        for (unsigned idx = col_ptr[i]+ss; idx < col_ptr[i+1]; idx+=gg)
        {
            b[idx-col_ptr[i]] = val[idx];
            for(int ii=0;ii<j;ii++)
            {
                a[(idx-col_ptr[i])*j+ii]=W[(row_idx[idx] * j) + ii];
            }
        }
        for(unsigned gh=0;gh<n;gh++)
        {
            subvector0 += b[gh]*a[gh*j];
            subvector1 += b[gh]*a[gh*j+1];
            subvector2 += b[gh]*a[gh*j+2];
            subvector3 += b[gh]*a[gh*j+3];
            subvector4 += b[gh]*a[gh*j+4];
            subvector5 += b[gh]*a[gh*j+5];
            subvector6 += b[gh]*a[gh*j+6];
            subvector7 += b[gh]*a[gh*j+7];
            subvector8 += b[gh]*a[gh*j+8];
            subvector9 += b[gh]*a[gh*j+9];
        }
    }
    result[basev+0]=subvector0;
    result[basev+1]=subvector1;
    result[basev+2]=subvector2;
    result[basev+3]=subvector3;
    result[basev+4]=subvector4;
    result[basev+5]=subvector5;
    result[basev+6]=subvector6;
    result[basev+7]=subvector7;
    result[basev+8]=subvector8;
    result[basev+9]=subvector9;
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
            barrier(CLK_LOCAL_MEM_FENCE);

            for (unsigned c = s; c < k; c+=g){
                subMatrix[base + c * k + c] += lambda;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            if(s==0)
            {
	           inverseMatrix_CholeskyMethod(k, subMatrix, p);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
		    for (unsigned c = s; c < k; c+=g){
                for(unsigned aa=0;aa<k;aa++){
                    subMatrix_f[c*k+aa]=subMatrix[base + c * k + aa];
                }
            }
/*
            for (unsigned c = s; c < k; c+=g){
                subVector[baseV + c] = 0;
                for (unsigned idx = row_ptr[Rw]; idx < row_ptr[Rw + 1]; ++idx){
                    unsigned idx2 = colMajored_sparse_idx[idx];
                    subVector[baseV + c] += val[idx2] * H[(col_idx[idx] * k) + c];
                }
            }
	*/	batchsolve(Rw,k,H,val,subVector,colMajored_sparse_idx,row_ptr,col_idx);
            barrier(CLK_LOCAL_MEM_FENCE);
            for (unsigned c = s; c < k; c+=g){
                 Wr[c]=0.0f;
	             for(unsigned subVid=0;subVid<k;++subVid){
                    Wr[c] +=subVector[baseV+subVid]*subMatrix[base + c * k+subVid];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
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
            /*
            for (unsigned c = s; c < k; c+=g){
                subVector[baseV + c] = 0;
				for (unsigned idx = col_ptr[Rh]; idx < col_ptr[Rh + 1]; ++idx){
                    subVector[baseV + c] += val[idx] * W[(row_idx[idx] * k) + c];
				}
			}
			*/
			batchsolve1(Rh,k,W,val,subVector,col_ptr,row_idx);
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

