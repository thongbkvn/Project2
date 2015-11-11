#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

#define FILE_NAME "data.txt"
#define MAX_THREAD 4
#define N 2 //So nguoi co so thich tuong tu


void onError(int rc, int fc, char* msg)
{
    if (rc == fc)
    {
	fprintf(stderr, "\nError: %s\n", msg);
	exit(1);
    }
}

/* Ham doc du lieu vao mang outArr tu filename 
   -outArr: output con tro toi mang 2 chieu chua du lieu doc duoc
   -row: output so cot ma tran
   -colmn: output so hang ma tran
   -filename: input ten file du lieu dau vao */
void readData(float ***outArr, int *row, int *colmn, char* filename)
{
    FILE *fp = fopen(filename, "r");
    if (fp == NULL)
    {
	perror("fopen()");
	exit(1);
    }

    float **A;
    int m, n;

    fscanf(fp, "%d%d", &m, &n);
    A = calloc(m, sizeof(float*));
    for (int i=0; i<m; i++)
	A[i] = calloc(n, sizeof(float));

    for (int i=0; i<m; i++)
	for (int j=0; j<n; j++)
	    fscanf(fp, "%f", &A[i][j]);
    *outArr = A;
    *row = m; *colmn = n;
    fclose(fp);
}

/*Ham tao du lieu kiem thu*/
void createData(int m, int n, char* filename)
{
    FILE *fp = fopen(filename, "w");
    fprintf(fp, "%d %d\n", m, n);
    for (int i=0; i<m; i++)
    {
//	srand(time(NULL));
	for (int j=0; j<n; j++)
	    fprintf(fp, "%4d", rand() % 6);
	fprintf(fp, "\n");
    }
    fclose(fp);
}

/*Ham hien thi ma tran 2 chieu ra man hinh*/
void readArr(float **A, int m, int n)
{
    for (int i=0; i<m; i++)
    {
	printf("\n");
	for (int j=0; j<n; j++)
	    printf("%8.2f", A[i][j]);
    }
}


/* Ham tinh goc 2 vector
   - A, B vector dau vao
   - n so chieu */
float cosine(float *A, float *B, int n)
{
    float sum = 0;
    float modA = 0, modB = 0;
    
#pragma omp parallel for schedule(static,3) reduction(+:sum,modA,modB)
    for (int i=0; i<n; i++)
    {
	sum += A[i] * B[i];
	modA += A[i] * A[i];
	modB += B[i] * B[i];
    }

    //Truong hop rating cac phim nhu nhau hoac chua xem, coi nhu la trung binh
    //Khong cung so thich, cung khong trai so thich
    if (modA * modB == 0)
	return 0;
    return sum/sqrt(modA*modB);
}





int main(int argc, char** argv)
{
    int m, n; //Hang, cot
    float **A; //Ma tran dau vao
    float **B; //Ma tran chua sim(A[0],A[1])...sim(A[0],A[n-1]), sim(A[1],A[0]), ... sim(A[1],A[n-1]),...
    
    
    readData(&A, &m, &n, FILE_NAME);
    readArr(A, m, n);


    /*--------------------------------------------------*/
    /*                    CHUAN HOA                     */
    /*--------------------------------------------------*/    
#pragma omp parallel num_threads(MAX_THREAD)
    {
	int threadNum = omp_get_thread_num();
	int numThread = omp_get_num_threads();

#pragma omp for
	for (int i=0; i<m; i++)
	{
	    float sum = 0; //Tong cac rating
	    int t = 0;     //So rating
	    
#pragma omp parallel for schedule(static, 3) reduction(+:sum)
	    for (int j=0; j<n; j++)
	    {
		if (A[i][j] != 0)
		{
		    sum += A[i][j];
		    t++;
		}
	    }

	    float rowMean = sum/t;
	    
#pragma omp parallel for schedule(static, 3)
	    for (int j=0; j<n; j++)
	    {
		if (A[i][j] != 0)
		    A[i][j] -= rowMean;
	    }
	}
    }

    readArr(A, m, n);



    /*--------------------------------------------------*/
    /*                    Tinh sim                      */
    /*--------------------------------------------------*/
    B = calloc(m, sizeof(float*));
    for (int i=0; i<m; i++)
	B[i] = calloc(m, sizeof(float*));

    for (int i=0; i<m; i++)
    {
#pragma omp parallel for
	for (int j=i+1; j<m; j++)
	{
	    B[i][j] = cosine(A[i], A[j], n);
	}
    }

    for (int i=0; i<m; i++)
    	for (int j=0; j<=i; j++)
    	{
    	    if (i == j)
    		B[i][j] = 1;
    	    else
    		B[i][j] = B[j][i];
    	}

    readArr(B,m,m);











    



    free(A);
    free(B);
    printf("\n");
    return 0;
}
