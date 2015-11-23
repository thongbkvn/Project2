#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

#define FILE_NAME_DEFAULT "input.txt"
#define MAX_THREAD 4
#define K 2 //So nguoi co so thich tuong tu
#define MAXP 4 //Xac suat toi thieu so phim da xem cua 1 nguoi, dung trong ham tao du lieu kiem thu
//#define DEBUG


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

//Ham ghi mang A gom row hang, col cot ra filename
void writeData(float **A, int row, int col, char* filename)
{
    FILE *fp = fopen(filename, "w");
    if (fp == NULL)
    {
	perror("fopen()");
	exit(1);
    }

    fprintf(fp, "%8d %8d\n", row, col);
    
    for (int i=0; i<row; i++)
    {
	for (int j=0; j<col; j++)
	    fprintf(fp,"%8.2f", A[i][j]);
	fprintf(fp, "\n");
    }
    
    fclose(fp);
}

    
/*Ham tao du lieu kiem thu
  Tao ma tran m dong, n cot cac so ngau nhien vao filename */
void createData(int m, int n, char* filename)
{
    FILE *fp = fopen(filename, "w");
    fprintf(fp, "%d %d\n", m, n);
    fprintf(fp, "\n");
    srand(time(NULL));
    for (int i=0; i<m; i++)
    {
	int p = rand() % MAXP + 1;
	int k;
	for (int j=0; j<n; j++)
	{
	    k = rand() % p;
	    if (k == 0)
	    {
		fprintf(fp, "%4d", rand() % 6);
	    }
	    else
		fprintf(fp, "%4d", 0);
	}
	fprintf(fp, "\n");
    }
    fclose(fp);
}


/*Ham hien thi ma tran 2 chieu ra man hinh*/
void readArr(float **A, int m, int n)
{
    puts("");
    for (int i=0; i<m; i++)
    {
	printf("\n");
	for (int j=0; j<n; j++)
	    printf("%8.2f", A[i][j]);
    }
    puts("");
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



/*Tim k nguoi da xem item j
  -A, m: input mang A gom m hang
  -y: item thu j
  -len: output chieu dai mang ket qua
  -Tra ve mang nhung nguoi da xem item j, T[i] = -1 la dau hieu ket thuc

*/
int* findUserRated(float **A, int m, int j, int *len)
{
    int *T  =calloc(m, sizeof(int));
    int k=0;
    for (int i=0; i<m; i++)
	if (A[i][j] != 0)
	    T[k++] = i;
 
    *len = k;
    return T;
}

/*Ham tim k vi tri co gia tri gan 1 nhat trong mang A[0...n], -1 <= A[i] <= 1
  thoa man A[i] co trong B
  -A, n: mang A gom n phan tu
  -B, m: Mang B gom len phan tu
  -k: so phan tu can tim
  -Tra ve mang ket qua, neu co gia tri -1 la ko hop le
  Giai thuat selection sort, co the cai tien*/
int* findKClosest(float A[], int n, int B[], int len, int k)
{
    typedef struct {
	int index;
	float value;
    } element;

    element* T = calloc(len, sizeof(element));


    for (int i=0; i<len; i++)
    {
	T[i].index = B[i];
	T[i].value = A[B[i]];
    }

    int max;
    element tmp;
    for (int i=0; i<len-1; i++)
    {
	max = i;
	for (int j=i+1; j<len; j++)
	    if (T[j].value > T[max].value)
		max = j;
	tmp = T[i];
	T[i] = T[max];
	T[max] = tmp;
    }

    int *C = calloc(k, sizeof(int));
    for (int i=0; i<k && i<len; i++)
	C[i] = T[i].index;
    if (len < k)
	for (int i=len; i<k; i++)
	    C[i] = -1;

    /* printf("\n\n"); */
    /* for (int i=0; i<k; i++) */
    /* 	printf("%5d", C[i]); */
    /* printf("\n\n"); */
    
    free(T);
    return C;    
}



int main(int argc, char** argv)
{
    int m, n; //Hang, cot
    float **A; //Ma tran dau vao
    float **B; //Ma tran chua sim(A[0],A[1])...sim(A[0],A[n-1]), sim(A[1],A[0]), ... sim(A[1],A[n-1]),...
    float *S;
    char* file_name;
    if (argc == 1)
	file_name = FILE_NAME_DEFAULT;
    else
	file_name = argv[1];
    
    readData(&A, &m, &n, file_name);

    #ifdef DEBUG
    printf("INPUT: \n");
    readArr(A,m,n);
    puts("");
    #endif

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

    #ifdef DEBUG
    printf("STANDARDIZED MARTRIX: \n");
    readArr(A, m, n);
    puts("");
    #endif



    /*--------------------------------------------------*/
    /*                    TINH SIM                      */
    /*--------------------------------------------------*/
    
    B = calloc(m, sizeof(float*));
    for (int i=0; i<m; i++)
	B[i] = calloc(m, sizeof(float*));

#pragma omp parallel for schedule(static, 3)
    for (int i=0; i<m; i++)
    {
#pragma omp parallel for schedule(static, 3)
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

    #ifdef DEBUG
    printf("SIMILARY MATRIX: \n");
    readArr(B,m,m);
    puts("");
    #endif

    //Doc lai bang du lieu dau vao
    readData(&A, &m, &n, file_name);



    //Song song theo cot
#pragma omp parallel for schedule(static, 3)
    for (int j=0; j<n; j++)
    {
	int *T, len;
	T = findUserRated(A, m, j, &len);

	
#pragma omp parallel for schedule(static, 3)
	for (int i=0; i<m; i++)
	    if (A[i][j] == 0)
	    {
		int *R;
		float S = 0, P = 0;
		R = findKClosest(B[i], m, T, len, K);
	        
		/* puts(""); */
		/* for (int i=0; i<K; i++) */
		/*     printf("%5d", R[i]); */
		/* puts(""); */
		
		for (int t=0; t<K; t++)
		    if (R[t] >= 0)
		    {
			P += B[i][R[t]];
			S += A[R[t]][j] * B[i][R[t]];
		    }

		/* printf("  %.2f  -   %.2f  -  %.2f", P, S, S/P); */
		/* puts(""); */
		if (P != 0)
		    A[i][j] = S/P;
	    }
    }


    #ifdef DEBUG
    printf("OUTPUT: \n");
    readArr(A, m, n);
    puts("");
    #endif

    writeData(A, m, n, "output.txt");



    



    free(A);
    free(B);
    printf("\n");
    return 0;
}
