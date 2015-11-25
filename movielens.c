/* Collaborative fittering
   Item-item
   Pham Van Thong
*/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <unistd.h>
#include <string.h>

#define MAX_THREAD 4
#define SIM_MATRIX "data/sim_matrix.txt"
#define CONFIG_FILE "movielens.cfg"
#define DEBUG

typedef struct
{
    int mID;
    int uID;
    float P; //predict
    float R; //real rating
} Rating;



//KHAI BAO CAC BIEN TOAN CUC
float **A;//rating matrix
float **sim; //sim matrix
int nMovies;
int nUsers;
char fileFormat[10];
char trainSet[128];
char testSet[128];
int hasSimMatrix;
char pathToSimMatrix[128];
float minSim;
int K;


void loadConfig(char* filename)
{
    FILE *fp = fopen(filename, "r");
    if (fp == NULL)
    {
	perror("fopen() in loadConfig()");
	exit(1);
    }

    char buf[128];
	
    fscanf(fp, "NUMBER OF MOVIES: ");
    fgets(buf, 128, fp);
    nMovies = atoi(buf);
    
    fscanf(fp, "NUMBER OF USERS: ");
    fgets(buf, 128, fp);
    nUsers = atoi(buf);

    fscanf(fp, "FILE FORMAT: ");
    fgets(buf, 128, fp);
    strncpy(fileFormat, buf, strlen(buf)-1);
    
    fscanf(fp, "TRAIN SET: ");
    fgets(buf, 128, fp);
    strncpy(trainSet, buf, strlen(buf)-1); //do la bien toan cuc nen da co null-terminated
    
    fscanf(fp, "TEST SET: ");
    fgets(buf, 128, fp);
    strncpy(testSet, buf, strlen(buf)-1);
    
    fscanf(fp, "SIMILAR MOVIES: ");
    fgets(buf, 128, fp);
    K = atoi(buf);

    fscanf(fp, "HAS SIM MATRIX: ");
    fgets(buf, 128, fp);
    hasSimMatrix = atoi(buf);

    fscanf(fp, "PATH TO SIM MATRIX: ");
    fgets(buf, 128, fp);
    strncpy(pathToSimMatrix, buf, strlen(buf)-1);
    fscanf(fp, "MIN SIM: %f", &minSim);
    
    
    printf("\nLoad config success: ");
    printf("\nnMovies: %d        nUsers: %d", nMovies, nUsers);
    printf("\nFile format: %s", fileFormat);
    printf("\nTrain set: %s", trainSet);
    printf("\nTest set: %s", testSet);
    printf("\nSimilar movies: %d", K);
    printf("\nHas sim matrix: %d", hasSimMatrix);
    printf("\nPath to sim matrix: %s", pathToSimMatrix);
    
    fclose(fp);
}

//Ham doc du lieu tu file training
void readData(float ***outArr, int nMovies, int nUsers, char* filename)
{
    FILE *fp = fopen(filename, "r");
    if (fp == NULL)
    {
	perror("fopen()");
	exit(1);
    }

    float **A;
    A = calloc(nMovies, sizeof(float*));
    for (int i=0; i<nMovies; i++)
	A[i] = calloc(nUsers, sizeof(float));

    int uID, mID;
    long t;
    float rate;
 
    while (!feof(fp))
    {
	if (!strcmp(fileFormat, "ML10M"))
	    fscanf(fp, "%d::%d::%f::%ld", &uID, &mID, &rate, &t);

	if (!strcmp(fileFormat, "ML100K"))
	fscanf(fp, "%d%d%f%ld", &uID, &mID, &rate, &t);
	
	if (mID <= nMovies && uID <= nUsers)
	    A[mID-1][uID-1] = rate;
    }
    
    *outArr = A;
}



//Ham doc du lieu tu file test
void readTestData(Rating **B, int *len, char* filename)
{
    FILE *fp = fopen(filename, "r");
    if (fp == NULL)
    {
	perror("fopen() in readTestData()");
	exit(1);
    }

    int n=0;
    char buff[1024];
    while (!feof(fp))
    {
	fgets(buff, 1024, fp);
	n++;
    }

    int nRating = n-1;// = countLines(filename);
    Rating *T = calloc(nRating, sizeof(Rating));
   
    
    int uID, mID, j=0;
    long t;
    float rate;

    rewind(fp);

    for (int i=0; i<nRating; i++)
    {
	if (!strcmp(fileFormat, "ML10M"))
	    fscanf(fp, "%d::%d::%f::%ld", &uID, &mID, &rate, &t);

	if (!strcmp(fileFormat, "ML100K"))
	fscanf(fp, "%d%d%f%ld", &uID, &mID, &rate, &t);
	
	if (mID <= nMovies && uID <= nUsers)
	{
	    T[j].uID = uID;
	    T[j].mID = mID;
	    T[j].R = rate;
	    j++;
	}
    }
    
    *B = T;
    *len = j; //Vi chi xet nhung phan tu hop le

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
	    printf("%6.2f", A[i][j]);
    }
    puts("");
}

//Ham in ma tran vao file
void writeDataToFile(float **A, int row, int col, char* filename)
{
    FILE *fp = fopen(filename, "w");
    if (fp == NULL)
    {
	perror("fopen() in writeDataToFile()");
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

void freeArr(float **A, int row)
{
    for (int i=0; i<row; i++)
	free(A[i]);
    free(A);
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
    
    int i;
    for (i=0; i<k && i<len; i++)
    {
	if (T[i].value < minSim)
	    break;
	C[i] = T[i].index;
    }

    for (; i<k; i++)
	C[i] = -1;
    
    /* if (len < k) */
    /* 	for (int i=len; i<k; i++) */
    /* 	    C[i] = -1; */

    /* printf("\n\n"); */
    /* for (int i=0; i<k; i++) */
    /* 	printf("%5d", C[i]); */
    /* printf("\n\n"); */
    
    free(T);
    return C;    
}

//Du doan rating
void ratePredict(Rating *R, float** A, float **sim)
{
    int m = (*R).mID, u = (*R).uID;
    int *userRateSet, len;
    int *simSet;//length of simSet = K
    float S = 0, P = 0;
    int t;

    userRateSet = findUserRated(A, nMovies, u, &len);
    
    simSet = findKClosest(sim[m], nMovies, userRateSet, len, K);
    for (int i=0; i<K; i++)
    {
	//Truong hop khong du k nguoi tuong tu thi R[i] = -1
	if (simSet[i] < 0)
	    break;
	
	t = simSet[i];//Phim tuong tu voi phim m
	P += sim[m][t];
	S += A[t][u] * sim[m][t];
    }

    if (P != 0)
	(*R).P = S/P;
    free(simSet);
    free(userRateSet);
}

void computeAvgRating(float **AVG, float **A, int nMovies, int nUsers)
{
    float *T;
    T = calloc(nUsers, sizeof(float));
#pragma omp parallel for schedule(static, 3)
    for (int j=0; j<nUsers; j++)
    {
	float S = 0;
	int P = 0;
	for (int i=0; i<nMovies; i++)
	{
	    S += A[i][j];
	    P ++;
	}

	if (T != 0)
	    T[j] = S/P;
	else
	    T[j] = 3; //rating trung binh	    
    }
	
    *AVG = T;
}

void loadArrFromFile(float ***outArr, int *row, int *colmn, char* filename)
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

void writeResultToFile(Rating* B, int len, char* filename)
{
    FILE *fp = fopen(filename, "w");
    if (fp == NULL)
    {
	perror("Error fopen() in writeResultToFile");
	exit(1);
    }

    for (int i=0; i<len; i++)
    {
	fprintf(fp, "%d::%d::%.2f::%.2f\n",B[i].uID, B[i].mID, B[i].R, B[i].P);
    }
    fclose(fp);
}








	
int main(int argc, char** argv)
{
    loadConfig(CONFIG_FILE);

    if (!hasSimMatrix)
    {
	readData(&A, nMovies, nUsers, trainSet);

#ifdef DEBUG
	readArr(A, nMovies, nUsers);
#endif

	omp_set_num_threads(MAX_THREAD);


    
	/*        CHUAN HOA       */
	printf("\n\nChuan hoa: ");
#pragma omp parallel for
	for (int i=0; i<nMovies; i++)
	{
	    float sum = 0; //Tong cac rating
	    int t = 0;     //So rating
	    
#pragma omp parallel for schedule(static, 3) reduction(+:sum)
	    for (int j=0; j<nUsers; j++)
	    {
		if (A[i][j] != 0)
		{
		    sum += A[i][j];
		    t++;
		}
	    }

	    float rowMean = sum/t;
	    
#pragma omp parallel for schedule(static, 3)
	    for (int j=0; j<nUsers; j++)
	    {
		if (A[i][j] != 0)
		    A[i][j] -= rowMean;
	    }

	    printf("\n%d/%d", i, nMovies);
	}


#ifdef DEBUG
	puts("");
	printf("\nSTANDARDIZED MARTRIX: \n");
	readArr(A, nMovies, nUsers);
	puts("");
#endif



	/*         TINH SIMILARY MATRIX           */
	printf("\n\nTinh sim: ");    sim = calloc(nMovies, sizeof(float*));
	for (int i=0; i<nMovies; i++)
	    sim[i] = calloc(nMovies, sizeof(float*));

#pragma omp parallel for schedule(static, 3)
	for (int i=0; i<nMovies; i++)
	{
#pragma omp parallel for schedule(static, 3)
	    for (int j=i+1; j<nMovies; j++)
	    {
		sim[i][j] = cosine(A[i], A[j], nUsers);
	    }
	    printf("\n%d/%d: ", i, nMovies);
	}

	for (int i=0; i<nMovies; i++)
	    for (int j=0; j<=i; j++)
	    {
		if (i == j)
		    sim[i][j] = 1;
		else
		    sim[i][j] = sim[j][i];
	    }

	writeDataToFile(sim, nMovies, nMovies, pathToSimMatrix);
	freeArr(A, nMovies);
    }
    else
    {
	int tmp;
	loadArrFromFile(&sim, &tmp, &tmp, pathToSimMatrix);
    }
    
#ifdef DEBUG
    puts("");
    printf("\nSIMILARY MATRIX: \n");
    readArr(sim, nMovies, nMovies);
    puts("");
#endif


    
    /*          TINH RATING            */
    Rating *B;
    int len;
    readTestData(&B, &len, testSet);
    readData(&A, nMovies, nUsers, trainSet);
    float *AVG;
    computeAvgRating(&AVG, A, nMovies, nUsers);

#pragma omp parallel for schedule(static, 3)
    for (int i=0; i<len; i++)
    {
	ratePredict(&B[i], A, sim);
	if (B[i].P == 0)
	    B[i].P = AVG[B[i].uID];
    }

    writeResultToFile(B, len, "data/result.txt");

    float T = 0;
    float temp;
#pragma omp parallel for schedule(static, 3) reduction(+:T) private(temp)
    for (int i=0; i<len; i++)
    {
	temp = B[i].R - B[i].P;
	T += temp * temp;
    }

    printf("\n %f", sqrt(T/len));
    
#ifdef DEBUG
    puts("");
    printf("RESULT: ");
    for (int i=0; i<len; i++)
    {
	printf("\nmID: %5d    uID: %5d     Real: %6.2fd      Predict: %6.2f", B[i].mID, B[i].uID, B[i].R, B[i].P);
	usleep(1000000);
    }
    puts("");
#endif

    


    free(AVG);
    freeArr(sim, nMovies);
    free(B);
    freeArr(A, nMovies);
    puts("");
    return 0;
}
