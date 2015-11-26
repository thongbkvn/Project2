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
//#define DEBUG

typedef struct
{
    int mID;
    int uID;
    float P; //predict
    float R; //real rating
} Rating;
    

typedef enum {ML100K, ML1M, ML10M, ML100M} DataFormatType;

//KHAI BAO CAC BIEN TOAN CUC
float **A;//rating matrix
float **sim; //sim matrix
int nMovies;
int nUsers;
int *MI;
DataFormatType dataFormat;
char moviesData[128];
char usersData[128];
char trainSet[128];
char testSet[128];
int hasSimMatrix;
char pathToSimMatrix[128];
float minSim = 0;
int K;


void loadConfig(char* filename)
{
    FILE *fp = fopen(filename, "r");
    if (fp == NULL)
    {
	perror("fopen() in loadConfig()");
	exit(1);
    }

    char ff[10], buf[20];
	
    fscanf(fp, "Numbers of movie: ");
    fgets(buf, 20, fp);
    nMovies = atoi(buf);
    fscanf(fp, "Numbers of user: ");
    fgets(buf, 20, fp);
    nUsers = atoi(buf);
    
    fscanf(fp, "File format: ");
    fgets(ff, 10, fp);
    ff[strlen(ff)-1] = 0;
    if (!strcmp(ff, "ML100K"))
	dataFormat = ML100K;
    else if (!strcmp(ff, "ML1M"))
	dataFormat = ML1M;
    else if (!strcmp(ff, "ML10M"))
	dataFormat = ML10M;
    else
	dataFormat = ML100M;
    

    fscanf(fp, "Movies data: ");
    fgets(moviesData, 128, fp);
    moviesData[strlen(moviesData)-1] = 0;
    
    fscanf(fp, "Train set: ");
    fgets(trainSet, 128, fp);
    trainSet[strlen(trainSet)-1] = 0;
    fscanf(fp, "Test set: ");
    fgets(testSet, 128, fp);
    testSet[strlen(testSet)-1] = 0;
    
    fscanf(fp, "Similar movies: ");
    fgets(buf, 20, fp);
    K = atoi(buf);
    fscanf(fp, "Has sim matrix: ");
    fgets(buf, 20, fp);
    hasSimMatrix = atoi(buf);

    fscanf(fp, "Path to sim matrix: ");
    fgets(pathToSimMatrix, 128, fp);
    pathToSimMatrix[strlen(pathToSimMatrix)-1] = 0;
    
    
    printf("\nLoad config success: ");
    printf("\nnMovies: %d        nUsers: %d", nMovies, nUsers);
    printf("\nFile format: %s", ff);
    printf("\nMovies data: %s", moviesData);
    printf("\nTrain set: %s", trainSet);
    printf("\nTest set: %s", testSet);
    printf("\nSimilar movies: %d", K);
    printf("\nHas sim matrix: %d", hasSimMatrix);
    printf("\nPath to sim matrix: %s", pathToSimMatrix);
    
    fclose(fp);
}

int indexOfMovieID(int movieID)
{
    int m, l=0, h=nMovies - 1;
    while (l<=h)
    {
	m = (l+h)/2;
	if (movieID == MI[m])
	    return m;
	else if (movieID < MI[m])
	    h = m-1;
	else
	    l = m+1;
    }

    return -1;
}
int countLines(char *filename)
{
    FILE *fp = fopen(filename, "r");
    if (fp == NULL)
    {
	perror("fopen() in countLines");
	exit(1);
    }

    int n = 0;
    char buf[1024];
    while (!feof(fp))
    {
	fgets(buf, 1024, fp);
	n++;
    }
    fclose(fp);

    return n-1; //do loi cua fgets()
}


//Ham doc du lieu tu file training
//Tham so nMovie va nUser khong duoc su dung, du dinh cho nhung truong hop
//Mo rong chuong trinh va tai su dung
void readData(float ***outArr, int nMovie, int nUser, char* filename)
{
    nMovies = countLines(moviesData);

    MI = calloc(nMovies, sizeof(int));

    FILE *fp = fopen(moviesData, "r");
    char buf[1024];
    for (int i=0; i<nMovies; i++)
    {
	fgets(buf, 1024, fp);
	sscanf(buf, "%d", &MI[i]);
    }
    fclose(fp);

    
    fp = fopen(filename, "r");
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
    int mIndex;

    //   FILE *fp1 = fopen(moviesData, "r");
    //fclose(fp1);
 
    while (!feof(fp))
    {
	
	if (dataFormat == ML100K)
	    fscanf(fp, "%d%d%f%ld", &uID, &mID, &rate, &t);
	else if (dataFormat == ML10M)
	    fscanf(fp, "%d::%d::%f::%ld", &uID, &mID, &rate, &t);
	mIndex = indexOfMovieID(mID);
	if (mIndex < nMovies && uID <= nUsers)
	    A[mIndex][uID-1] = rate;
    }
    
    *outArr = A;
}



//Ham doc du lieu tu file test
void readTestData(Rating **B, int *len, char* filename)
{  
    int nRating = countLines(filename);
    Rating *T = calloc(nRating, sizeof(Rating));

    FILE *fp = fopen(filename, "r");
    if (fp == NULL)
    {
	perror("fopen() in readTestData()");
	exit(1);
    }

    int uID, mID, j=0;
    long t;
    float rate;
    int mIndex;

    for (int i=0; i<nRating; i++)
    {
	if (dataFormat == ML100K)
	    fscanf(fp, "%d%d%f%ld", &uID, &mID, &rate, &t);
	else if (dataFormat == ML10M)
	    fscanf(fp, "%d::%d::%f::%ld", &uID, &mID, &rate, &t);
	mIndex = indexOfMovieID(mID);
	
	if (mIndex < nMovies && uID <= nUsers)
	{
	    T[j].uID = uID-1;
	    T[j].mID = mIndex;
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
    else
	(*R).P = 0; //Khong the du doan
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
	    if (A[i][j] > 0)
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
	fprintf(fp, "%d::%d::%.2f::%.2f\n",B[i].uID+1, MI[B[i].mID], B[i].R, B[i].P);
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

	    float rowMean = 0;
	    if (t!=0)
		rowMean = sum/t;
	    
#pragma omp parallel for schedule(static, 3)
	    for (int j=0; j<nUsers; j++)
	    {
		if (A[i][j] != 0)
		    A[i][j] -= rowMean;
	    }

	    printf("\r\%-10d/%10d", i, nMovies);
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
	    printf("\r%-10d/%10d", i, nMovies);
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
	//Neu khong the du doan thi dua ve gia tri trung binh
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
