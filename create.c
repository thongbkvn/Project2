#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define FILE_NAME "data1.txt"
#define P 4

void createData(int m, int n, char* filename)
{
    FILE *fp = fopen(filename, "w");
    fprintf(fp, "%d %d\n", m, n);
    fprintf(fp, "\n");
    for (int i=0; i<m; i++)
    {
	srand(time(NULL));
	int p = rand() % 4 + 1;
	for (int j=0; j<n; j++)
	{
	    srand(time(NULL));
	    int k = rand() % p;
	    //srand(time(NULL));
	    if (k == 1)
		fprintf(fp, "%4d", rand() % 6);
	    else
		fprintf(fp, "%4d", 0);
	}
	fprintf(fp, "\n");
    }
    fclose(fp);
}

int main(int argc, char** argv)
{
    int row = atoi(argv[1]);
    int col = atoi(argv[2]);
    createData(row, col, FILE_NAME);
    return 0;
}
