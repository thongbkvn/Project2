#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define FILE_NAME_DEFAULT "data1.txt"
#define P 4

void createData(int m, int n, char* filename)
{
    FILE *fp = fopen(filename, "w");
    fprintf(fp, "%d %d\n", m, n);
    fprintf(fp, "\n");
    srand(time(NULL));
    for (int i=0; i<m; i++)
    {
	int p = rand() % P + 1;
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

int main(int argc, char** argv)
{
    int row = atoi(argv[1]);
    int col = atoi(argv[2]);
    char* file_name;
    if (argc < 4)
	file_name = FILE_NAME_DEFAULT;
    else
	file_name = argv[3];
    createData(row, col, file_name);
    return 0;
}
