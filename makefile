all: test
test: test.c
	gcc -fopenmp test.c -o test -lm
