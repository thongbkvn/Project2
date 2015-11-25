all: main create test
main: main.c
	gcc -fopenmp main.c -o main -lm
create: create.c
	gcc create.c -o create
test: test.c
	gcc -fopenmp test.c -o test -lm
