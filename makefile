all: test create
test: test.c
	gcc -fopenmp test.c -o test -lm
create: create.c
	gcc create.c -o create
