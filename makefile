all: main create test ml10m
main: main.c
	gcc -fopenmp main.c -o main -lm
create: create.c
	gcc create.c -o create
test: test.c
	gcc -fopenmp test.c -o test -lm
ml10m: ml10m.c
	gcc -fopenmp ml10m.c -o ml10m -lm
