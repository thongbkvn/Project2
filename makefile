all: main create report.pdf
main: main.c
	gcc -fopenmp main.c -o main -lm
create: create.c
	gcc create.c -o create
report.pdf: report.tex
	latex -shell-escape report.tex
