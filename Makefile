CC = gcc
CFLAGS = -fopenmp -g -Wall

all: lab

lab: main.c
	$(CC) $(CFLAGS) main.c -o lab

clean:
	rm -f lab
