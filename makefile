CC=nvcc
CFLAGS=-I.

%.o: %.c $(DEPS)
	$(CC) -O2 -c -o $@ $< $(CFLAGS)

lab1: main.o
			nvcc -arch=sm_30 -o lab1 main.o

main.o: main.cu
			nvcc -arch=sm_30 -c main.cu

clean:
			rm -r *.o lab1