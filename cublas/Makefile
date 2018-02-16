LIBS = -lcublas
all: gemm.exe

%.exe: %.cu
	nvcc -o $@ $< $(LIBS)

clean :
	rm -f *.exe
