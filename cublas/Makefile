LIBS = -lcublas
all: gemm_f.exe gemm_c.exe

%.exe: %.cu
	nvcc -o $@ $< $(LIBS)

clean :
	rm -f *.exe
