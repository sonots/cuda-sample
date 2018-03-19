all: cupy_sum.exe

%.exe: %.cu
	nvcc -o $@ $< $(LIBS)

clean :
	rm -f *.exe
