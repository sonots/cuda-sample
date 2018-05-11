all: indexer1.ptx indexer2.ptx indexer3.ptx indexer4.ptx

%.exe: %.cu
	nvcc -o $@ $< $(LIBS)

%.ptx: %.cu
	nvcc -std=c++14 -ptx -c -o $@ $< $(LIBS)

clean :
	rm -f *.exe *.ptx
