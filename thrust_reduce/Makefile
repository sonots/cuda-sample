all : sum async_reduce

sum : sum.cu
	nvcc $(INCLUDE) -o $@ $< $(LIBS)

async_reduce : async_reduce.cu
	nvcc $(INCLUDE) -o $@ $< $(LIBS)

clean :
	rm -f sum async_reduce
