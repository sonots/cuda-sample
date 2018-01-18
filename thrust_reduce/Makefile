all : sum async_reduce inner_product

sum : sum.cu
	nvcc $(INCLUDE) -o $@ $< $(LIBS)

async_reduce : async_reduce.cu
	nvcc $(INCLUDE) -o $@ $< $(LIBS)

inner_product : inner_product.cu
	nvcc $(INCLUDE) -o $@ $< $(LIBS)

clean :
	rm -f sum async_reduce inner_product
