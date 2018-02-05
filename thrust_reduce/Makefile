all : sum max_element async_reduce inner_product async_max_element

sum : sum.cu
	nvcc $(INCLUDE) -o $@ $< $(LIBS)

max_element : max_element.cu
	nvcc $(INCLUDE) -o $@ $< $(LIBS)

async_reduce : async_reduce.cu
	nvcc $(INCLUDE) -o $@ $< $(LIBS)

async_max_element : async_max_element.cu
	nvcc $(INCLUDE) -o $@ $< $(LIBS)

inner_product : inner_product.cu
	nvcc $(INCLUDE) -o $@ $< $(LIBS)

clean :
	rm -f sum max_element async_reduce inner_product async_max_element
