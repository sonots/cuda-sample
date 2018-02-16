all: sum.exe max_element.exe async_reduce.exe inner_product.exe async_max_element.exe

%.exe: %.cu
	nvcc -o $@ $< $(LIBS)

clean :
	rm -f *.exe
