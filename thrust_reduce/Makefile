TARGET = sum

all : $(TARGET)

$(TARGET) : sum.cu
	nvcc $(INCLUDE) -o $(TARGET) sum.cu $(LIBS) 

run :
	./sum

clean :
	rm -f $(TARGET)
