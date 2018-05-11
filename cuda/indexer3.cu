#include <stdio.h>

#define NDIM 2

template <int8_t kNdim>
class IndexIterator {
public:
    __host__ __device__ void Set(int64_t i) {
        index_[1] = i % shape_[1];
        i /= shape_[1];
        
        index_[0] = i % shape_[0];
    }

    __host__ __device__ int64_t* index() { return index_; }

private:
    //const int64_t* shape_;
    int64_t shape_[kNdim];
    //int64_t total_size_{};
    //int64_t raw_index_{};
    //int64_t step_{};
    int64_t index_[kNdim];
};

__global__ void test(IndexIterator<NDIM> i) {
    i.Set(1);
    printf("1\n", i.index()[0]);
}

int main() {
    IndexIterator<NDIM> i{};
    test<<<1,1>>>(i);
}
