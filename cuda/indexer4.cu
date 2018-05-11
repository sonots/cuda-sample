#include <stdio.h>

#define NDIM 2


template <int8_t kNdim>
class IndexIterator {
public:
    __host__ __device__ void Set(int64_t i) {
        int64_t t = 0;
        for (int8_t j = kNdim; --j >= 0;) {
            t = i / shape_[j];
            index_[j] = i - t * shape_[j];
            i = t;
        }
    }

    __host__ __device__ int64_t* index() { return index_; }

private:
    //const int64_t* shape_;
    int64_t shape_[kNdim] = {1000, 768};
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
