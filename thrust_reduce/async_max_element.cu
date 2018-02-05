#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/system/cuda/execution_policy.h>
#include <cassert>
#include <thrust/extrema.h>

#if __cplusplus >= 201103L
#include <future>
#endif

template<typename Iterator, typename Reference>
__global__ void max_element_kernel(Iterator begin, Iterator end, Reference idx)
{
    Iterator iter = thrust::max_element(thrust::cuda::par, begin, end);
    idx = (size_t)(iter - begin);
}

int my_rand(void)
{
  static thrust::default_random_engine rng;
  static thrust::uniform_int_distribution<int> dist(0, 9999);
  return dist(rng);
}

int main()
{
  // generate random data on the host
  thrust::host_vector<int> h_vec(100);
  thrust::generate(h_vec.begin(), h_vec.end(), my_rand);

  // transfer to device
  thrust::device_vector<int> d_vec = h_vec;
  thrust::device_vector<size_t> idx(1);

  max_element_kernel<<<1,1>>>(d_vec.begin(), d_vec.end(), idx[0]);

  // our result should be ready
  cudaDeviceSynchronize();
  std::cout << idx[0] << std::endl;

  return 0;
}
