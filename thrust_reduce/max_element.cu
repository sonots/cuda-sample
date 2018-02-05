#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/extrema.h>

int my_rand(void)
{
  static thrust::default_random_engine rng;
  static thrust::uniform_int_distribution<int> dist(0, 9999);
  return dist(rng);
}

int main(void)
{
  // generate random data on the host
  thrust::host_vector<int> h_vec(100);
  thrust::generate(h_vec.begin(), h_vec.end(), my_rand);

  // transfer to device
  thrust::device_vector<int> d_vec = h_vec;

  // max element
  thrust::device_vector<int>::iterator iter = thrust::max_element(d_vec.begin(), d_vec.end());
  size_t idx = (size_t)(iter - d_vec.begin());
  int max = *iter;

  // print the sum
  std::cout << "idx: " << idx << " max: " << max << std::endl;

  return 0;
}
