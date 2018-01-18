#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>

int main(void)
{
  float vec1[3] = {1.0f, 2.0f, 5.0f};
  float vec2[3] = {4.0f, 1.0f, 5.0f}; 
  float result = thrust::inner_product(vec1, vec1 + 3, vec2, 0.0f, thrust::plus<float>(), thrust::multiplies<float>());

  // print the sum
  std::cout << "reulst is " << result << std::endl;

  return 0;
}
