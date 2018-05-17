// Reference: https://github.com/cupy/cupy
// MIT License

#include "carray.cuh"
#include <stdio.h>
#include <iostream>

typedef long long type_in0_raw; typedef long long type_out0_raw;

#define REDUCE(a, b) (a + b)
#define POST_MAP(a) (out0 = type_out0_raw(a))
#define _REDUCE(_offset) \
    if (_tid < _offset) { \
        _type_reduce _a = _sdata[_tid], _b = _sdata[(_tid + _offset)]; \
        _sdata[_tid] = REDUCE(_a, _b); \
    }

typedef long long _type_reduce;
extern "C" __global__ void cupy_sum(CArray<float, 1> _raw_in0, CArray<float, 0> _raw_out0, CIndexer<1> _in_ind, CIndexer<0> _out_ind, int _block_stride) {
// extern "C" __global__ void cupy_sum(CArray<float, 2> _raw_in0, CArray<float, 1> _raw_out0, CIndexer<2> _in_ind, CIndexer<1> _out_ind, int _block_stride) {
// extern "C" __global__ void cupy_sum(CArray<float, 3> _raw_in0, CArray<float, 1> _raw_out0, CIndexer<3> _in_ind, CIndexer<1> _out_ind, int _block_stride) {
    extern __shared__ _type_reduce _sdata_raw[];
    _type_reduce *_sdata = _sdata_raw;
    unsigned int _tid = threadIdx.x;

    int _J_offset = _tid / _block_stride;
    int _j_offset = _J_offset * _out_ind.size();
    int _J_stride = 512 / _block_stride;
    long long _j_stride = (long long)_J_stride * _out_ind.size();
    for (int _i_base = blockIdx.x * _block_stride;
            _i_base < _out_ind.size();
            _i_base += gridDim.x * _block_stride) {
        _type_reduce _s = _type_reduce(0);
        int _i = _i_base + _tid % _block_stride;
        int _J = _J_offset;
        for (long long _j = _i + _j_offset; _j < _in_ind.size();
                _j += _j_stride, _J += _J_stride) {
            _in_ind.set(_j);
            const type_in0_raw in0 = _raw_in0[_in_ind.get()];
            _type_reduce _a = in0;
            _s = REDUCE(_s, _a);
        }
        if (_block_stride < 512) {
            _sdata[_tid] = _s;
            __syncthreads();
            if (_block_stride <= 256) {
                _REDUCE(256);
                __syncthreads();
                if (_block_stride <= 128) {
                    _REDUCE(128);
                    __syncthreads();
                    if (_block_stride <= 64) {
                        _REDUCE(64);
                        __syncthreads();
                        if (_block_stride <= 32) {
                            _REDUCE(32);
                            if (_block_stride <= 16) {
                                _REDUCE(16);
                                if (_block_stride <= 8) {
                                    _REDUCE(8);
                                    if (_block_stride <= 4) {
                                        _REDUCE(4);
                                        if (_block_stride <= 2) {
                                            _REDUCE(2);
                                            if (_block_stride <= 1) {
                                                _REDUCE(1);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            _s = _sdata[_tid];
            __syncthreads();
        }
        if (_J_offset == 0 && _i < _out_ind.size()) {
            _out_ind.set(_i);
            type_out0_raw &out0 = _raw_out0[_out_ind.get()];
            POST_MAP(_s);
        }
    }
}

typedef long long dtype;
// ('block_stride', 1, 'block_size', 512, 'shared_mem', 16384, 'size', 512)

int main() {
    // warm up
    {
        const int in_ndim = 1;
        ptrdiff_t in_size = 6;
        ptrdiff_t in_shape[] = {6};
        ptrdiff_t in_strides[] = {sizeof(dtype)};
        void* in_data;
        cudaMallocManaged(&in_data, in_size * sizeof(dtype), cudaMemAttachGlobal);
        for (int i = 0; i < in_size; ++i) {
            ((dtype*)in_data)[i] = i;
        }

        const int out_ndim = 0;
        ptrdiff_t out_size = 1;
        void* out_data;
        cudaMallocManaged(&out_data, out_size * sizeof(dtype), cudaMemAttachGlobal);

        CArray<dtype, in_ndim> in_array((dtype*)in_data, in_size, in_shape, in_strides);
        CArray<dtype, out_ndim> out_array((dtype*)out_data, out_size);
        CIndexer<in_ndim> in_ind(in_size, in_shape);
        CIndexer<out_ndim> out_ind(out_size);

        cupy_sum<<<1, 512, 16384>>>(in_array, out_array, in_ind, out_ind, 1);

        cudaDeviceSynchronize();
        std::cout << ((dtype*)out_data)[0] << std::endl;
    }
    {
        const int in_ndim = 1;
        ptrdiff_t in_size = 1 << 20;
        ptrdiff_t in_shape[] = {in_size};
        ptrdiff_t in_strides[] = {sizeof(dtype)};
        void* in_data;
        cudaMallocManaged(&in_data, in_size * sizeof(dtype), cudaMemAttachGlobal);
        for (int i = 0; i < in_size; ++i) {
            ((dtype*)in_data)[i] = i;
        }

        const int out_ndim = 0;
        ptrdiff_t out_size = 1;
        void* out_data;
        cudaMallocManaged(&out_data, out_size * sizeof(dtype), cudaMemAttachGlobal);

        CArray<dtype, in_ndim> in_array((dtype*)in_data, in_size, in_shape, in_strides);
        CArray<dtype, out_ndim> out_array((dtype*)out_data, out_size);
        CIndexer<in_ndim> in_ind(in_size, in_shape);
        CIndexer<out_ndim> out_ind(out_size);

        cupy_sum<<<1, 512, 16384>>>(in_array, out_array, in_ind, out_ind, 1);

        cudaDeviceSynchronize();
        std::cout << ((dtype*)out_data)[0] << std::endl;
    }
}
