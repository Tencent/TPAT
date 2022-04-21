#include "tpat_test_lstm_bs1.h"
#include <cuda_runtime.h>
#include <thread>
#include <stdio.h>
#include <nvfunctional>
#include <chrono>

#define BLOCKSIZE_X 16
#define BLOCKSIZE_Y 16

using namespace nvinfer1;
using namespace plugin;

// CUDA Runtime error messages
#ifdef __DRIVER_TYPES_H__
static const char *_cudaGetErrorEnum(cudaError_t error)
{
  return cudaGetErrorName(error);
}
#endif

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line)
{
  if (result)
  {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    exit(EXIT_FAILURE);
  }
}
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)



#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = int;
  using uint64_t = unsigned int;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t int
  #define uint64_t unsigned int
#endif
extern "C" __global__ void tvmgen_default_fused_split_kernel0(float* __restrict__ T_split_sections, float* __restrict__ placeholder) {
  T_split_sections[(0)] = placeholder[(0)];
}

extern "C" __global__ void __launch_bounds__(5) tvmgen_default_fused_squeeze_concatenate_kernel0(float* __restrict__ T_concat, float* __restrict__ placeholder, float* __restrict__ placeholder1) {
  T_concat[(((int)threadIdx.x))] = ((1 <= ((int)threadIdx.x)) ? placeholder[((((int)threadIdx.x) - 1))] : placeholder1[(((int)threadIdx.x))]);
}

extern "C" __global__ void __launch_bounds__(4) tvmgen_default_fused_split_1_kernel2(float* __restrict__ T_split_sections, float* __restrict__ placeholder) {
  T_split_sections[(((int)threadIdx.x))] = placeholder[((((int)threadIdx.x) + 8))];
}

extern "C" __global__ void __launch_bounds__(4) tvmgen_default_fused_expand_dims_kernel0(float* __restrict__ T_expand_dims, float* __restrict__ placeholder) {
  T_expand_dims[(((int)threadIdx.x))] = placeholder[(((int)threadIdx.x))];
}

extern "C" __global__ void __launch_bounds__(64) tvmgen_default_fused_nn_dense_add_add_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_add, float* __restrict__ placeholder2, float* __restrict__ placeholder3) {
  float T_matmul_NT_rf[1];
  __shared__ float red_buf0[64];
  __shared__ float T_matmul_NT[1];
  T_matmul_NT_rf[(0)] = 0.000000e+00f;
  if (((int)threadIdx.x) < 5) {
    T_matmul_NT_rf[(0)] = (T_matmul_NT_rf[(0)] + (placeholder[(((int)threadIdx.x))] * placeholder1[(((((int)blockIdx.x) * 5) + ((int)threadIdx.x)))]));
  }
  __syncthreads();
  ((volatile float*)red_buf0)[(((int)threadIdx.x))] = T_matmul_NT_rf[(0)];
  __syncthreads();
  if (((int)threadIdx.x) < 32) {
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 32))]);
  }
  __syncthreads();
  if (((int)threadIdx.x) < 16) {
    float w_16_0 = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 16))]);
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = w_16_0;
    float w_8_0 = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 8))]);
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = w_8_0;
    float w_4_0 = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 4))]);
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = w_4_0;
    float w_2_0 = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 2))]);
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = w_2_0;
    float w_1_0 = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 1))]);
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = w_1_0;
  }
  __syncthreads();
  if (((int)threadIdx.x) == 0) {
    T_matmul_NT[(0)] = ((volatile float*)red_buf0)[(0)];
  }
  if (((int)threadIdx.x) == 0) {
    T_add[(((int)blockIdx.x))] = ((T_matmul_NT[(0)] + placeholder2[(((int)blockIdx.x))]) + placeholder3[(((int)blockIdx.x))]);
  }
}

extern "C" __global__ void __launch_bounds__(4) tvmgen_default_fused_split_1_kernel3(float* __restrict__ T_split_sections, float* __restrict__ placeholder) {
  T_split_sections[(((int)threadIdx.x))] = placeholder[((((int)threadIdx.x) + 12))];
}

extern "C" __global__ void tvmgen_default_fused_split_kernel1(float* __restrict__ T_split_sections, float* __restrict__ placeholder) {
  T_split_sections[(0)] = placeholder[(1)];
}

extern "C" __global__ void __launch_bounds__(4) tvmgen_default_fused_split_1_kernel0(float* __restrict__ T_split_sections, float* __restrict__ placeholder) {
  T_split_sections[(((int)threadIdx.x))] = placeholder[(((int)threadIdx.x))];
}

extern "C" __global__ void __launch_bounds__(4) tvmgen_default_fused_sigmoid_tanh_multiply_kernel0(float* __restrict__ T_multiply, float* __restrict__ placeholder, float* __restrict__ placeholder1) {
  T_multiply[(((int)threadIdx.x))] = ((1.000000e+00f / (1.000000e+00f + __expf((0.000000e+00f - placeholder[(((int)threadIdx.x))])))) * tanhf(placeholder1[(((int)threadIdx.x))]));
}

extern "C" __global__ void __launch_bounds__(5) tvmgen_default_fused_squeeze_concatenate_1_kernel0(float* __restrict__ T_concat, float* __restrict__ placeholder, float* __restrict__ placeholder1) {
  T_concat[(((int)threadIdx.x))] = ((1 <= ((int)threadIdx.x)) ? placeholder[((((int)threadIdx.x) - 1))] : placeholder1[(((int)threadIdx.x))]);
}

extern "C" __global__ void __launch_bounds__(4) tvmgen_default_fused_split_1_kernel1(float* __restrict__ T_split_sections, float* __restrict__ placeholder) {
  T_split_sections[(((int)threadIdx.x))] = placeholder[((((int)threadIdx.x) + 4))];
}

extern "C" __global__ void __launch_bounds__(8) tvmgen_default_fused_stack_expand_dims_kernel0(float* __restrict__ T_expand_dims, float* __restrict__ placeholder, float* __restrict__ placeholder1) {
  T_expand_dims[(((int)threadIdx.x))] = (((((int)threadIdx.x) >> 2) == 1) ? placeholder[((((int)threadIdx.x) & 3))] : placeholder1[((((int)threadIdx.x) & 3))]);
}

extern "C" __global__ void __launch_bounds__(4) tvmgen_default_fused_sigmoid_multiply_sigmoid_tanh_multiply_add_kernel0(float* __restrict__ T_add, float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ placeholder2, float* __restrict__ placeholder3) {
  T_add[(((int)threadIdx.x))] = (((1.000000e+00f / (1.000000e+00f + __expf((0.000000e+00f - placeholder[(((int)threadIdx.x))])))) * placeholder1[(((int)threadIdx.x))]) + ((1.000000e+00f / (1.000000e+00f + __expf((0.000000e+00f - placeholder2[(((int)threadIdx.x))])))) * tanhf(placeholder3[(((int)threadIdx.x))])));
}



PluginFieldCollection tpat_test_lstm_bs1Creator::mFC{};
std::vector<PluginField> tpat_test_lstm_bs1Creator::mPluginAttributes;

int tpat_test_lstm_bs1::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
    
    const float constant_3[4] = { 0.0 ,0.0 ,0.0 ,0.0  };
    checkCudaErrors(cudaMemcpyAsync((workspace + 20), &constant_3, 4 * sizeof(float), cudaMemcpyHostToDevice, stream));
    
    const float constant_5[80] = { -0.359612 ,0.12187034 ,0.40744418 ,0.055673122 ,-0.23598936 ,0.19718719 ,-0.3880626 ,-0.25875205 ,-0.0203318 ,0.35245568 ,0.25677764 ,-0.3080562 ,0.042511463 ,-0.4102647 ,0.41951877 ,-0.3953541 ,0.2518921 ,-0.40849596 ,0.4531693 ,-0.05447173 ,0.24137652 ,-0.4751196 ,-0.13000393 ,-0.0737738 ,0.4529441 ,0.52328056 ,-0.10926381 ,0.36876613 ,-0.40898925 ,0.02022028 ,-0.25830767 ,0.19898218 ,-0.33739698 ,-0.06393796 ,0.32226884 ,0.12574333 ,0.2687691 ,-0.25348523 ,-0.35659266 ,-0.1506598 ,0.47657627 ,0.12069249 ,-0.45934454 ,-0.33851504 ,0.39704776 ,0.17563194 ,-0.17199594 ,-0.31947905 ,-0.115478426 ,-0.4918034 ,0.22508967 ,-0.18123522 ,-0.23298904 ,0.46371907 ,-0.36562777 ,-0.31080586 ,-0.06718668 ,-0.33123899 ,-0.20640528 ,0.5311995 ,0.5067504 ,-0.49210507 ,-0.46982205 ,0.11495352 ,-0.14386049 ,0.0047035813 ,0.024517536 ,0.1988467 ,0.01115495 ,0.32315785 ,-0.14157778 ,0.13437283 ,0.2783149 ,0.056137085 ,0.28821647 ,0.40561646 ,-0.22218439 ,0.17320204 ,-0.44786054 ,-0.16016951  };
    checkCudaErrors(cudaMemcpyAsync((workspace + 36), &constant_5, 80 * sizeof(float), cudaMemcpyHostToDevice, stream));
    
    const float constant_6[16] = { 0.0 ,0.0 ,0.0 ,0.0 ,1.0 ,1.0 ,1.0 ,1.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0  };
    checkCudaErrors(cudaMemcpyAsync((workspace + 356), &constant_6, 16 * sizeof(float), cudaMemcpyHostToDevice, stream));
    
    const float constant_7[16] = { 0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0  };
    checkCudaErrors(cudaMemcpyAsync((workspace + 420), &constant_7, 16 * sizeof(float), cudaMemcpyHostToDevice, stream));
    
    const float constant_13[4] = { 0.0 ,0.0 ,0.0 ,0.0  };
    checkCudaErrors(cudaMemcpyAsync((workspace + 516), &constant_13, 4 * sizeof(float), cudaMemcpyHostToDevice, stream));
    
    const float constant_17[80] = { -0.359612 ,0.12187034 ,0.40744418 ,0.055673122 ,-0.23598936 ,0.19718719 ,-0.3880626 ,-0.25875205 ,-0.0203318 ,0.35245568 ,0.25677764 ,-0.3080562 ,0.042511463 ,-0.4102647 ,0.41951877 ,-0.3953541 ,0.2518921 ,-0.40849596 ,0.4531693 ,-0.05447173 ,0.24137652 ,-0.4751196 ,-0.13000393 ,-0.0737738 ,0.4529441 ,0.52328056 ,-0.10926381 ,0.36876613 ,-0.40898925 ,0.02022028 ,-0.25830767 ,0.19898218 ,-0.33739698 ,-0.06393796 ,0.32226884 ,0.12574333 ,0.2687691 ,-0.25348523 ,-0.35659266 ,-0.1506598 ,0.47657627 ,0.12069249 ,-0.45934454 ,-0.33851504 ,0.39704776 ,0.17563194 ,-0.17199594 ,-0.31947905 ,-0.115478426 ,-0.4918034 ,0.22508967 ,-0.18123522 ,-0.23298904 ,0.46371907 ,-0.36562777 ,-0.31080586 ,-0.06718668 ,-0.33123899 ,-0.20640528 ,0.5311995 ,0.5067504 ,-0.49210507 ,-0.46982205 ,0.11495352 ,-0.14386049 ,0.0047035813 ,0.024517536 ,0.1988467 ,0.01115495 ,0.32315785 ,-0.14157778 ,0.13437283 ,0.2783149 ,0.056137085 ,0.28821647 ,0.40561646 ,-0.22218439 ,0.17320204 ,-0.44786054 ,-0.16016951  };
    checkCudaErrors(cudaMemcpyAsync((workspace + 548), &constant_17, 80 * sizeof(float), cudaMemcpyHostToDevice, stream));
    
    dim3 dimBlock, dimGrid;
    
    dimGrid = dim3(1,1,1);
    dimBlock = dim3(1,1,1);
    tvmgen_default_fused_split_kernel0<<<dimGrid, dimBlock, 0, stream>>>((float*)workspace, (float*)inputs[0]);
    
    dimGrid = dim3(1,1,1);
    dimBlock = dim3(1,1,1);
    tvmgen_default_fused_split_kernel1<<<dimGrid, dimBlock, 0, stream>>>((float*)(workspace + 4), (float*)inputs[0]);
    
    dimGrid = dim3(1,1,1);
    dimBlock = dim3(5,1,1);
    tvmgen_default_fused_squeeze_concatenate_kernel0<<<dimGrid, dimBlock, 0, stream>>>((float*)outputs[0], (float*)(workspace + 20), (float*)workspace);
    
    dimGrid = dim3(16,1,1);
    dimBlock = dim3(64,1,1);
    tvmgen_default_fused_nn_dense_add_add_kernel0<<<dimGrid, dimBlock, 0, stream>>>((float*)outputs[0], (float*)(workspace + 36), (float*)outputs[1], (float*)(workspace + 356), (float*)(workspace + 420));
    
    dimGrid = dim3(1,1,1);
    dimBlock = dim3(4,1,1);
    tvmgen_default_fused_split_1_kernel0<<<dimGrid, dimBlock, 0, stream>>>((float*)outputs[0], (float*)outputs[1]);
    
    dimGrid = dim3(1,1,1);
    dimBlock = dim3(4,1,1);
    tvmgen_default_fused_split_1_kernel1<<<dimGrid, dimBlock, 0, stream>>>((float*)(workspace + 484), (float*)outputs[1]);
    
    dimGrid = dim3(1,1,1);
    dimBlock = dim3(4,1,1);
    tvmgen_default_fused_split_1_kernel2<<<dimGrid, dimBlock, 0, stream>>>((float*)(workspace + 500), (float*)outputs[1]);
    
    dimGrid = dim3(1,1,1);
    dimBlock = dim3(4,1,1);
    tvmgen_default_fused_split_1_kernel3<<<dimGrid, dimBlock, 0, stream>>>((float*)outputs[2], (float*)outputs[1]);
    
    dimGrid = dim3(1,1,1);
    dimBlock = dim3(4,1,1);
    tvmgen_default_fused_sigmoid_multiply_sigmoid_tanh_multiply_add_kernel0<<<dimGrid, dimBlock, 0, stream>>>((float*)outputs[1], (float*)(workspace + 484), (float*)(workspace + 516), (float*)outputs[0], (float*)(workspace + 500));
    
    dimGrid = dim3(1,1,1);
    dimBlock = dim3(4,1,1);
    tvmgen_default_fused_sigmoid_tanh_multiply_kernel0<<<dimGrid, dimBlock, 0, stream>>>((float*)(workspace + 532), (float*)outputs[2], (float*)outputs[1]);
    
    dimGrid = dim3(1,1,1);
    dimBlock = dim3(5,1,1);
    tvmgen_default_fused_squeeze_concatenate_1_kernel0<<<dimGrid, dimBlock, 0, stream>>>((float*)outputs[0], (float*)(workspace + 532), (float*)(workspace + 4));
    
    dimGrid = dim3(16,1,1);
    dimBlock = dim3(64,1,1);
    tvmgen_default_fused_nn_dense_add_add_kernel0<<<dimGrid, dimBlock, 0, stream>>>((float*)outputs[0], (float*)(workspace + 548), (float*)outputs[2], (float*)(workspace + 356), (float*)(workspace + 420));
    
    dimGrid = dim3(1,1,1);
    dimBlock = dim3(4,1,1);
    tvmgen_default_fused_split_1_kernel0<<<dimGrid, dimBlock, 0, stream>>>((float*)(workspace + 484), (float*)outputs[2]);
    
    dimGrid = dim3(1,1,1);
    dimBlock = dim3(4,1,1);
    tvmgen_default_fused_split_1_kernel1<<<dimGrid, dimBlock, 0, stream>>>((float*)(workspace + 500), (float*)outputs[2]);
    
    dimGrid = dim3(1,1,1);
    dimBlock = dim3(4,1,1);
    tvmgen_default_fused_split_1_kernel2<<<dimGrid, dimBlock, 0, stream>>>((float*)outputs[0], (float*)outputs[2]);
    
    dimGrid = dim3(1,1,1);
    dimBlock = dim3(4,1,1);
    tvmgen_default_fused_split_1_kernel3<<<dimGrid, dimBlock, 0, stream>>>((float*)(workspace + 4), (float*)outputs[2]);
    
    dimGrid = dim3(1,1,1);
    dimBlock = dim3(4,1,1);
    tvmgen_default_fused_sigmoid_multiply_sigmoid_tanh_multiply_add_kernel0<<<dimGrid, dimBlock, 0, stream>>>((float*)outputs[2], (float*)(workspace + 500), (float*)outputs[1], (float*)(workspace + 484), (float*)outputs[0]);
    
    dimGrid = dim3(1,1,1);
    dimBlock = dim3(4,1,1);
    tvmgen_default_fused_sigmoid_tanh_multiply_kernel0<<<dimGrid, dimBlock, 0, stream>>>((float*)outputs[1], (float*)(workspace + 4), (float*)outputs[2]);
    
    dimGrid = dim3(1,1,1);
    dimBlock = dim3(8,1,1);
    tvmgen_default_fused_stack_expand_dims_kernel0<<<dimGrid, dimBlock, 0, stream>>>((float*)outputs[0], (float*)outputs[1], (float*)(workspace + 532));
    
}

REGISTER_TENSORRT_PLUGIN(tpat_test_lstm_bs1Creator);