#include "tpat_test_lstm_without_bias.h"
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
extern "C" __global__ void __launch_bounds__(64) tvmgen_default_fused_split_kernel1(float* __restrict__ T_split_sections, float* __restrict__ placeholder) {
  T_split_sections[(((int)threadIdx.x))] = placeholder[(((((((int)threadIdx.x) >> 5) * 128) + (((int)threadIdx.x) & 31)) + 32))];
}

extern "C" __global__ void __launch_bounds__(1024) tvmgen_default_fused_split_squeeze_split_concatenate_split_squeeze_split_concatenate_concatenat_3246268966947405992__kernel0(float* __restrict__ T_concat, float* __restrict__ placeholder, float* __restrict__ placeholder1) {
  T_concat[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))] = ((16 <= (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) % 48)) ? ((4608 <= ((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))) ? placeholder[(((((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) / 48) * 32) + (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) % 48)) - 2064))] : ((3072 <= ((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))) ? placeholder[(((((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) / 48) * 32) + (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) % 48)) + 1008))] : ((1536 <= ((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))) ? placeholder[(((((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) / 48) * 32) + (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) % 48)) + 1008))] : placeholder[(((((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) / 48) * 32) + (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) % 48)) - 16))]))) : ((4608 <= ((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))) ? placeholder1[(((((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) / 48) * 16) + (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) % 48)) - 1024))] : ((3072 <= ((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))) ? placeholder1[(((((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) / 48) * 16) + (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) % 48)) + 512))] : ((1536 <= ((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))) ? placeholder1[(((((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) / 48) * 16) + (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) % 48)) + 512))] : placeholder1[((((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) / 48) * 16) + (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) % 48)))]))));
}

extern "C" __global__ void __launch_bounds__(64) tvmgen_default_fused_split_kernel0(float* __restrict__ T_split_sections, float* __restrict__ placeholder) {
  T_split_sections[(((int)threadIdx.x))] = placeholder[((((((int)threadIdx.x) >> 5) * 128) + (((int)threadIdx.x) & 31)))];
}

extern "C" __global__ void __launch_bounds__(64) tvmgen_default_fused_split_kernel3(float* __restrict__ T_split_sections, float* __restrict__ placeholder) {
  T_split_sections[(((int)threadIdx.x))] = placeholder[(((((((int)threadIdx.x) >> 5) * 128) + (((int)threadIdx.x) & 31)) + 96))];
}

extern "C" __global__ void __launch_bounds__(64) tvmgen_default_fused_nn_dense_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_matmul_NT) {
  float T_matmul_NT_rf[1];
  __shared__ float red_buf0[64];
  T_matmul_NT_rf[(0)] = 0.000000e+00f;
  if (((int)threadIdx.x) < 48) {
    T_matmul_NT_rf[(0)] = (T_matmul_NT_rf[(0)] + (placeholder[(((((int)blockIdx.y) * 48) + ((int)threadIdx.x)))] * placeholder1[(((((int)blockIdx.x) * 48) + ((int)threadIdx.x)))]));
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
    T_matmul_NT[(((((int)blockIdx.y) * 128) + ((int)blockIdx.x)))] = ((volatile float*)red_buf0)[(0)];
  }
}

extern "C" __global__ void __launch_bounds__(64) tvmgen_default_fused_multiply_add_clip_tanh_multiply_kernel0(float* __restrict__ T_multiply, float* __restrict__ placeholder, float* __restrict__ placeholder1) {
  T_multiply[(((int)threadIdx.x))] = (max(min(((placeholder[(((int)threadIdx.x))] * 2.000000e-01f) + 5.000000e-01f), 1.000000e+00f), 0.000000e+00f) * tanhf(placeholder1[(((int)threadIdx.x))]));
}

extern "C" __global__ void __launch_bounds__(64) tvmgen_default_fused_expand_dims_kernel0(float* __restrict__ T_expand_dims, float* __restrict__ placeholder) {
  T_expand_dims[(((int)threadIdx.x))] = placeholder[(((int)threadIdx.x))];
}

extern "C" __global__ void __launch_bounds__(64) tvmgen_default_fused_multiply_add_clip_zeros_split_squeeze_multiply_multiply_add_clip_tanh_mult_8552778803643229406__kernel0(float* __restrict__ T_add, float* __restrict__ placeholder, float* __restrict__ placeholder1) {
  T_add[(((int)threadIdx.x))] = (max(min(((placeholder[(((int)threadIdx.x))] * 2.000000e-01f) + 5.000000e-01f), 1.000000e+00f), 0.000000e+00f) * tanhf(placeholder1[(((int)threadIdx.x))]));
}

extern "C" __global__ void __launch_bounds__(64) tvmgen_default_fused_stack_expand_dims_kernel0(float* __restrict__ T_expand_dims, float* __restrict__ placeholder) {
  T_expand_dims[(((int)threadIdx.x))] = placeholder[(((int)threadIdx.x))];
}

extern "C" __global__ void __launch_bounds__(64) tvmgen_default_fused_split_kernel2(float* __restrict__ T_split_sections, float* __restrict__ placeholder) {
  T_split_sections[(((int)threadIdx.x))] = placeholder[(((((((int)threadIdx.x) >> 5) * 128) + (((int)threadIdx.x) & 31)) + 64))];
}

extern "C" __global__ void __launch_bounds__(96) tvmgen_default_fused_split_squeeze_zeros_split_squeeze_concatenate_kernel0(float* __restrict__ T_concat, float* __restrict__ placeholder) {
  T_concat[(((int)threadIdx.x))] = ((16 <= (((int)threadIdx.x) % 48)) ? 0.000000e+00f : placeholder[((((((int)threadIdx.x) / 48) * 16) + (((int)threadIdx.x) % 48)))]);
}



PluginFieldCollection tpat_test_lstm_without_biasCreator::mFC{};
std::vector<PluginField> tpat_test_lstm_without_biasCreator::mPluginAttributes;

int tpat_test_lstm_without_bias::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
    
    dim3 dimBlock, dimGrid;
    
    dimGrid = dim3(1,1,1);
    dimBlock = dim3(96,1,1);
    tvmgen_default_fused_split_squeeze_zeros_split_squeeze_concatenate_kernel0<<<dimGrid, dimBlock, 0, stream>>>((float*)workspace, (float*)inputs[0]);
    
    dimGrid = dim3(6,1,1);
    dimBlock = dim3(1024,1,1);
    tvmgen_default_fused_split_squeeze_split_concatenate_split_squeeze_split_concatenate_concatenat_3246268966947405992__kernel0<<<dimGrid, dimBlock, 0, stream>>>((float*)(workspace + 384), (float*)inputs[2], (float*)inputs[1]);
    
    dimGrid = dim3(128,2,1);
    dimBlock = dim3(64,1,1);
    tvmgen_default_fused_nn_dense_kernel0<<<dimGrid, dimBlock, 0, stream>>>((float*)workspace, (float*)(workspace + 384), (float*)outputs[2]);
    
    dimGrid = dim3(1,1,1);
    dimBlock = dim3(64,1,1);
    tvmgen_default_fused_split_kernel0<<<dimGrid, dimBlock, 0, stream>>>((float*)workspace, (float*)outputs[2]);
    
    dimGrid = dim3(1,1,1);
    dimBlock = dim3(64,1,1);
    tvmgen_default_fused_split_kernel1<<<dimGrid, dimBlock, 0, stream>>>((float*)outputs[0], (float*)outputs[2]);
    
    dimGrid = dim3(1,1,1);
    dimBlock = dim3(64,1,1);
    tvmgen_default_fused_split_kernel2<<<dimGrid, dimBlock, 0, stream>>>((float*)(workspace + 24960), (float*)outputs[2]);
    
    dimGrid = dim3(1,1,1);
    dimBlock = dim3(64,1,1);
    tvmgen_default_fused_split_kernel3<<<dimGrid, dimBlock, 0, stream>>>((float*)(workspace + 25216), (float*)outputs[2]);
    
    dimGrid = dim3(1,1,1);
    dimBlock = dim3(64,1,1);
    tvmgen_default_fused_multiply_add_clip_zeros_split_squeeze_multiply_multiply_add_clip_tanh_mult_8552778803643229406__kernel0<<<dimGrid, dimBlock, 0, stream>>>((float*)outputs[2], (float*)workspace, (float*)(workspace + 24960));
    
    dimGrid = dim3(1,1,1);
    dimBlock = dim3(64,1,1);
    tvmgen_default_fused_multiply_add_clip_tanh_multiply_kernel0<<<dimGrid, dimBlock, 0, stream>>>((float*)outputs[1], (float*)(workspace + 25216), (float*)outputs[2]);
    
    dimGrid = dim3(1,1,1);
    dimBlock = dim3(64,1,1);
    tvmgen_default_fused_stack_expand_dims_kernel0<<<dimGrid, dimBlock, 0, stream>>>((float*)outputs[0], (float*)outputs[1]);
    
}

REGISTER_TENSORRT_PLUGIN(tpat_test_lstm_without_biasCreator);