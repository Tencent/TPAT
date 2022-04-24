#include "tpat_test_conv_with_autopad_same.h"
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
extern "C" __global__ void __launch_bounds__(49) tvmgen_default_fused_nn_pad_kernel0(float* __restrict__ T_pad, float* __restrict__ placeholder) {
  T_pad[(((int)threadIdx.x))] = (((((7 <= ((int)threadIdx.x)) && (((int)threadIdx.x) < 42)) && (1 <= (((int)threadIdx.x) % 7))) && ((((int)threadIdx.x) % 7) < 6)) ? placeholder[(((((((int)threadIdx.x) / 7) * 5) + (((int)threadIdx.x) % 7)) - 6))] : 0.000000e+00f);
}

extern "C" __global__ void __launch_bounds__(3) tvmgen_default_fused_nn_conv2d_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ compute) {
  float compute_local[1];
  __shared__ float pad_temp_shared[7];
  __shared__ float placeholder_shared[3];
  compute_local[(0)] = 0.000000e+00f;
  pad_temp_shared[((((int)threadIdx.x) * 3))] = placeholder[(((((int)blockIdx.y) * 14) + (((int)threadIdx.x) * 3)))];
  if (((int)threadIdx.x) < 2) {
    pad_temp_shared[(((((int)threadIdx.x) * 3) + 1))] = placeholder[((((((int)blockIdx.y) * 14) + (((int)threadIdx.x) * 3)) + 1))];
  }
  if (((int)threadIdx.x) < 2) {
    pad_temp_shared[(((((int)threadIdx.x) * 3) + 2))] = placeholder[((((((int)blockIdx.y) * 14) + (((int)threadIdx.x) * 3)) + 2))];
  }
  placeholder_shared[(((int)threadIdx.x))] = placeholder1[(((int)threadIdx.x))];
  __syncthreads();
  compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(0)]));
  compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1))] * placeholder_shared[(1)]));
  compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 2))] * placeholder_shared[(2)]));
  __syncthreads();
  pad_temp_shared[((((int)threadIdx.x) * 3))] = placeholder[((((((int)blockIdx.y) * 14) + (((int)threadIdx.x) * 3)) + 7))];
  if (((int)threadIdx.x) < 2) {
    pad_temp_shared[(((((int)threadIdx.x) * 3) + 1))] = placeholder[((((((int)blockIdx.y) * 14) + (((int)threadIdx.x) * 3)) + 8))];
  }
  if (((int)threadIdx.x) < 2) {
    pad_temp_shared[(((((int)threadIdx.x) * 3) + 2))] = placeholder[((((((int)blockIdx.y) * 14) + (((int)threadIdx.x) * 3)) + 9))];
  }
  placeholder_shared[(((int)threadIdx.x))] = placeholder1[((((int)threadIdx.x) + 3))];
  __syncthreads();
  compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(0)]));
  compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1))] * placeholder_shared[(1)]));
  compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 2))] * placeholder_shared[(2)]));
  __syncthreads();
  pad_temp_shared[((((int)threadIdx.x) * 3))] = placeholder[((((((int)blockIdx.y) * 14) + (((int)threadIdx.x) * 3)) + 14))];
  if (((int)threadIdx.x) < 2) {
    pad_temp_shared[(((((int)threadIdx.x) * 3) + 1))] = placeholder[((((((int)blockIdx.y) * 14) + (((int)threadIdx.x) * 3)) + 15))];
  }
  if (((int)threadIdx.x) < 2) {
    pad_temp_shared[(((((int)threadIdx.x) * 3) + 2))] = placeholder[((((((int)blockIdx.y) * 14) + (((int)threadIdx.x) * 3)) + 16))];
  }
  placeholder_shared[(((int)threadIdx.x))] = placeholder1[((((int)threadIdx.x) + 6))];
  __syncthreads();
  compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(0)]));
  compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1))] * placeholder_shared[(1)]));
  compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 2))] * placeholder_shared[(2)]));
  compute[(((((int)blockIdx.y) * 3) + ((int)threadIdx.x)))] = compute_local[(0)];
}



PluginFieldCollection tpat_test_conv_with_autopad_sameCreator::mFC{};
std::vector<PluginField> tpat_test_conv_with_autopad_sameCreator::mPluginAttributes;

int tpat_test_conv_with_autopad_same::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
    
    dim3 dimBlock, dimGrid;
    
    dimGrid = dim3(1,1,1);
    dimBlock = dim3(49,1,1);
    tvmgen_default_fused_nn_pad_kernel0<<<dimGrid, dimBlock, 0, stream>>>((float*)workspace, (float*)inputs[0]);
    
    dimGrid = dim3(1,3,1);
    dimBlock = dim3(3,1,1);
    tvmgen_default_fused_nn_conv2d_kernel0<<<dimGrid, dimBlock, 0, stream>>>((float*)workspace, (float*)inputs[1], (float*)outputs[0]);
    
}

REGISTER_TENSORRT_PLUGIN(tpat_test_conv_with_autopad_sameCreator);