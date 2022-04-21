#include "tpat_ReduceMean_20.h"
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



#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
#define __shfl_sync(mask, var, lane, width) \
        __shfl((var), (lane), (width))

#define __shfl_down_sync(mask, var, offset, width) \
        __shfl_down((var), (offset), (width))

#define __shfl_up_sync(mask, var, offset, width) \
        __shfl_up((var), (offset), (width))
#endif


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
extern "C" __global__ void __launch_bounds__(128) tvmgen_default_fused_mean_kernel1(float* __restrict__ T_divide, float* __restrict__ placeholder_red) {
  T_divide[(((int)threadIdx.x))] = (placeholder_red[(((int)threadIdx.x))] * 2.604167e-03f);
}

extern "C" __global__ void __launch_bounds__(1024) tvmgen_default_fused_mean_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder_red) {
  float placeholder_red_rf[1];
  float red_buf0[1];
  placeholder_red_rf[(0)] = 0.000000e+00f;
  for (int k2_outer = 0; k2_outer < 12; ++k2_outer) {
    placeholder_red_rf[(0)] = (placeholder_red_rf[(0)] + placeholder[(((((((int)blockIdx.x) * 12288) + (((int)threadIdx.y) * 384)) + (k2_outer * 32)) + ((int)threadIdx.x)))]);
  }
  uint mask[1];
  float t0[1];
  red_buf0[(0)] = placeholder_red_rf[(0)];
  mask[(0)] = __activemask();
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 16, 32);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 8, 32);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 4, 32);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 2, 32);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 1, 32);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  red_buf0[(0)] = __shfl_sync(mask[(0)], red_buf0[(0)], 0, 32);
  if (((int)threadIdx.x) == 0) {
    placeholder_red[(((((int)blockIdx.x) * 32) + ((int)threadIdx.y)))] = red_buf0[(0)];
  }
}



PluginFieldCollection tpat_ReduceMean_20Creator::mFC{};
std::vector<PluginField> tpat_ReduceMean_20Creator::mPluginAttributes;

int tpat_ReduceMean_20::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
    
    dim3 dimBlock, dimGrid;
    
    dimGrid = dim3(4,1,1);
    dimBlock = dim3(32,32,1);
    tvmgen_default_fused_mean_kernel0<<<dimGrid, dimBlock, 0, stream>>>((float*)inputs[0], (float*)(workspace + 0));
    
    dimGrid = dim3(1,1,1);
    dimBlock = dim3(128,1,1);
    tvmgen_default_fused_mean_kernel1<<<dimGrid, dimBlock, 0, stream>>>((float*)outputs[0], (float*)(workspace + 0));
    
}

REGISTER_TENSORRT_PLUGIN(tpat_ReduceMean_20Creator);