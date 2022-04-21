#include "tpat_tpat_split.h"
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

extern "C" __global__ void __launch_bounds__(64) tvmgen_default_fused_split_kernel0_bs1(float* __restrict__ T_split, float* __restrict__ placeholder) {
  T_split[(((int)threadIdx.x))] = placeholder[(((int)threadIdx.x))];
}


extern "C" __global__ void __launch_bounds__(128) tvmgen_default_fused_split_kernel1_bs1(float* __restrict__ T_split, float* __restrict__ placeholder) {
  T_split[(((int)threadIdx.x))] = placeholder[((((int)threadIdx.x) + 64))];
}




extern "C" __global__ void __launch_bounds__(1024) tvmgen_default_fused_split_kernel1_bs128(float* __restrict__ T_split, float* __restrict__ placeholder) {
  T_split[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))] = placeholder[(((((((int)blockIdx.x) * 1536) + ((((int)threadIdx.x) >> 7) * 192)) + (((int)threadIdx.x) & 127)) + 64))];
}


extern "C" __global__ void __launch_bounds__(1024) tvmgen_default_fused_split_kernel0_bs128(float* __restrict__ T_split, float* __restrict__ placeholder) {
  printf("idx : %d\n", (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))));
  T_split[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))] = placeholder[((((((int)blockIdx.x) * 3072) + ((((int)threadIdx.x) >> 6) * 192)) + (((int)threadIdx.x) & 63)))];
}




extern "C" __global__ void __launch_bounds__(1024) tvmgen_default_fused_split_kernel1_bs256(float* __restrict__ T_split, float* __restrict__ placeholder) {
  T_split[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))] = placeholder[(((((((int)blockIdx.x) * 1536) + ((((int)threadIdx.x) >> 7) * 192)) + (((int)threadIdx.x) & 127)) + 64))];
}


extern "C" __global__ void __launch_bounds__(1024) tvmgen_default_fused_split_kernel0_bs256(float* __restrict__ T_split, float* __restrict__ placeholder) {
  T_split[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))] = placeholder[((((((int)blockIdx.x) * 3072) + ((((int)threadIdx.x) >> 6) * 192)) + (((int)threadIdx.x) & 63)))];
}



PluginFieldCollection tpat_tpat_splitCreator::mFC{};
std::vector<PluginField> tpat_tpat_splitCreator::mPluginAttributes;

int tpat_tpat_split::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
    if( inputDesc[0].dims.d[0] == 1){
      
      dim3 dimBlock, dimGrid;
      
      dimGrid = dim3(1,1,1);
      dimBlock = dim3(64,1,1);
      tvmgen_default_fused_split_kernel0_bs1<<<dimGrid, dimBlock, 0, stream>>>((float*)outputs[0], (float*)inputs[0]);
      
      dimGrid = dim3(1,1,1);
      dimBlock = dim3(128,1,1);
      tvmgen_default_fused_split_kernel1_bs1<<<dimGrid, dimBlock, 0, stream>>>((float*)outputs[1], (float*)inputs[0]);
      
    }
    else if( 1  < inputDesc[0].dims.d[0] && inputDesc[0].dims.d[0] <= 128){
      
      dim3 dimBlock, dimGrid;
      
      dimGrid = dim3(8,1,1);
      dimBlock = dim3(1024,1,1);
      tvmgen_default_fused_split_kernel0_bs128<<<dimGrid, dimBlock, 0, stream>>>((float*)outputs[0], (float*)inputs[0]);
      
      dimGrid = dim3(16,1,1);
      dimBlock = dim3(1024,1,1);
      tvmgen_default_fused_split_kernel1_bs128<<<dimGrid, dimBlock, 0, stream>>>((float*)outputs[1], (float*)inputs[0]);
      
    }else if( 128  < inputDesc[0].dims.d[0] && inputDesc[0].dims.d[0] <= 256){
      
      dim3 dimBlock, dimGrid;
      
      dimGrid = dim3(16,1,1);
      dimBlock = dim3(1024,1,1);
      tvmgen_default_fused_split_kernel0_bs256<<<dimGrid, dimBlock, 0, stream>>>((float*)outputs[0], (float*)inputs[0]);
      
      dimGrid = dim3(32,1,1);
      dimBlock = dim3(1024,1,1);
      tvmgen_default_fused_split_kernel1_bs256<<<dimGrid, dimBlock, 0, stream>>>((float*)outputs[1], (float*)inputs[0]);
      
    }
}

REGISTER_TENSORRT_PLUGIN(tpat_tpat_splitCreator);
