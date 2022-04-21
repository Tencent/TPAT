#include "tpat_test_gatternd.h"
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

extern "C" __global__ void __launch_bounds__(4) tvmgen_default_fused_transpose_gather_nd_kernel0_bs1(float* __restrict__ T_gather_nd, float* __restrict__ placeholder, int64_t* __restrict__ placeholder1) {
  T_gather_nd[(((int)threadIdx.x))] = placeholder[((((placeholder1[(0)] * (int64_t)8) + (placeholder1[(1)] * (int64_t)4)) + ((int64_t)((int)threadIdx.x))))];
}




extern "C" __global__ void __launch_bounds__(508) tvmgen_default_fused_transpose_gather_nd_kernel0_bs128(float* __restrict__ T_gather_nd, float* __restrict__ placeholder, int64_t* __restrict__ placeholder1) {
  printf("threadIdx.x : %d\n", threadIdx.x );
  T_gather_nd[(((int)threadIdx.x))] = placeholder[((((placeholder1[(((((int)threadIdx.x) >> 2) * 2))] * (int64_t)8) + (placeholder1[((((((int)threadIdx.x) >> 2) * 2) + 1))] * (int64_t)4)) + (((int64_t)((int)threadIdx.x)) & (int64_t)3)))];
}




extern "C" __global__ void __launch_bounds__(1024) tvmgen_default_fused_transpose_gather_nd_kernel0_bs256(float* __restrict__ T_gather_nd, float* __restrict__ placeholder, int64_t* __restrict__ placeholder1) {
  T_gather_nd[(((int)threadIdx.x))] = placeholder[((((placeholder1[(((((int)threadIdx.x) >> 2) * 2))] * (int64_t)8) + (placeholder1[((((((int)threadIdx.x) >> 2) * 2) + 1))] * (int64_t)4)) + (((int64_t)((int)threadIdx.x)) & (int64_t)3)))];
}



PluginFieldCollection tpat_test_gatterndCreator::mFC{};
std::vector<PluginField> tpat_test_gatterndCreator::mPluginAttributes;

int tpat_test_gatternd::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
    if( inputDesc[0].dims.d[0] == 1){
      
      dim3 dimBlock, dimGrid;
      
      dimGrid = dim3(1,1,1);
      dimBlock = dim3(4,1,1);
      tvmgen_default_fused_transpose_gather_nd_kernel0_bs1<<<dimGrid, dimBlock, 0, stream>>>((float*)outputs[0], (float*)inputs[0], (int*)inputs[1]);
      
    }
    else if( 1  < inputDesc[0].dims.d[0] && inputDesc[0].dims.d[0] <= 128){
      
      dim3 dimBlock, dimGrid;
      printf("input0 len : %d, input1 len : %d, output len:%d\n", sizeof(inputs[0]) / 4, sizeof(inputs[1]) / 4, sizeof(outputs[0]) / 4);   
      dimGrid = dim3(1,1,1);
      dimBlock = dim3(508,1,1);
      tvmgen_default_fused_transpose_gather_nd_kernel0_bs128<<<dimGrid, dimBlock, 0, stream>>>((float*)outputs[0], (float*)inputs[0], (int*)inputs[1]);
      
    }else if( 128  < inputDesc[0].dims.d[0] && inputDesc[0].dims.d[0] <= 256){
      
      dim3 dimBlock, dimGrid;
      
      dimGrid = dim3(1,1,1);
      dimBlock = dim3(1024,1,1);
      tvmgen_default_fused_transpose_gather_nd_kernel0_bs256<<<dimGrid, dimBlock, 0, stream>>>((float*)outputs[0], (float*)inputs[0], (int*)inputs[1]);
      
    }
}

REGISTER_TENSORRT_PLUGIN(tpat_test_gatterndCreator);
