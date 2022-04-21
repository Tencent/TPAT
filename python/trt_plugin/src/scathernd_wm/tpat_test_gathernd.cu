#include "tpat_test_gathernd.h"
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

extern "C" __global__ void __launch_bounds__(600) tvmgen_default_fused_transpose_gather_nd_kernel0_bs1(float* __restrict__ T_gather_nd, float* __restrict__ placeholder, int64_t* __restrict__ placeholder1) {
  T_gather_nd[(((int)threadIdx.x))] = placeholder[((((placeholder1[(0)] * (int64_t)94800) + (placeholder1[(1)] * (int64_t)600)) + ((int64_t)((int)threadIdx.x))))];
}




extern "C" __global__ void __launch_bounds__(1024) tvmgen_default_fused_transpose_gather_nd_kernel0_bs128(float* __restrict__ T_gather_nd, float* __restrict__ placeholder, int64_t* __restrict__ placeholder1) {
  T_gather_nd[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))] = placeholder[((((placeholder1[(((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) / 600) * 2))] * (int64_t)94800) + (placeholder1[((((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) / 600) * 2) + 1))] * (int64_t)600)) + (((((int64_t)((int)blockIdx.x)) * (int64_t)1024) + ((int64_t)((int)threadIdx.x))) % (int64_t)600)))];
}




extern "C" __global__ void __launch_bounds__(1024) tvmgen_default_fused_transpose_gather_nd_kernel0_bs256(float* __restrict__ T_gather_nd, float* __restrict__ placeholder, int64_t* __restrict__ placeholder1) {
  T_gather_nd[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))] = placeholder[((((placeholder1[(((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) / 600) * 2))] * (int64_t)94800) + (placeholder1[((((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) / 600) * 2) + 1))] * (int64_t)600)) + (((((int64_t)((int)blockIdx.x)) * (int64_t)1024) + ((int64_t)((int)threadIdx.x))) % (int64_t)600)))];
}



PluginFieldCollection tpat_test_gatherndCreator::mFC{};
std::vector<PluginField> tpat_test_gatherndCreator::mPluginAttributes;

int tpat_test_gathernd::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
    if( inputDesc[0].dims.d[0] == 1){
      
      dim3 dimBlock, dimGrid;
      
      dimGrid = dim3(1,1,1);
      dimBlock = dim3(600,1,1);
      tvmgen_default_fused_transpose_gather_nd_kernel0_bs1<<<dimGrid, dimBlock, 0, stream>>>((float*)outputs[0], (float*)inputs[0], (int*)inputs[1]);
      
    }
    else if( 1  < inputDesc[0].dims.d[0] && inputDesc[0].dims.d[0] <= 128){
      
      dim3 dimBlock, dimGrid;
      
      dimGrid = dim3(75,1,1);
      dimBlock = dim3(1024,1,1);
      tvmgen_default_fused_transpose_gather_nd_kernel0_bs128<<<dimGrid, dimBlock, 0, stream>>>((float*)outputs[0], (float*)inputs[0], (int*)inputs[1]);
      
    }else if( 128  < inputDesc[0].dims.d[0] && inputDesc[0].dims.d[0] <= 256){
      int bs = inputDesc[0].dims.d[0];
      int offset_input_0 = 0;
      int offset_input_1 = offset_input_0 + 256 * 158 * 600 * sizeof(float);
      int offset_output_0 = offset_input_1 + 256 * 2 * sizeof(float);
      int workspace_size = offset_output_0 + 256 * 600 * sizeof(float);
      std::cout << "batch size = " << bs << std::endl;
      checkCudaErrors(cudaMemset(workspace, 0, workspace_size));
      checkCudaErrors(cudaMemcpyAsync((workspace + offset_input_0), (void *)inputs[0], bs * 158 * 600 * sizeof(float), cudaMemcpyDeviceToDevice, stream));
      checkCudaErrors(cudaMemcpyAsync((workspace + offset_input_1), (void *)inputs[1], bs * 2 * sizeof(int), cudaMemcpyDeviceToDevice, stream));
      // std::cout << "input memcpy d2d finished" << std::endl;
      dim3 dimBlock, dimGrid;
      
      dimGrid = dim3(150,1,1);
      dimBlock = dim3(1024,1,1);
      tvmgen_default_fused_transpose_gather_nd_kernel0_bs256<<<dimGrid, dimBlock, 0, stream>>>((float *)(workspace + offset_output_0), (float *)(workspace + offset_input_0), (int *)(workspace + offset_input_1));
      checkCudaErrors(cudaMemcpyAsync((void *)outputs[0], (void *)(workspace + offset_output_0), bs * 600 * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    }
}

REGISTER_TENSORRT_PLUGIN(tpat_test_gatherndCreator);