#include "tpat_test_scatternd.h"
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

extern "C" __global__ void __launch_bounds__(256) tvmgen_default_fused_scatter_nd_kernel1_bs1(float* __restrict__ scatter_nd_cuda, float* __restrict__ placeholder, int64_t* __restrict__ placeholder1) {
  for (int i = 0; i < 1; ++i) {
    scatter_nd_cuda[((((placeholder1[(0)] * (int64_t)32768) + (placeholder1[(1)] * (int64_t)256)) + ((int64_t)((int)threadIdx.x))))] = placeholder[(((int)threadIdx.x))];
  }
}


extern "C" __global__ void __launch_bounds__(256) tvmgen_default_fused_scatter_nd_kernel0_bs1(float* __restrict__ scatter_nd_cuda, float* __restrict__ placeholder) {
  scatter_nd_cuda[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)))] = placeholder[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)))];
}


extern "C" __global__ void __launch_bounds__(2) tvmgen_default_fused_transpose_kernel0_bs1(int64_t* __restrict__ T_transpose, int64_t* __restrict__ placeholder) {
  T_transpose[(((int)threadIdx.x))] = placeholder[(((int)threadIdx.x))];
}




extern "C" __global__ void __launch_bounds__(256) tvmgen_default_fused_transpose_kernel0_bs128(int64_t* __restrict__ T_transpose, int64_t* __restrict__ placeholder) {
  T_transpose[(((int)threadIdx.x))] = placeholder[((((((int)threadIdx.x) & 127) * 2) + (((int)threadIdx.x) >> 7)))];
}


extern "C" __global__ void __launch_bounds__(256) tvmgen_default_fused_scatter_nd_kernel0_bs128(float* __restrict__ scatter_nd_cuda, float* __restrict__ placeholder) {
  scatter_nd_cuda[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)))] = placeholder[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)))];
}


extern "C" __global__ void __launch_bounds__(256) tvmgen_default_fused_scatter_nd_kernel1_bs128(float* __restrict__ scatter_nd_cuda, float* __restrict__ placeholder, int64_t* __restrict__ placeholder1) {
  for (int i = 0; i < 128; ++i) {
    scatter_nd_cuda[((((placeholder1[(i)] * (int64_t)32768) + (placeholder1[((i + 128))] * (int64_t)256)) + ((int64_t)((int)threadIdx.x))))] = placeholder[(((i * 256) + ((int)threadIdx.x)))];
  }
}




extern "C" __global__ void __launch_bounds__(512) tvmgen_default_fused_transpose_kernel0_bs256(int64_t* __restrict__ T_transpose, int64_t* __restrict__ placeholder) {
  T_transpose[(((int)threadIdx.x))] = placeholder[((((((int)threadIdx.x) & 255) * 2) + (((int)threadIdx.x) >> 8)))];
}


extern "C" __global__ void __launch_bounds__(256) tvmgen_default_fused_scatter_nd_kernel0_bs256(float* __restrict__ scatter_nd_cuda, float* __restrict__ placeholder) {
  scatter_nd_cuda[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)))] = placeholder[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)))];
}


extern "C" __global__ void __launch_bounds__(256) tvmgen_default_fused_scatter_nd_kernel1_bs256(float* __restrict__ scatter_nd_cuda, float* __restrict__ placeholder, int64_t* __restrict__ placeholder1) {
  for (int i = 0; i < 256; ++i) {
    scatter_nd_cuda[((((placeholder1[(i)] * (int64_t)32768) + (placeholder1[((i + 256))] * (int64_t)256)) + ((int64_t)((int)threadIdx.x))))] = placeholder[(((i * 256) + ((int)threadIdx.x)))];
  }
}



PluginFieldCollection tpat_test_scatterndCreator::mFC{};
std::vector<PluginField> tpat_test_scatterndCreator::mPluginAttributes;

int tpat_test_scatternd::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
    if( inputDesc[0].dims.d[0] == 1){
      
      dim3 dimBlock, dimGrid;
      
      dimGrid = dim3(1,1,1);
      dimBlock = dim3(2,1,1);
      tvmgen_default_fused_transpose_kernel0_bs1<<<dimGrid, dimBlock, 0, stream>>>((int*)workspace, (int*)inputs[1]);
      
      dimGrid = dim3(128,1,1);
      dimBlock = dim3(256,1,1);
      tvmgen_default_fused_scatter_nd_kernel0_bs1<<<dimGrid, dimBlock, 0, stream>>>((float*)outputs[0], (float*)inputs[0]);
      
      dimGrid = dim3(1,1,1);
      dimBlock = dim3(256,1,1);
      tvmgen_default_fused_scatter_nd_kernel1_bs1<<<dimGrid, dimBlock, 0, stream>>>((float*)outputs[0], (float*)inputs[2], (int*)workspace);
      
    }
    else if( 1  < inputDesc[0].dims.d[0] && inputDesc[0].dims.d[0] <= 128){
      
      dim3 dimBlock, dimGrid;
      
      dimGrid = dim3(1,1,1);
      dimBlock = dim3(256,1,1);
      tvmgen_default_fused_transpose_kernel0_bs128<<<dimGrid, dimBlock, 0, stream>>>((int*)workspace, (int*)inputs[1]);
      
      dimGrid = dim3(16384,1,1);
      dimBlock = dim3(256,1,1);
      tvmgen_default_fused_scatter_nd_kernel0_bs128<<<dimGrid, dimBlock, 0, stream>>>((float*)outputs[0], (float*)inputs[0]);
      
      dimGrid = dim3(1,1,1);
      dimBlock = dim3(256,1,1);
      tvmgen_default_fused_scatter_nd_kernel1_bs128<<<dimGrid, dimBlock, 0, stream>>>((float*)outputs[0], (float*)inputs[2], (int*)workspace);
      
    }else if( 128  < inputDesc[0].dims.d[0] && inputDesc[0].dims.d[0] <= 256){
      
      int bs = inputDesc[0].dims.d[0];
      int offset_input_0 = 4096;
      int offset_input_1 = offset_input_0 + 256 * 128 * 256 * sizeof(float);
      int offset_input_2 = offset_input_1 + 256 * 2 * sizeof(int);
      int offset_output_0 = offset_input_2 + 256 * 256 * sizeof(float);
      int workspace_size = offset_output_0 + 256 * 128 * 256 * sizeof(float);
      checkCudaErrors(cudaMemset(workspace, 0, workspace_size));
      checkCudaErrors(cudaMemcpyAsync((workspace + offset_input_0), (void *)inputs[0], bs * 128 * 256 * sizeof(float), cudaMemcpyDeviceToDevice, stream));
      checkCudaErrors(cudaMemcpyAsync((workspace + offset_input_1), (void *)inputs[1], bs * 2 * sizeof(int), cudaMemcpyDeviceToDevice, stream));
      checkCudaErrors(cudaMemcpyAsync((workspace + offset_input_2), (void *)inputs[2], bs * 256 * sizeof(float), cudaMemcpyDeviceToDevice, stream));
      // std::cout << "input memcpy d2d finished" << std::endl;
      dim3 dimBlock, dimGrid;

      dimGrid = dim3(1,1,1);
      dimBlock = dim3(512,1,1);
      // tvmgen_default_fused_transpose_kernel0_bs256<<<dimGrid, dimBlock, 0, stream>>>((int*)workspace, (int*)inputs[1]);
      tvmgen_default_fused_transpose_kernel0_bs256<<<dimGrid, dimBlock, 0, stream>>>((int *)workspace, (int *)(workspace + offset_input_1));

      dimGrid = dim3(32768,1,1);
      dimBlock = dim3(256,1,1);
      // tvmgen_default_fused_scatter_nd_kernel0_bs256<<<dimGrid, dimBlock, 0, stream>>>((float*)outputs[0], (float*)inputs[0]);
      tvmgen_default_fused_scatter_nd_kernel0_bs256<<<dimGrid, dimBlock, 0, stream>>>((float *)(workspace + offset_output_0), (float *)(workspace + offset_input_0));

      dimGrid = dim3(1,1,1);
      dimBlock = dim3(256,1,1);
      // tvmgen_default_fused_scatter_nd_kernel1_bs256<<<dimGrid, dimBlock, 0, stream>>>((float*)outputs[0], (float*)inputs[2], (int*)workspace);
      tvmgen_default_fused_scatter_nd_kernel1_bs256<<<dimGrid, dimBlock, 0, stream>>>((float *)(workspace + offset_output_0), (float *)(workspace + offset_input_2), (int *)workspace);
      checkCudaErrors(cudaMemcpyAsync((void *)outputs[0], (void *)(workspace + offset_output_0), bs * 128 * 256 * sizeof(float), cudaMemcpyDeviceToDevice, stream));
      // std::cout << "output memcpy d2d finished" << std::endl;
    }
}

REGISTER_TENSORRT_PLUGIN(tpat_test_scatterndCreator);