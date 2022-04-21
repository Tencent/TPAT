#include "tpat_test_lstm.h"
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

extern "C" __global__ void __launch_bounds__(5) tvmgen_default_fused_split_squeeze_concatenate_kernel0_bs1(float* __restrict__ T_concat, float* __restrict__ placeholder, float* __restrict__ placeholder1) {
  T_concat[(((int)threadIdx.x))] = ((1 <= ((int)threadIdx.x)) ? placeholder[((((int)threadIdx.x) - 1))] : placeholder1[(((int)threadIdx.x))]);
}


extern "C" __global__ void __launch_bounds__(4) tvmgen_default_fused_split_squeeze_kernel0_bs1(float* __restrict__ T_squeeze, float* __restrict__ placeholder) {
  T_squeeze[(((int)threadIdx.x))] = placeholder[(((int)threadIdx.x))];
}


extern "C" __global__ void __launch_bounds__(64) tvmgen_default_fused_nn_dense_add_add_kernel0_bs1(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_add, float* __restrict__ placeholder2, float* __restrict__ placeholder3) {
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


extern "C" __global__ void __launch_bounds__(4) tvmgen_default_fused_split_kernel2_bs1(float* __restrict__ T_split_sections, float* __restrict__ placeholder) {
  T_split_sections[(((int)threadIdx.x))] = placeholder[((((int)threadIdx.x) + 8))];
}


extern "C" __global__ void __launch_bounds__(4) tvmgen_default_fused_sigmoid_multiply_sigmoid_tanh_multiply_add_kernel0_bs1(float* __restrict__ T_add, float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ placeholder2, float* __restrict__ placeholder3) {
  T_add[(((int)threadIdx.x))] = (((1.000000e+00f / (1.000000e+00f + __expf((0.000000e+00f - placeholder[(((int)threadIdx.x))])))) * placeholder1[(((int)threadIdx.x))]) + ((1.000000e+00f / (1.000000e+00f + __expf((0.000000e+00f - placeholder2[(((int)threadIdx.x))])))) * tanhf(placeholder3[(((int)threadIdx.x))])));
}


extern "C" __global__ void __launch_bounds__(4) tvmgen_default_fused_split_kernel3_bs1(float* __restrict__ T_split_sections, float* __restrict__ placeholder) {
  T_split_sections[(((int)threadIdx.x))] = placeholder[((((int)threadIdx.x) + 12))];
}


extern "C" __global__ void __launch_bounds__(4) tvmgen_default_fused_sigmoid_tanh_multiply_kernel0_bs1(float* __restrict__ T_multiply, float* __restrict__ placeholder, float* __restrict__ placeholder1) {
  T_multiply[(((int)threadIdx.x))] = ((1.000000e+00f / (1.000000e+00f + __expf((0.000000e+00f - placeholder[(((int)threadIdx.x))])))) * tanhf(placeholder1[(((int)threadIdx.x))]));
}


extern "C" __global__ void __launch_bounds__(4) tvmgen_default_fused_split_kernel1_bs1(float* __restrict__ T_split_sections, float* __restrict__ placeholder) {
  T_split_sections[(((int)threadIdx.x))] = placeholder[((((int)threadIdx.x) + 4))];
}


extern "C" __global__ void __launch_bounds__(4) tvmgen_default_fused_split_kernel0_bs1(float* __restrict__ T_split_sections, float* __restrict__ placeholder) {
  T_split_sections[(((int)threadIdx.x))] = placeholder[(((int)threadIdx.x))];
}


extern "C" __global__ void __launch_bounds__(4) tvmgen_default_fused_stack_expand_dims_kernel0_bs1(float* __restrict__ T_expand_dims, float* __restrict__ placeholder) {
  T_expand_dims[(((int)threadIdx.x))] = placeholder[(((int)threadIdx.x))];
}




extern "C" __global__ void __launch_bounds__(64) tvmgen_default_fused_nn_dense_add_add_kernel0_bs2(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_add, float* __restrict__ placeholder2, float* __restrict__ placeholder3) {
  float T_matmul_NT_rf[1];
  __shared__ float red_buf0[64];
  __shared__ float T_matmul_NT[1];
  T_matmul_NT_rf[(0)] = 0.000000e+00f;
  if (((int)threadIdx.x) < 5) {
    T_matmul_NT_rf[(0)] = (T_matmul_NT_rf[(0)] + (placeholder[(((((int)blockIdx.y) * 5) + ((int)threadIdx.x)))] * placeholder1[(((((int)blockIdx.x) * 5) + ((int)threadIdx.x)))]));
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
    T_add[(((((int)blockIdx.y) * 16) + ((int)blockIdx.x)))] = ((T_matmul_NT[(0)] + placeholder2[(((int)blockIdx.x))]) + placeholder3[(((int)blockIdx.x))]);
  }
}


extern "C" __global__ void __launch_bounds__(8) tvmgen_default_fused_split_kernel0_bs2(float* __restrict__ T_split_sections, float* __restrict__ placeholder) {
  T_split_sections[(((int)threadIdx.x))] = placeholder[((((((int)threadIdx.x) >> 2) * 16) + (((int)threadIdx.x) & 3)))];
}


extern "C" __global__ void __launch_bounds__(8) tvmgen_default_fused_split_squeeze_kernel0_bs2(float* __restrict__ T_squeeze, float* __restrict__ placeholder) {
  T_squeeze[(((int)threadIdx.x))] = placeholder[(((int)threadIdx.x))];
}


extern "C" __global__ void __launch_bounds__(8) tvmgen_default_fused_split_kernel1_bs2(float* __restrict__ T_split_sections, float* __restrict__ placeholder) {
  T_split_sections[(((int)threadIdx.x))] = placeholder[(((((((int)threadIdx.x) >> 2) * 16) + (((int)threadIdx.x) & 3)) + 4))];
}


extern "C" __global__ void __launch_bounds__(8) tvmgen_default_fused_sigmoid_tanh_multiply_kernel0_bs2(float* __restrict__ T_multiply, float* __restrict__ placeholder, float* __restrict__ placeholder1) {
  T_multiply[(((int)threadIdx.x))] = ((1.000000e+00f / (1.000000e+00f + __expf((0.000000e+00f - placeholder[(((int)threadIdx.x))])))) * tanhf(placeholder1[(((int)threadIdx.x))]));
}


extern "C" __global__ void __launch_bounds__(10) tvmgen_default_fused_split_squeeze_concatenate_kernel0_bs2(float* __restrict__ T_concat, float* __restrict__ placeholder, float* __restrict__ placeholder1) {
  T_concat[(((int)threadIdx.x))] = ((1 <= (((int)threadIdx.x) % 5)) ? placeholder[(((((((int)threadIdx.x) / 5) * 4) + (((int)threadIdx.x) % 5)) - 1))] : placeholder1[(((((int)threadIdx.x) / 5) + (((int)threadIdx.x) % 5)))]);
}


extern "C" __global__ void __launch_bounds__(8) tvmgen_default_fused_split_kernel2_bs2(float* __restrict__ T_split_sections, float* __restrict__ placeholder) {
  T_split_sections[(((int)threadIdx.x))] = placeholder[(((((((int)threadIdx.x) >> 2) * 16) + (((int)threadIdx.x) & 3)) + 8))];
}


extern "C" __global__ void __launch_bounds__(8) tvmgen_default_fused_sigmoid_multiply_sigmoid_tanh_multiply_add_kernel0_bs2(float* __restrict__ T_add, float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ placeholder2, float* __restrict__ placeholder3) {
  T_add[(((int)threadIdx.x))] = (((1.000000e+00f / (1.000000e+00f + __expf((0.000000e+00f - placeholder[(((int)threadIdx.x))])))) * placeholder1[(((int)threadIdx.x))]) + ((1.000000e+00f / (1.000000e+00f + __expf((0.000000e+00f - placeholder2[(((int)threadIdx.x))])))) * tanhf(placeholder3[(((int)threadIdx.x))])));
}


extern "C" __global__ void __launch_bounds__(8) tvmgen_default_fused_split_kernel3_bs2(float* __restrict__ T_split_sections, float* __restrict__ placeholder) {
  T_split_sections[(((int)threadIdx.x))] = placeholder[(((((((int)threadIdx.x) >> 2) * 16) + (((int)threadIdx.x) & 3)) + 12))];
}


extern "C" __global__ void __launch_bounds__(8) tvmgen_default_fused_stack_expand_dims_kernel0_bs2(float* __restrict__ T_expand_dims, float* __restrict__ placeholder) {
  T_expand_dims[(((int)threadIdx.x))] = placeholder[(((int)threadIdx.x))];
}




extern "C" __global__ void __launch_bounds__(16) tvmgen_default_fused_sigmoid_tanh_multiply_kernel0_bs4(float* __restrict__ T_multiply, float* __restrict__ placeholder, float* __restrict__ placeholder1) {
  T_multiply[(((int)threadIdx.x))] = ((1.000000e+00f / (1.000000e+00f + __expf((0.000000e+00f - placeholder[(((int)threadIdx.x))])))) * tanhf(placeholder1[(((int)threadIdx.x))]));
}


extern "C" __global__ void __launch_bounds__(16) tvmgen_default_fused_split_kernel3_bs4(float* __restrict__ T_split_sections, float* __restrict__ placeholder) {
  T_split_sections[(((int)threadIdx.x))] = placeholder[(((((((int)threadIdx.x) >> 2) * 16) + (((int)threadIdx.x) & 3)) + 12))];
}


extern "C" __global__ void __launch_bounds__(16) tvmgen_default_fused_split_kernel0_bs4(float* __restrict__ T_split_sections, float* __restrict__ placeholder) {
  T_split_sections[(((int)threadIdx.x))] = placeholder[((((((int)threadIdx.x) >> 2) * 16) + (((int)threadIdx.x) & 3)))];
}


extern "C" __global__ void __launch_bounds__(20) tvmgen_default_fused_split_squeeze_concatenate_kernel0_bs4(float* __restrict__ T_concat, float* __restrict__ placeholder, float* __restrict__ placeholder1) {
  T_concat[(((int)threadIdx.x))] = ((1 <= (((int)threadIdx.x) % 5)) ? placeholder[(((((((int)threadIdx.x) / 5) * 4) + (((int)threadIdx.x) % 5)) - 1))] : placeholder1[(((((int)threadIdx.x) / 5) + (((int)threadIdx.x) % 5)))]);
}


extern "C" __global__ void __launch_bounds__(16) tvmgen_default_fused_split_kernel1_bs4(float* __restrict__ T_split_sections, float* __restrict__ placeholder) {
  T_split_sections[(((int)threadIdx.x))] = placeholder[(((((((int)threadIdx.x) >> 2) * 16) + (((int)threadIdx.x) & 3)) + 4))];
}


extern "C" __global__ void __launch_bounds__(64) tvmgen_default_fused_nn_dense_add_add_kernel0_bs4(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_add, float* __restrict__ placeholder2, float* __restrict__ placeholder3) {
  float T_matmul_NT_rf[1];
  __shared__ float red_buf0[64];
  __shared__ float T_matmul_NT[1];
  T_matmul_NT_rf[(0)] = 0.000000e+00f;
  if (((int)threadIdx.x) < 5) {
    T_matmul_NT_rf[(0)] = (T_matmul_NT_rf[(0)] + (placeholder[(((((int)blockIdx.y) * 5) + ((int)threadIdx.x)))] * placeholder1[(((((int)blockIdx.x) * 5) + ((int)threadIdx.x)))]));
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
    T_add[(((((int)blockIdx.y) * 16) + ((int)blockIdx.x)))] = ((T_matmul_NT[(0)] + placeholder2[(((int)blockIdx.x))]) + placeholder3[(((int)blockIdx.x))]);
  }
}


extern "C" __global__ void __launch_bounds__(16) tvmgen_default_fused_split_squeeze_kernel0_bs4(float* __restrict__ T_squeeze, float* __restrict__ placeholder) {
  T_squeeze[(((int)threadIdx.x))] = placeholder[(((int)threadIdx.x))];
}


extern "C" __global__ void __launch_bounds__(16) tvmgen_default_fused_stack_expand_dims_kernel0_bs4(float* __restrict__ T_expand_dims, float* __restrict__ placeholder) {
  T_expand_dims[(((int)threadIdx.x))] = placeholder[(((int)threadIdx.x))];
}


extern "C" __global__ void __launch_bounds__(16) tvmgen_default_fused_sigmoid_multiply_sigmoid_tanh_multiply_add_kernel0_bs4(float* __restrict__ T_add, float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ placeholder2, float* __restrict__ placeholder3) {
  T_add[(((int)threadIdx.x))] = (((1.000000e+00f / (1.000000e+00f + __expf((0.000000e+00f - placeholder[(((int)threadIdx.x))])))) * placeholder1[(((int)threadIdx.x))]) + ((1.000000e+00f / (1.000000e+00f + __expf((0.000000e+00f - placeholder2[(((int)threadIdx.x))])))) * tanhf(placeholder3[(((int)threadIdx.x))])));
}


extern "C" __global__ void __launch_bounds__(16) tvmgen_default_fused_split_kernel2_bs4(float* __restrict__ T_split_sections, float* __restrict__ placeholder) {
  T_split_sections[(((int)threadIdx.x))] = placeholder[(((((((int)threadIdx.x) >> 2) * 16) + (((int)threadIdx.x) & 3)) + 8))];
}



PluginFieldCollection tpat_test_lstmCreator::mFC{};
std::vector<PluginField> tpat_test_lstmCreator::mPluginAttributes;

int tpat_test_lstm::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
    if( inputDesc[0].dims.d[1] == 1){
      const float constant_4[80] = { 0.12387955 ,-0.20074937 ,-0.24207678 ,-0.4079863 ,-0.47693703 ,-0.322245 ,0.1401614 ,-0.024244845 ,-0.0009294152 ,0.48796695 ,-0.46809548 ,-0.15450531 ,0.40640873 ,-0.16986427 ,0.12398505 ,0.119716704 ,-0.44268203 ,0.09522551 ,-0.055396676 ,0.37981492 ,0.024076879 ,-0.44315916 ,-0.47987413 ,0.05779779 ,0.023085535 ,0.04221672 ,0.08570331 ,-0.41609216 ,-0.42373613 ,0.4453668 ,-0.52089316 ,0.23437148 ,0.38061476 ,0.22216696 ,0.026045084 ,-0.44769794 ,0.53210074 ,0.46300644 ,-0.5237252 ,-0.31928265 ,-0.4694685 ,-0.096541226 ,-0.48961195 ,0.4640221 ,0.2845959 ,-0.13909537 ,-0.506791 ,-0.17840046 ,-0.008621931 ,0.17275614 ,-0.13541988 ,-0.45349696 ,0.28284144 ,0.2012549 ,0.4583398 ,-0.4332764 ,-0.40414375 ,0.22432274 ,-0.12811655 ,0.53304535 ,-0.1891337 ,0.49263114 ,-0.28280523 ,-0.47508252 ,0.4665435 ,0.18526947 ,-0.22607157 ,-0.12552798 ,-0.21662751 ,0.31361616 ,0.4173324 ,-0.50423914 ,0.2623557 ,0.017793417 ,-0.29670906 ,0.48881108 ,-0.261464 ,-0.34464884 ,0.23304445 ,-0.035270065  };
      checkCudaErrors(cudaMemcpyAsync((workspace + 20), &constant_4, 80 * sizeof(float), cudaMemcpyHostToDevice, stream));const float constant_5[16] = { 0.0 ,0.0 ,0.0 ,0.0 ,1.0 ,1.0 ,1.0 ,1.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0  };
      checkCudaErrors(cudaMemcpyAsync((workspace + 340), &constant_5, 16 * sizeof(float), cudaMemcpyHostToDevice, stream));const float constant_6[16] = { 0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0  };
      checkCudaErrors(cudaMemcpyAsync((workspace + 404), &constant_6, 16 * sizeof(float), cudaMemcpyHostToDevice, stream));
      int bs = inputDesc[0].dims.d[1];
      int input_offset_0_begin = 656;
      int input_offset_0_end = input_offset_0_begin + 1 * 1 * 1 * sizeof(float);
      int output_offset_0_begin = input_offset_0_end;
      int output_offset_0_end = output_offset_0_begin + 1 * 1 * 1 * 4 * sizeof(float);
      int output_offset_1_begin = output_offset_0_end;
      int output_offset_1_end = output_offset_1_begin + 1 * 4 * 1 * sizeof(float);
      int output_offset_2_begin = output_offset_1_end;
      int output_offset_2_end = output_offset_2_begin + 1 * 4 * 1 * sizeof(float);

      checkCudaErrors(cudaMemset(workspace + input_offset_0_begin, 0, output_offset_2_end - input_offset_0_begin));
      checkCudaErrors(cudaMemcpyAsync((void *)(workspace + input_offset_0_begin),(void *) inputs[0], bs * 1 * 1 * sizeof(float), cudaMemcpyDeviceToDevice, stream));
      dim3 dimBlock, dimGrid;
      
      dimGrid = dim3(1,1,1);
      dimBlock = dim3(4,1,1);
      tvmgen_default_fused_split_squeeze_kernel0_bs1<<<dimGrid, dimBlock, 0, stream>>>((float*)(workspace + output_offset_1_begin), (float*)inputs[6]);
      
      dimGrid = dim3(1,1,1);
      dimBlock = dim3(5,1,1);
      tvmgen_default_fused_split_squeeze_concatenate_kernel0_bs1<<<dimGrid, dimBlock, 0, stream>>>((float*)(workspace + 0), (float*)(workspace + output_offset_1_begin), (float*)(workspace + input_offset_0_begin));
      
      dimGrid = dim3(16,1,1);
      dimBlock = dim3(64,1,1);
      tvmgen_default_fused_nn_dense_add_add_kernel0_bs1<<<dimGrid, dimBlock, 0, stream>>>((float*)(workspace + 0), (float*)(workspace + 20), (float*)(workspace + output_offset_2_begin), (float*)(workspace + 340), (float*)(workspace + 404));
      
      dimGrid = dim3(1,1,1);
      dimBlock = dim3(4,1,1);
      tvmgen_default_fused_split_kernel0_bs1<<<dimGrid, dimBlock, 0, stream>>>((float*)(workspace + 0), (float*)(workspace + output_offset_2_begin));
      
      dimGrid = dim3(1,1,1);
      dimBlock = dim3(4,1,1);
      tvmgen_default_fused_split_kernel1_bs1<<<dimGrid, dimBlock, 0, stream>>>((float*)(workspace + output_offset_0_begin), (float*)(workspace + output_offset_2_begin));
      
      dimGrid = dim3(1,1,1);
      dimBlock = dim3(4,1,1);
      tvmgen_default_fused_split_kernel2_bs1<<<dimGrid, dimBlock, 0, stream>>>((float*)(workspace + 468), (float*)(workspace + output_offset_2_begin));
      
      dimGrid = dim3(1,1,1);
      dimBlock = dim3(4,1,1);
      tvmgen_default_fused_split_kernel3_bs1<<<dimGrid, dimBlock, 0, stream>>>((float*)(workspace + 484), (float*)(workspace + output_offset_2_begin));
      
      dimGrid = dim3(1,1,1);
      dimBlock = dim3(4,1,1);
      tvmgen_default_fused_sigmoid_multiply_sigmoid_tanh_multiply_add_kernel0_bs1<<<dimGrid, dimBlock, 0, stream>>>((float*)(workspace + output_offset_2_begin), (float*)(workspace + output_offset_0_begin), (float*)(workspace + output_offset_1_begin), (float*)(workspace + 0), (float*)(workspace + 468));
      
      dimGrid = dim3(1,1,1);
      dimBlock = dim3(4,1,1);
      tvmgen_default_fused_sigmoid_tanh_multiply_kernel0_bs1<<<dimGrid, dimBlock, 0, stream>>>((float*)(workspace + output_offset_1_begin), (float*)(workspace + 484), (float*)(workspace + output_offset_2_begin));
      
      dimGrid = dim3(1,1,1);
      dimBlock = dim3(4,1,1);
      tvmgen_default_fused_stack_expand_dims_kernel0_bs1<<<dimGrid, dimBlock, 0, stream>>>((float*)(workspace + output_offset_0_begin), (float*)(workspace + output_offset_1_begin));
      checkCudaErrors(cudaMemcpyAsync((void*) outputs[0], (void* )(workspace + output_offset_0_begin), bs * 1 * 1 * 4 * sizeof(float), cudaMemcpyDeviceToDevice, stream));
      checkCudaErrors(cudaMemcpyAsync((void*) outputs[1], (void* )(workspace + output_offset_1_begin), bs * 1 * 4 * sizeof(float), cudaMemcpyDeviceToDevice, stream ));
      checkCudaErrors(cudaMemcpyAsync((void*) outputs[2], (void* )(workspace + output_offset_2_begin), bs * 1 * 4 * sizeof(float), cudaMemcpyDeviceToDevice, stream ));      

    }
    else if( 1  < inputDesc[0].dims.d[1] && inputDesc[0].dims.d[1] <= 2){
      const float constant_4[80] = { 0.12387955 ,-0.20074937 ,-0.24207678 ,-0.4079863 ,-0.47693703 ,-0.322245 ,0.1401614 ,-0.024244845 ,-0.0009294152 ,0.48796695 ,-0.46809548 ,-0.15450531 ,0.40640873 ,-0.16986427 ,0.12398505 ,0.119716704 ,-0.44268203 ,0.09522551 ,-0.055396676 ,0.37981492 ,0.024076879 ,-0.44315916 ,-0.47987413 ,0.05779779 ,0.023085535 ,0.04221672 ,0.08570331 ,-0.41609216 ,-0.42373613 ,0.4453668 ,-0.52089316 ,0.23437148 ,0.38061476 ,0.22216696 ,0.026045084 ,-0.44769794 ,0.53210074 ,0.46300644 ,-0.5237252 ,-0.31928265 ,-0.4694685 ,-0.096541226 ,-0.48961195 ,0.4640221 ,0.2845959 ,-0.13909537 ,-0.506791 ,-0.17840046 ,-0.008621931 ,0.17275614 ,-0.13541988 ,-0.45349696 ,0.28284144 ,0.2012549 ,0.4583398 ,-0.4332764 ,-0.40414375 ,0.22432274 ,-0.12811655 ,0.53304535 ,-0.1891337 ,0.49263114 ,-0.28280523 ,-0.47508252 ,0.4665435 ,0.18526947 ,-0.22607157 ,-0.12552798 ,-0.21662751 ,0.31361616 ,0.4173324 ,-0.50423914 ,0.2623557 ,0.017793417 ,-0.29670906 ,0.48881108 ,-0.261464 ,-0.34464884 ,0.23304445 ,-0.035270065  };
      checkCudaErrors(cudaMemcpyAsync((workspace + 40), &constant_4, 80 * sizeof(float), cudaMemcpyHostToDevice, stream));const float constant_5[16] = { 0.0 ,0.0 ,0.0 ,0.0 ,1.0 ,1.0 ,1.0 ,1.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0  };
      checkCudaErrors(cudaMemcpyAsync((workspace + 360), &constant_5, 16 * sizeof(float), cudaMemcpyHostToDevice, stream));const float constant_6[16] = { 0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0  };
      checkCudaErrors(cudaMemcpyAsync((workspace + 424), &constant_6, 16 * sizeof(float), cudaMemcpyHostToDevice, stream));
      int bs = inputDesc[0].dims.d[1];
      int input_offset_0_begin = 656;
      int input_offset_0_end = input_offset_0_begin + 2 * 1 * 1 * sizeof(float);
      int output_offset_0_begin = input_offset_0_end;
      int output_offset_0_end = output_offset_0_begin + 1 * 1 * 2 * 4 * sizeof(float);
      int output_offset_1_begin = output_offset_0_end;
      int output_offset_1_end = output_offset_1_begin + 1 * 2 * 4 * sizeof(float);
      int output_offset_2_begin = output_offset_1_end;
      int output_offset_2_end = output_offset_2_begin + 1 * 2 * 4 * sizeof(float);

      checkCudaErrors(cudaMemset(workspace + input_offset_0_begin, 0, output_offset_2_end - input_offset_0_begin));
      checkCudaErrors(cudaMemcpyAsync((void *)(workspace + input_offset_0_begin),(void *) inputs[0], bs * 1 * 1 * sizeof(float), cudaMemcpyDeviceToDevice, stream));
      dim3 dimBlock, dimGrid;
      
      dimGrid = dim3(1,1,1);
      dimBlock = dim3(8,1,1);
      tvmgen_default_fused_split_squeeze_kernel0_bs2<<<dimGrid, dimBlock, 0, stream>>>((float*)(workspace + output_offset_1_begin), (float*)inputs[6]);
      
      dimGrid = dim3(1,1,1);
      dimBlock = dim3(10,1,1);
      tvmgen_default_fused_split_squeeze_concatenate_kernel0_bs2<<<dimGrid, dimBlock, 0, stream>>>((float*)(workspace + 0), (float*)(workspace + output_offset_1_begin), (float*)(workspace + input_offset_0_begin));
      
      dimGrid = dim3(16,2,1);
      dimBlock = dim3(64,1,1);
      tvmgen_default_fused_nn_dense_add_add_kernel0_bs2<<<dimGrid, dimBlock, 0, stream>>>((float*)(workspace + 0), (float*)(workspace + 40), (float*)(workspace + output_offset_2_begin), (float*)(workspace + 360), (float*)(workspace + 424));
      
      dimGrid = dim3(1,1,1);
      dimBlock = dim3(8,1,1);
      tvmgen_default_fused_split_kernel0_bs2<<<dimGrid, dimBlock, 0, stream>>>((float*)(workspace + 0), (float*)(workspace + output_offset_2_begin));
      
      dimGrid = dim3(1,1,1);
      dimBlock = dim3(8,1,1);
      tvmgen_default_fused_split_kernel1_bs2<<<dimGrid, dimBlock, 0, stream>>>((float*)(workspace + output_offset_0_begin), (float*)(workspace + output_offset_2_begin));
      
      dimGrid = dim3(1,1,1);
      dimBlock = dim3(8,1,1);
      tvmgen_default_fused_split_kernel2_bs2<<<dimGrid, dimBlock, 0, stream>>>((float*)(workspace + 488), (float*)(workspace + output_offset_2_begin));
      
      dimGrid = dim3(1,1,1);
      dimBlock = dim3(8,1,1);
      tvmgen_default_fused_split_kernel3_bs2<<<dimGrid, dimBlock, 0, stream>>>((float*)(workspace + 520), (float*)(workspace + output_offset_2_begin));
      
      dimGrid = dim3(1,1,1);
      dimBlock = dim3(8,1,1);
      tvmgen_default_fused_sigmoid_multiply_sigmoid_tanh_multiply_add_kernel0_bs2<<<dimGrid, dimBlock, 0, stream>>>((float*)(workspace + output_offset_2_begin), (float*)(workspace + output_offset_0_begin), (float*)(workspace + output_offset_1_begin), (float*)(workspace + 0), (float*)(workspace + 488));
      
      dimGrid = dim3(1,1,1);
      dimBlock = dim3(8,1,1);
      tvmgen_default_fused_sigmoid_tanh_multiply_kernel0_bs2<<<dimGrid, dimBlock, 0, stream>>>((float*)(workspace + output_offset_1_begin), (float*)(workspace + 520), (float*)(workspace + output_offset_2_begin));
      
      dimGrid = dim3(1,1,1);
      dimBlock = dim3(8,1,1);
      tvmgen_default_fused_stack_expand_dims_kernel0_bs2<<<dimGrid, dimBlock, 0, stream>>>((float*)(workspace + output_offset_0_begin), (float*)(workspace + output_offset_1_begin));
      checkCudaErrors(cudaMemcpyAsync((void*) outputs[0], (void* )(workspace + output_offset_0_begin), bs * 1 * 1 * 4 * sizeof(float), cudaMemcpyDeviceToDevice, stream));
      checkCudaErrors(cudaMemcpyAsync((void*) outputs[1], (void* )(workspace + output_offset_1_begin), bs * 1 * 4 * sizeof(float), cudaMemcpyDeviceToDevice, stream ));
      checkCudaErrors(cudaMemcpyAsync((void*) outputs[2], (void* )(workspace + output_offset_2_begin), bs * 1 * 4 * sizeof(float), cudaMemcpyDeviceToDevice, stream ));     
 
   eckCudaErrors(cudaMemcpyAsync((void*) outputs[0], (void* )(workspace + output_offset_0_begin), bs * 2 * 1 * 4 * sizeof(float), cudaMemcpyDeviceToDevice, stream));
      checkCudaErrors(cudaMemcpyAsync((void*) outputs[1], (void* )(workspace + output_offset_1_begin), bs * 1 * 4 * sizeof(float), cudaMemcpyDeviceToDevice, stream ));
      checkCudaErrors(cudaMemcpyAsync((void*) outputs[2], (void* )(workspace + output_offset_2_begin), bs * 1 * 4 * sizeof(float), cudaMemcpyDeviceToDevice, stream ));}else if( 2  < inputDesc[0].dims.d[1] && inputDesc[0].dims.d[1] <= 4){
      const float constant_4[80] = { 0.12387955 ,-0.20074937 ,-0.24207678 ,-0.4079863 ,-0.47693703 ,-0.322245 ,0.1401614 ,-0.024244845 ,-0.0009294152 ,0.48796695 ,-0.46809548 ,-0.15450531 ,0.40640873 ,-0.16986427 ,0.12398505 ,0.119716704 ,-0.44268203 ,0.09522551 ,-0.055396676 ,0.37981492 ,0.024076879 ,-0.44315916 ,-0.47987413 ,0.05779779 ,0.023085535 ,0.04221672 ,0.08570331 ,-0.41609216 ,-0.42373613 ,0.4453668 ,-0.52089316 ,0.23437148 ,0.38061476 ,0.22216696 ,0.026045084 ,-0.44769794 ,0.53210074 ,0.46300644 ,-0.5237252 ,-0.31928265 ,-0.4694685 ,-0.096541226 ,-0.48961195 ,0.4640221 ,0.2845959 ,-0.13909537 ,-0.506791 ,-0.17840046 ,-0.008621931 ,0.17275614 ,-0.13541988 ,-0.45349696 ,0.28284144 ,0.2012549 ,0.4583398 ,-0.4332764 ,-0.40414375 ,0.22432274 ,-0.12811655 ,0.53304535 ,-0.1891337 ,0.49263114 ,-0.28280523 ,-0.47508252 ,0.4665435 ,0.18526947 ,-0.22607157 ,-0.12552798 ,-0.21662751 ,0.31361616 ,0.4173324 ,-0.50423914 ,0.2623557 ,0.017793417 ,-0.29670906 ,0.48881108 ,-0.261464 ,-0.34464884 ,0.23304445 ,-0.035270065  };
      checkCudaErrors(cudaMemcpyAsync((workspace + 80), &constant_4, 80 * sizeof(float), cudaMemcpyHostToDevice, stream));const float constant_5[16] = { 0.0 ,0.0 ,0.0 ,0.0 ,1.0 ,1.0 ,1.0 ,1.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0  };
      checkCudaErrors(cudaMemcpyAsync((workspace + 400), &constant_5, 16 * sizeof(float), cudaMemcpyHostToDevice, stream));const float constant_6[16] = { 0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0  };
      checkCudaErrors(cudaMemcpyAsync((workspace + 464), &constant_6, 16 * sizeof(float), cudaMemcpyHostToDevice, stream));

      int bs = inputDesc[0].dims.d[1];
      int input_offset_0_begin = 656;
      int input_offset_0_end = input_offset_0_begin + 4 * 1 * 1 * sizeof(float);
      int output_offset_0_begin = input_offset_0_end;
      int output_offset_0_end = output_offset_0_begin + 1 * 1 * 4 * 4 * sizeof(float);
      int output_offset_1_begin = output_offset_0_end;
      int output_offset_1_end = output_offset_1_begin + 1 * 4 * 4 * sizeof(float);
      int output_offset_2_begin = output_offset_1_end;
      int output_offset_2_end = output_offset_2_begin + 1 * 4 * 4 * sizeof(float);

      checkCudaErrors(cudaMemset(workspace + input_offset_0_begin, 0, output_offset_2_end - input_offset_0_begin));
      checkCudaErrors(cudaMemcpyAsync((void *)(workspace + input_offset_0_begin),(void *) inputs[0], bs * 1 * 1 * sizeof(float), cudaMemcpyDeviceToDevice, stream));

      dim3 dimBlock, dimGrid;
      dimGrid = dim3(1,1,1);
      dimBlock = dim3(16,1,1);
      tvmgen_default_fused_split_squeeze_kernel0_bs4<<<dimGrid, dimBlock, 0, stream>>>((float*)(workspace + output_offset_1_begin), (float*)inputs[6]);
      
      dimGrid = dim3(1,1,1);
      dimBlock = dim3(20,1,1);
      tvmgen_default_fused_split_squeeze_concatenate_kernel0_bs4<<<dimGrid, dimBlock, 0, stream>>>((float*)(workspace + 0), (float*)(workspace + output_offset_1_begin), (float*)(workspace + input_offset_0_begin));
      
      dimGrid = dim3(16,4,1);
      dimBlock = dim3(64,1,1);
      tvmgen_default_fused_nn_dense_add_add_kernel0_bs4<<<dimGrid, dimBlock, 0, stream>>>((float*)(workspace + 0), (float*)(workspace + 80), (float*)(workspace + output_offset_2_begin), (float*)(workspace + 400), (float*)(workspace + 464));
      
      dimGrid = dim3(1,1,1);
      dimBlock = dim3(16,1,1);
      tvmgen_default_fused_split_kernel0_bs4<<<dimGrid, dimBlock, 0, stream>>>((float*)(workspace + 0), (float*)(workspace + output_offset_2_begin));
      
      dimGrid = dim3(1,1,1);
      dimBlock = dim3(16,1,1);
      tvmgen_default_fused_split_kernel1_bs4<<<dimGrid, dimBlock, 0, stream>>>((float*)(workspace + output_offset_0_begin), (float*)(workspace + output_offset_2_begin));
      
      dimGrid = dim3(1,1,1);
      dimBlock = dim3(16,1,1);
      tvmgen_default_fused_split_kernel2_bs4<<<dimGrid, dimBlock, 0, stream>>>((float*)(workspace + 528), (float*)(workspace + output_offset_2_begin));
      
      dimGrid = dim3(1,1,1);
      dimBlock = dim3(16,1,1);
      tvmgen_default_fused_split_kernel3_bs4<<<dimGrid, dimBlock, 0, stream>>>((float*)(workspace + 592), (float*)(workspace + output_offset_2_begin));
      
      dimGrid = dim3(1,1,1);
      dimBlock = dim3(16,1,1);
      tvmgen_default_fused_sigmoid_multiply_sigmoid_tanh_multiply_add_kernel0_bs4<<<dimGrid, dimBlock, 0, stream>>>((float*)(workspace + output_offset_2_begin), (float*)(workspace + output_offset_0_begin), (float*)(workspace + output_offset_1_begin), (float*)(workspace + 0), (float*)(workspace + 528));
      
      dimGrid = dim3(1,1,1);
      dimBlock = dim3(16,1,1);
      tvmgen_default_fused_sigmoid_tanh_multiply_kernel0_bs4<<<dimGrid, dimBlock, 0, stream>>>((float*)(workspace + output_offset_1_begin), (float*)(workspace + 592), (float*)(workspace + output_offset_2_begin));
      
      dimGrid = dim3(1,1,1);
      dimBlock = dim3(16,1,1);
      tvmgen_default_fused_stack_expand_dims_kernel0_bs4<<<dimGrid, dimBlock, 0, stream>>>((float*)(workspace + output_offset_0_begin), (float*)(workspace + output_offset_1_begin));

      checkCudaErrors(cudaMemcpyAsync((void*) outputs[0], (void* )(workspace + output_offset_0_begin), bs * 1 * 1 * 4 * sizeof(float), cudaMemcpyDeviceToDevice, stream));
      checkCudaErrors(cudaMemcpyAsync((void*) outputs[1], (void* )(workspace + output_offset_1_begin), bs * 1 * 4 * sizeof(float), cudaMemcpyDeviceToDevice, stream ));
      checkCudaErrors(cudaMemcpyAsync((void*) outputs[2], (void* )(workspace + output_offset_2_begin), bs * 1 * 4 * sizeof(float), cudaMemcpyDeviceToDevice, stream )); 
    }
}

REGISTER_TENSORRT_PLUGIN(tpat_test_lstmCreator);
