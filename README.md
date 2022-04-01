# TPAT - TensorRT Plugin Autogen Tool
## Introduction
1. Automatically generate high-performance TensorRT plugins for unsupported operators or replacing inefficient kernels.
2. End-to-end command line tool. No requirement for any CUDA programming knowledge. Users only need to provide the ONNX model and assign the node names or types to auto-generate TensorRT plugin.
3. The performance of auto-generated TensorRT plugins in real cases:
    * [Performance comparation with hand-written kernels](/docs/Compare_handwritten.md)
    * [Optimization for TensorRT's original kernels](/docs/Optimize_TensorRT.md)

## Support Matrix
* [ONNX Operators supported by TPAT-1.0](/docs/Operators.md)

## Build
### 1. Prerequisites
#### System Packages
* LLVM >= 9.0.1, (LLVM==9.0.1 recommended)
* GCC >= 7.3.0, (GCC==7.4.0 recommended)
* [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)

#### PyPI packages
* numpy pycuda onnx onnxruntime onnx_graphsurgeon xgboost jinja2 ctypes tornado cloudpickle psutil
> NOTE: these necessary packages are recorded in requirements.txt

#### Optional packages
* tensorflow-gpu==1.15
* tf2onnx
* torch
* pytest
> NOTE: these optional packages are required by Example and UnitTest

### 2. Clone the TPAT repository
	
	git clone --recursive https://github.com/Tencent/TPAT.git TPAT
	
### 3. Build BlazerML-TVM
	
	cd TPAT/3rdparty/blazerml-tvm
    mkdir build && cp cmake/config.cmake build
    #Edit build/config.cmake to customize the compilation options
    set(USE_LLVM /usr/local/llvm/bin/llvm-config)
    set(USE_CUDA ON)
    #gcc compiler is required to support C++14
    cd build && cmake .. 
    make -j
    #TVM Python package
    export TVM_HOME=/path/to/tvm
	export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
    
### 4. Plugin Compiler Env
Modify python/trt_plugin/Makefile according to your environment setup.

    
    CUDA_PATH: local CUDA installation path
    TRT_LIB_PATH: local TensorRT installation path
    

## Usage 
TPAT provides a Python function and command line for usage.

### Python function 
	
	onnx2plugin(
	   input_model_path, 
	   output_model_path, 
	   node_names=None, 
	   node_types=None, 
	   plugin_name_dict=None
	   )
    
* input_model_path[*required*] : input onnx model including nodes which require TRT plugin
* output_model_path[*required*] : output onnx model where the corresponding node types are replaced by plugin names. The output onnx model can be directly converted to TRT with onnx parser and built plugin dynamic library.
* node_names : list of node names for autogen
* node_types : list of node types for autogen
* plugin_name_dict : dict of {plugin_name: node_name} for autogen
> NOTE: For node_names, node_types, plugin_name_dict, at least one of them should be provided

### Command line
	
	python3 Onnx2Plugin.py -i input.onnx -o output.onnx -n op_name1 op_name2
	python3 Onnx2Plugin.py -i input.onnx -o output.onnx -t op_type1 op_type2
	python3 Onnx2Plugin.py -i input.onnx -o output.onnx -p '{"op_name1": "plugin_name1", "op_name2": "plugin_name2"}'
    
* -i[*required*]: input_model_path
* -o[*required*]: output_model_path
* -n: node_names
* -t: node_types
* -p: plugin_name_dict

### Output
#### 1. Assign nodes and plugin names through plugin_name_dict
* trt_plugin/src contains {plugin_name}.cu and {plugin_name}.h
* trt_plugin/lib contains {plugin_name}.so

#### 2. Assign node names or node types
* trt_plugin/src contains tpat_{node_name}.cu and tpat_{node_name}.h
* trt_plugin/lib contains tpat_{node_name}.so

## Example && UnitTest
* Example : [example_tensorflow.py](/examples/example_tensorflow.py)
* UnitTest : [test_tapt.py](/tests/test_tpat.py)

## Release notes
### Changelog
* Support mutiple nodes for autogen
* Support boolean input/outputs
* Able to reuse plugins

### Known issues
* Dynamic shapes are not supported
* Opeartors with int8/float16/double inputs/outputs are not supported

### TODO
* Support ONNX subgraph for autogen
* Support direction conversion from TensorFlow and PyTorch