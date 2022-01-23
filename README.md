# TPAT - TensorRT Plugin Autogen Tool
## 背景介绍
1. 自动生成高性能TensorRT插件，快速支持TensorRT不支持的算子，或替换现有TensorRT性能不佳的算子/形状。
2. 端到端命令行工具，用户不需要有任何cuda编程知识，只需要输入ONNX模型并指定需要生成插件的算子名称或类型。
3. 暂时不支持dynamic shape的op以及int8、float16、double类型的输入输出。
4. 在真实场景下TPAT自动生成plugin性能：
    - [TPAT自动生成plugin对比手写Plugin](/docs/Compare_handwritten.md)
    - [TPAT优化TensorRT原生实现operator](/docs/Optimize_TensorRT.md)


## op支持
- [TPAT-1.0支持的op](/docs/Operators.md)


## 环境配置
1. 获取TPAT项目

    使用 `git clone --recursive` 下载TPAT项目或是
    `git clone` 配合 `git submodule init && git submodule update`来获取BlazerML-TVM源码

2. BlazerML-TVM
   
    BlazerML-TVM 的编译需要`LLVM >= 9.0.1`与`GCC >= 7.3.0`;推荐`LLVM==9.0.1` && `GCC==7.4.0`
   
    ```
    mkdir build && cp -r cmake/config.cmake build
    #修改config.cmake
    set(USE_LLVM /usr/local/llvm/bin/llvm-config)
    set(USE_CUDA ON)
    # 需要支持C++14的gcc编译器
    cd build && cmake .. 
    make -j
    # TVM加入环境变量
    export TVM_HOME=/path/to/tvm
	export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
    ```

3. TensorRT

    [安装TensorRT](https://github.com/NVIDIA/TensorRT)

4. TPAT

    Python3 Env
    ```
    pip3 install numpy, tensorflow-gpu==1.15, pycuda, tensorrt, tf2onnx,torch, pytest, onnx, onnxruntime, onnx_graphsurgeon, xgboost, jinja2, ctypes, tornado, cloudpickle, psutil
    ```
    Plugin Compiler Env
    ```
    python/trt_plugin/Makefile : CUDA_PATH 修改为本机的CUDA路径, TRT_LIB_PATH修改为本机TRT路径

    ```


## 用法
### 输入
- input_model_path [-i][*required*] : 输入的onnx model, 包含需要生成plugin的op, 支持多个同类型op 
- output_model_path [-o][*required*] : 输出onnx model, 对应的op type修改为生成的plugin名称, 配合plugin通过onnx-parser可以直接生成trt-engine.
- node_names [-n] : 指定需要生成plugin的op 名字, 支持多个
- node_types [-t] : 指定需要生成plugin的op 类型, 支持多个
- plugin_name_dict [-p] : 字典, key为 op名字, value为生成对应op 的plugin的名字, 支持多个

### 使用
1. 命令行调用
    ```
   python3 onnx_to_plugin.py -i input.onnx -o output.onnx -n op_name1 op_name2
   python3 onnx_to_plugin.py -i input.onnx -o output.onnx -t op_type1 op_type2
   python3 onnx_to_plugin.py -i input.onnx -o output.onnx -p '{"op_name1": "plugin_name1", "op_name2": "plugin_name2"}'
    ```
2. 函数调用
    ```
   onnx2plugin(
       input_model_path, 
       output_model_path, 
       node_names=None, 
       node_types=None, 
       plugin_name_dict=None
       )
    ```
    ***node_names, node_types, plugin_name_dict 至少指定一项***

### 输出
1. 通过 plugin_name_dict 指定plugin name
- python/trt_plugin/src 下生成有 {plugin_name}.cu 和 {plugin_name}.h 的plugin源码
- python/trt_plugin/src 下生成有 {plugin_name}.so 的动态链接库

2. 只输入了node names和node types
- python/trt_plugin/src 下生成有 tpat_{node_name}.cu 和 tpat_{node_name}.h的plugin源码
- python/trt_plugin/src 下生成有 tpat_{node_name}.so 的动态链接库

### Example && UnitTest

- Example : [example_tensorflow.py](/examples/example_tensorflow.py)
- UnitTest : [test_tapt.py](/tests/test_tpat.py)
