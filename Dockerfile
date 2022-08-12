FROM nvcr.io/nvidia/tensorflow:20.06-tf1-py3
#FROM nvcr.io/nvidia/tensorflow:21.08-tf1-py3
RUN wget -O "llvm-9.0.1.src.tar.xz" https://github.com/llvm/llvm-project/releases/download/llvmorg-9.0.1/llvm-9.0.1.src.tar.xz \
    && tar -xvf llvm-9.0.1.src.tar.xz && mkdir llvm-9.0.1.src/build \
    && cd llvm-9.0.1.src/build && cmake -G "Unix Makefiles" -DLLVM_TARGETS_TO_BUILD=X86 -DCMAKE_BUILD_TYPE="Release" -DCMAKE_INSTALL_PREFIX="/usr/local/llvm" .. && make -j8 && make install PREFIX="/usr/local/llvm"
RUN pip install pycuda onnx==1.10.0 nvidia-pyindex && pip install onnx-graphsurgeon onnxruntime==1.9.0 tf2onnx==1.11.1 xgboost onnxruntime-gpu==1.10.0
RUN git clone --recursive https://github.com/Tencent/TPAT.git /workspace/TPAT && cd /workspace/TPAT/3rdparty/blazerml-tvm && mkdir build && cp cmake/config.cmake build && cd build 
RUN sed -i 's/set(USE_LLVM OFF)/set(USE_LLVM \/usr\/local\/llvm\/bin\/llvm-config)/g' /workspace/TPAT/3rdparty/blazerml-tvm/build/config.cmake 
RUN sed -i 's/set(USE_CUDA OFF)/set(USE_CUDA ON)/g' /workspace/TPAT/3rdparty/blazerml-tvm/build/config.cmake
RUN cd /workspace/TPAT/3rdparty/blazerml-tvm/build/ && cmake .. && make -j8 
ENV TVM_HOME="/workspace/TPAT/3rdparty/blazerml-tvm/"
ENV PYTHONPATH="$TVM_HOME/python:${PYTHONPATH}"
