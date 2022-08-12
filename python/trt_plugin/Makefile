#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

CUDA_PATH   = /usr/local/cuda/
#TRT_LIB_PATH = /root/workspace/download/TensorRT-7.2.2.3/lib
TRT_LIB_PATH = /root/workspace/download/ft_local/TensorRT-8.0.0.3/lib

CUDA_INC_PATH = $(CUDA_PATH)/include
CUDA_COM_PATH = $(CUDA_PATH)/samples/common/inc

TRT_INC_PATH   = /usr/include/x86_64-linux-gnu
CUDNN_INC_PATH = /usr/include/x86_64-linux-gnu

GCC = g++
NVCC = $(CUDA_PATH)/bin/nvcc
# CCFLAGS = -g -std=c++11 -DNDEBUG
CCFLAGS = -w -std=c++11
# CCFLAGS+= -DDEBUG_ME
INCLUDES := -I. -I$(CUDA_COM_PATH) -I$(CUDA_INC_PATH) -I$(CUDNN_INC_PATH) -I$(TRT_INC_PATH) -I/usr/include

CUDA_LIB_PATH  = $(CUDA_PATH)/lib64
CUDNN_LIB_PATH = $(CUDA_PATH)/lib64


LDFLAGS := -L$(CUDA_LIB_PATH) -L$(CUDNN_LIB_PATH) -L$(TRT_LIB_PATH) 
LDFLAGS += -lnvinfer -lcudart -lcuda

LDFLAGS += -Wl,-rpath=$(CUDA_LIB_PATH)
LDFLAGS += -Wl,-rpath=$(CUDNN_LIB_PATH)
LDFLAGS += -Wl,-rpath=$(TRT_LIB_PATH)

SO = $(plugin_name).so
OBJ = $(shell find . -name '*.o')
DEP = $(OBJ:.o=.d)

SRCDIR := ./src
OBJDIR := ./obj
LIBDIR := ./lib

all: $(SO)

$(plugin_name).so : $(plugin_name).o

-include $(DEP)

clean:
	rm -rf $(LIBDIR)/$(SO) $(OBJDIR)/*

%.o: $(SRCDIR)/%.cpp
	$(AT)if [ ! -d $(OBJDIR) ]; then mkdir -p $(OBJDIR); fi
	$(GCC) $(CCFLAGS) -fPIC -MD -MP $(INCLUDES) -o $@ -c $<

%.o: $(SRCDIR)/%.cu
	$(AT)if [ ! -d $(OBJDIR) ]; then mkdir -p $(OBJDIR); fi
	$(NVCC) $(CCFLAGS) -M -MT $@ $(INCLUDES) -o $(@:.o=.d) $<
	$(NVCC) $(CCFLAGS) $(INCLUDES) -Xcompiler -fPIC -arch=sm_75 -o $@ -c $<
	# $(NVCC) $(CCFLAGS) $(INCLUDES) -Xcompiler -fPIC -arch=sm_75 -G -lineinfo -o $@ -c $<

$(SO):
	$(GCC) $(CCFLAGS) -shared -o $@ $+ $(LDFLAGS)
	$(AT)if [ ! -d  $(LIBDIR) ]; then mkdir -p $(LIBDIR); fi
	$(AT) mv *.o   $(OBJDIR)/
	$(AT) mv *.d   $(OBJDIR)/
	$(AT) mv *.so $(LIBDIR)/
