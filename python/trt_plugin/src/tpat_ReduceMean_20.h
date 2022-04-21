/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "NvInfer.h"
#include <iostream>
#include <cstring>
#include <vector>
#include <assert.h>

namespace nvinfer1
{
namespace plugin
{

class tpat_ReduceMean_20: public IPluginV2DynamicExt {
public:
    tpat_ReduceMean_20() {}
    
    tpat_ReduceMean_20(const void *buffer, size_t length) {
    }

    virtual size_t getSerializationSize() const noexcept override {
        return 0;
    }
    virtual void serialize(void *buffer) const noexcept override {}
    
    //! The combination of kLINEAR + kFLOAT is supported.
    bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override
    {
        bool condition = true;
        if (pos == 0){
            //std::cout << (inOut[pos].format == nvinfer1::TensorFormat::kLINEAR) << ", " << (inOut[pos].type == nvinfer1::DataType::kFLOAT) << std::endl;
            condition &= inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
            condition &= inOut[pos].type == nvinfer1::DataType::kFLOAT;
        }
        if (pos == 1){
            //std::cout << (inOut[pos].format == nvinfer1::TensorFormat::kLINEAR) << ", " << (inOut[pos].type == nvinfer1::DataType::kFLOAT) << std::endl;
            condition &= inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
            condition &= inOut[pos].type == nvinfer1::DataType::kFLOAT;
        }
        
        return condition;
    }

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override {
        return new tpat_ReduceMean_20();
    }
    int getNbOutputs() const noexcept override {
        //std::cout << __FUNCTION__ << std::endl;
        return 1;
    }
    nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept override {
        //std::cout << __FUNCTION__ << std::endl;
        if (outputIndex == 0){
            nvinfer1::DimsExprs output_shape;
            output_shape.nbDims = 3;
            output_shape.d[0] = exprBuilder.constant(1);
            output_shape.d[1] = exprBuilder.constant(128);
            output_shape.d[2] = exprBuilder.constant(1);
            
            return output_shape;
        }
        
    }
    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override{
        //std::cout << __FUNCTION__ << std::endl;
        if (index == 0){
            return nvinfer1::DataType::kFLOAT;
        }
        
    }
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs, const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override{
        return 512;
    }
    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs, const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override {}
    int initialize() noexcept override {return 0;}
    void terminate() noexcept override {}
    void destroy() noexcept override { delete this; }
    void setPluginNamespace(const char* szNamespace) noexcept override {mNamespace = szNamespace;}
    const char* getPluginNamespace() const noexcept override {return mNamespace.c_str();}
    const char* getPluginType() const noexcept override {return "tpat_ReduceMean_20";}
    const char* getPluginVersion() const noexcept override {return "1";}
    void attachToContext(cudnnContext * /*cudnn*/, cublasContext * /*cublas*/, nvinfer1::IGpuAllocator * /*allocator*/) noexcept {}
    void detachFromContext() noexcept {}

private:

    const char* mPluginNamespace;
    std::string mNamespace;
};

class tpat_ReduceMean_20Creator: public nvinfer1::IPluginCreator {
public:
    tpat_ReduceMean_20Creator(){
	    mFC.nbFields = mPluginAttributes.size();
	    mFC.fields = mPluginAttributes.data();
    }
    nvinfer1::IPluginV2DynamicExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override {
        tpat_ReduceMean_20* obj = new tpat_ReduceMean_20{serialData, serialLength};
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    
    const char* getPluginName() const noexcept override {return "tpat_ReduceMean_20";}
    const char* getPluginVersion() const noexcept override {return "1";}

    void setPluginNamespace(const char* szNamespace) noexcept override {mNamespace = szNamespace;}
    const char* getPluginNamespace() const noexcept override {return mNamespace.c_str();}
    
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override {
        //std::cout << __FUNCTION__ << std::endl;
        return &mFC;
    }
    nvinfer1::IPluginV2DynamicExt* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override {
        //std::cout << __FUNCTION__ << std::endl;
        tpat_ReduceMean_20* obj = new tpat_ReduceMean_20{};
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
private:
    std::string mNamespace;
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
};

} // namespace plugin

} // namespace nvinfer1