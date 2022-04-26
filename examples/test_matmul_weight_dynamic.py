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

import os
import sys
import ctypes
from datetime import datetime
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import tensorflow as tf
import argparse
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior
sys.path.append("..")
from python import *
os.chdir("../python")
IGPU = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(IGPU)
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

parser = argparse.ArgumentParser()
parser.add_argument('-batch_size', type=int, default=256, metavar='NUMBER')
parser.add_argument('-dynamic', type=str, default="true", metavar='STRING')

args = parser.parse_args()

dynamic = True

if args.dynamic.lower() == 'false':
    dynamic = False

batch_size = args.batch_size

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        print(binding, flush=True)
        # size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        size = trt.volume(engine.get_binding_shape(binding))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # if dynamic == True and 'input' in binding:
        if dynamic == True and size < 0:
            size = size * -1 * batch_size 
        print(size, dtype)
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(
        batch_size=batch_size, bindings=bindings, stream_handle=stream.handle
    )
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def main():
    tf.set_random_seed(1234)
    np.random.seed(0)
    iterations = 1
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        # test reduce
        #op_name = "rightshift"
        #batch_size = 256
        #dtype = "int32"
        #in_shape = (3, 11)
        #lh_data = np.random.randint(1, 8, size=(3, 11)).astype(dtype)
        #rh_data = np.random.randint(1, 3, size=(3, 11)).astype(dtype)
        #lft_data = tf.placeholder(dtype, in_shape, name="lft_data")
        #rgt_data = tf.placeholder(dtype, in_shape, name="rgt_data")
        #x = tf.bitwise.right_shift(lft_data, rgt_data, name=op_name)
        #input_node = [lft_data, rgt_data]
        #input_data = [lh_data, rh_data]
        #output = tf.identity(x, name="output")
        #sess.run(tf.global_variables_initializer())
        #input_dict = {e: input_data[i] for i, e in enumerate(input_node)}
        #input_with_num = ["lft_data:0", "rgt_data:0"]
        #output_with_num = ["output:0"]
        #time_sum = 0
        #for i in range(20):
        #    tf_result = sess.run([output], input_dict)
        #a = datetime.now()
        #for i in range(iterations):
        #    tf_result = sess.run([output], input_dict)
        #b = datetime.now()
        #time_sum = (b - a).total_seconds()
        #tf_time = (
        #    "[INFO] TF  execution time " + str(time_sum * 1000 / iterations) + " ms"
        #)

        if dynamic == True:
            input_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='input') 
            op_name = 'test_matmul'
        else:
            input_ph = tf.placeholder(dtype=tf.float32, shape=[batch_size, 256], name='input') 
            op_name = 'test_matmul_bs%d' % batch_size
        ## test matmul
        input_data = np.random.rand(batch_size, 1).astype(np.float32)     
        w1 = tf.get_variable('weight1', shape=[1, 8], initializer=tf.orthogonal_initializer())
        x = tf.matmul(input_ph, w1, name=op_name)
        w2 = tf.get_variable('weight2', shape=[8, 2], initializer=tf.orthogonal_initializer())
        x = tf.matmul(x, w2, name=op_name+"_2")
        w3 = tf.get_variable('weight3', shape=[2, 4], initializer=tf.orthogonal_initializer())
        x = tf.matmul(x, w3, name=op_name+"_3")
        ## test matmul
        print("\n\noperator shape: ", input_ph.shape, w1.shape, x.shape)
      

        output = tf.identity(x, name="output")
        sess.run(tf.global_variables_initializer())

        input_node = [input_ph]
        input_data = [input_data]
        input_with_num = ["input:0"]
        output_with_num = ["output:0"]


        input_dict = {e: input_data[i] for i, e in enumerate(input_node)}
        time_sum = 0
        #a = datetime.now()
        for i in range(iterations):
            tf_result = sess.run([output], input_dict)
        #b = datetime.now()
        #time_sum = (b - a).total_seconds()
        #tf_time = "[INFO] TF  execution time " + str(
        #    time_sum * 1000 / iterations) + " ms"

            
        output_name_without_port = ["output"]
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, output_name_without_port
        )
        # save frozen model
        with open("model/test_op_{}.pb".format(op_name), "wb") as ofile:
            ofile.write(frozen_graph.SerializeToString())

    input_model_file = "model/test_op_plugin.onnx"
    output_model_file = "model/test_op_trt.onnx"
    os.system(
        "python3 -m tf2onnx.convert --input model/test_op_{}.pb --inputs {} --outputs {} --output {} \
            --verbose --opset 11".format(
            op_name,
            str(",").join(input_with_num),
            str(",").join(output_with_num),
            input_model_file,
        )
    )
    os.remove("model/test_op_{}.pb".format(op_name))

    ### use ort to check results
    # import onnx
    # import onnxruntime as ort
    # from onnx import shape_inference
    # import onnx_graphsurgeon as gs
    # def run_ort(model_path):
    #     inferred_model = shape_inference.infer_shapes(onnx.load(model_path))
    #     graph = gs.import_onnx(inferred_model)
    #     EP_list = ['CUDAExecutionProvider']
    #     session = ort.InferenceSession(model_path, providers=EP_list)
    #     outname = [output.name for output in session.get_outputs()]
    #     print("onnx model output: ", outname)
    #     dummy_input = {}
    #     for gi in graph.inputs:
    #         gi.shape[0] = 2
    #         np.random.seed(1234)
    #         dummy_input[gi.name] = (np.random.random([int(i) for i in gi.shape])).astype(gi.dtype)
    #     for i in range(10):
    #         dummy_output = session.run(outname, dummy_input)
    #     print("ort result: ", dummy_output)
    #     return dummy_output
    # run_ort(input_model_file)
    # exit(0)

    node_names = [op_name + '_2']
    #node_types = ["MatMul"]
    #trt_plugin_names = onnx2plugin(
    #    input_model_file, output_model_file, node_names=node_names, dynamic_bs=dynamic, min_bs=1, max_bs=256, opt_bs=128
    #)
    trt_plugin_names = ['tpat_' + op_name + '_2']
    #trt_plugin_names = ['tpat_' + op_name, 'tpat_' + op_name + '_2', 'tpat_' + op_name + '_3']
    # trt_plugin_names = ['tpat_test_matmul_bs256']
    
    for trt_plugin_name in trt_plugin_names:
        assert os.path.isfile(f"./trt_plugin/lib/{trt_plugin_name}.so")
        ctypes.cdll.LoadLibrary("./trt_plugin/lib/{}.so".format(trt_plugin_name))

    # build trt model by onnx model
    cuda.Device(0).make_context()
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
        1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    ) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_batch_size = 1024
        builder_config = builder.create_builder_config()
        builder_config.max_workspace_size = 1 << 30
       
        if dynamic == True:
            profile = builder.create_optimization_profile()
            for input in input_node:
                shape_without_batch = input.shape.as_list()[1:]
                profile.set_shape(input.name, [1] + shape_without_batch, [256] + shape_without_batch, [256] + shape_without_batch )
            #profile.set_shape('input:0', [1, 64], [256, 64], [1024, 64])
            builder_config.add_optimization_profile(profile)


        trt_file_path = 'matmul.gie'
        if os.path.isfile(trt_file_path):
           os.remove(trt_file_path)
        if os.path.isfile(trt_file_path):
            with open(trt_file_path, 'rb') as f:
                engine_str = f.read()
            with trt.Runtime(TRT_LOGGER) as runtime:
                engine = runtime.deserialize_cuda_engine(engine_str)
        else:
            #with open(input_model_file_dynamic, "rb") as model:
            with open(output_model_file, "rb") as model:
                # parse onnx model
                parser.parse(model.read())
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
            engine = builder.build_engine(network, builder_config)
            if engine is None:
                print("[ERROR] engine is None")
                exit(-1)
            print('build engine done')
            with open(trt_file_path, 'wb') as f:
                f.write(engine.serialize())


        inputs, outputs, bindings, stream = allocate_buffers(engine)
        with engine.create_execution_context() as context:
            if dynamic == True:
                context.set_binding_shape(0, [batch_size] + input_node[0].shape.as_list()[1:])    
            
            for i in range(len(inputs)):
                data = input_data[i].ravel()
                np.copyto(inputs[i].host, data)
                print(f"{i} input is {data}")
            
            c = datetime.now()
            for i in range(iterations):
                output = do_inference(
                    context,
                    bindings=bindings,
                    inputs=inputs,
                    outputs=outputs,
                    stream=stream,
                    batch_size=batch_size,
                )
            d = datetime.now()
            time_sum = (d - c).total_seconds()
            trt_time = "TRT execution time " + str(time_sum * 1000 / iterations) + " ms"
            trt_result = output
    
    # np.allclose(tf_result[0].flatten(), trt_result[0].flatten(), atol=1e-5)
    # np.save("tf_test_matmul_bs256", tf_result[0])
    # np.save("tpat_test_matmul_bs256", trt_result[0])

    for i in range(len(trt_result)):
        print(tf_result[i].flatten()[:10])
        print(trt_result[i][:10])
        
        print(tf_result[i].shape)
        print(trt_result[i].shape)
        print(
            "trt cross_check output_%d " % i
            + str(np.allclose(tf_result[i].flatten(), trt_result[i], atol=1e-5)),
            flush=True,
        )
        print(
            "max diff " + str(np.fabs(tf_result[i].flatten() - trt_result[i]).max()),
            flush=True,
        )
        print(
            "min diff " + str(np.fabs(tf_result[i].flatten() - trt_result[i]).min()),
            flush=True,
        )

    # print(tf_time, flush=True)
    # print(trt_time, flush=True)

    cuda.Context.pop()


if __name__ == "__main__":
    main()
