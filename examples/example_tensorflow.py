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
sys.path.append("..")
from python import *
# import torch
# TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
IGPU = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(IGPU)
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)


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
        op_name = "rightshift"
        batch_size = 100
        dtype = "int32"
        in_shape = (3, 11)
        lh_data = np.random.randint(1, 8, size=(3, 11)).astype(dtype)
        rh_data = np.random.randint(1, 3, size=(3, 11)).astype(dtype)
        lft_data = tf.placeholder(dtype, in_shape, name="lft_data")
        rgt_data = tf.placeholder(dtype, in_shape, name="rgt_data")
        x = tf.bitwise.right_shift(lft_data, rgt_data, name=op_name)
        input_node = [lft_data, rgt_data]
        input_data = [lh_data, rh_data]
        output = tf.identity(x, name="output")
        sess.run(tf.global_variables_initializer())
        input_dict = {e: input_data[i] for i, e in enumerate(input_node)}
        input_with_num = ["lft_data:0", "rgt_data:0"]
        output_with_num = ["output:0"]
        time_sum = 0
        for i in range(20):
            tf_result = sess.run([output], input_dict)
        a = datetime.now()
        for i in range(iterations):
            tf_result = sess.run([output], input_dict)
        b = datetime.now()
        time_sum = (b - a).total_seconds()
        tf_time = (
            "[INFO] TF  execution time " + str(time_sum * 1000 / iterations) + " ms"
        )
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

    node_names = [op_name]
    trt_plugin_names = onnx2plugin(
        input_model_file, output_model_file, node_names=node_names
    )
    for trt_plugin_name in trt_plugin_names:
        assert os.path.isfile(f"./trt_plugin/lib/{trt_plugin_name}.so")
        ctypes.cdll.LoadLibrary("./trt_plugin/lib/{}.so".format(trt_plugin_name))

    # build trt model by onnx model
    cuda.Device(0).make_context()
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
        1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    ) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_batch_size = batch_size
        builder_config = builder.create_builder_config()
        builder_config.max_workspace_size = 1 << 30

        with open(output_model_file, "rb") as model:
            # parse onnx model
            parser.parse(model.read())
            for i in range(parser.num_errors):
                print(parser.get_error(i))
        engine = builder.build_engine(network, builder_config)
        if engine is None:
            print("[ERROR] engine is None")
            exit(-1)

        inputs, outputs, bindings, stream = allocate_buffers(engine)
        with engine.create_execution_context() as context:
            for i in range(len(inputs)):
                data = input_data[i].ravel()
                np.copyto(inputs[i].host, data)
                print(f"{i} input is {data}")
            for i in range(20):
                output = do_inference(
                    context,
                    bindings=bindings,
                    inputs=inputs,
                    outputs=outputs,
                    stream=stream,
                    batch_size=batch_size,
                )

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

    print(tf_result)
    print(trt_result)

    for i in range(len(trt_result)):
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

    print(tf_time, flush=True)
    print(trt_time, flush=True)

    cuda.Context.pop()


if __name__ == "__main__":
    main()
