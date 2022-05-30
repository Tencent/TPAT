##############################
# author : qianqiu
# email : qianqiu@tencent.com
# time : 2022.1.7
##############################
import os
import tvm
import tvm.relay as relay
from tvm.driver import tvmc
from tvm.contrib import graph_executor
from tvm import auto_scheduler
import onnx
import onnx_graphsurgeon as gs
import onnxruntime as ort
from onnx import shape_inference
import numpy as np


class CudaKernel(object):
    """Use Tvm AutoScheduler generate efficient cudaKernel and params needed by trt-plugin
    Parameters
    ----------
    Model path : str
        The onnx model
    Tuning name : str
        The Operator needed generate plugin which in onnx model
    """

    def __init__(
        self,
        model_path,
        tuning_node,
        plugin_name,
        one_node_model="submodel.onnx",
    ):
        self._model_path = model_path
        self._one_node_model = one_node_model
        self._tuning_name = tuning_node.name
        self._target = tvm.target.Target("cuda")
        self._log_file = "/tmp/tuning.log"
        if isinstance(model_path, str):
            self._onnx_model = onnx.load(model_path)
        else:
            self._onnx_model = model_path
        self._tuning_node = tuning_node
        self._onnx_op_type = tuning_node.op
        self._plugin_name = plugin_name
        # self._plugin_name = 'tpat_' + tuning_name

    def run(self, opt_level=3, input_data=None, opset=None):
        """
        Tvm Auto Scheduler
        """
        graph_def = self.extract_target_onnx_node(self._onnx_model)

        if not isinstance(input_data, list) and input_data is not None:
            input_data = [input_data]

        if input_data is not None:
            _, shape_dict = self.get_input_data_shape_dict(graph_def, input_data)

        if input_data is not None:
            mod, params = relay.frontend.from_onnx(graph_def, shape_dict, opset=opset)
        else:
            mod, params = relay.frontend.from_onnx(graph_def)
        tasks, weights = tvmc.autotuner.autoscheduler_get_tuning_tasks(
            mod, params, self._target, include_simple_tasks=True, opt_level=opt_level
        )

        #if len(tasks) != 0:
        #    self.tune(tasks, weights)

        # Compile with the history best
        print("Compile...", flush=True)
        with auto_scheduler.ApplyHistoryBest(self._log_file):
            with tvm.transform.PassContext(
                opt_level=opt_level, config={"relay.backend.use_auto_scheduler": True}
            ):
                self._lib = relay.build(mod, self._target, params=params)

        dev = tvm.device(str(self._target), 0)
        self._module = graph_executor.create(
            self._lib.get_graph_json(), self._lib.get_lib(), dev
        )

        print("Running...", flush=True)
        self._module.run()

    def check_plugin(self, onnx_node):
        """
        Check whether this operator's plugin had been generated.(For multiple ops which have the same type.)
        """
        print(
            "Check onnx node {}\n with plugin: {}".format(self._tuning_node, onnx_node)
        )
        if (
            self._tuning_node.op != onnx_node.op
            or self._tuning_node.attrs != onnx_node.attrs
        ):
            return False
        if len(self._tuning_node.inputs) != len(onnx_node.inputs) or len(
            self._tuning_node.outputs
        ) != len(onnx_node.outputs):
            return False
        for inp, onnx_node_inp in zip(self._tuning_node.inputs, onnx_node.inputs):
            if (
                inp.__class__ != onnx_node_inp.__class__
                or inp.shape != onnx_node_inp.shape
                or inp.dtype != onnx_node_inp.dtype
            ):
                return False
            if isinstance(inp, gs.ir.tensor.Constant):
                if not np.array_equal(inp.values, onnx_node_inp.values):
                    return False
        for out, onnx_node_out in zip(self._tuning_node.outputs, onnx_node.outputs):
            if (
                out.__class__ != onnx_node_out.__class__
                or out.shape != onnx_node_out.shape
                or out.dtype != onnx_node_out.dtype
            ):
                return False
        return True

    def check_existing_plugins(self, trt_plugin_mapping_onnx_node):
        for trt_plugin, onnx_node in trt_plugin_mapping_onnx_node.items():
            if self.check_plugin(onnx_node):
                return trt_plugin
        return None

    def compute_tensor_shape(self, input_model_path):
        """
        Get output shape through onnx-runtime and shape_inference.
        """
        inferred_model = shape_inference.infer_shapes(onnx.load(input_model_path))
        graph = gs.import_onnx(inferred_model)
        tuning_nodes = [node for node in graph.nodes if node.name == self._tuning_name]
        tuning_node = tuning_nodes[0]
        tuning_node_inputs = [
            graph.tensors()[inp.name].to_variable(dtype=inp.dtype, shape=inp.shape)
            for inp in tuning_node.inputs
            if inp.__class__ == gs.Variable and not inp.is_empty()
        ]
        tuning_node_outputs = [
            graph.tensors()[oup.name].to_variable(dtype=oup.dtype, shape=oup.shape)
            for oup in tuning_node.outputs
        ]
        graph.outputs = []
        graph.outputs.extend(tuning_node_inputs)
        graph.outputs.extend(tuning_node_outputs)
        # print("half graph: \n", graph.outputs)
        graph.cleanup()

        half_model = gs.export_onnx(graph)
        half_model_path = "half_model.onnx"
        onnx.save(half_model, half_model_path)
      
        #computed_tensor_shapes = [] 
        #for i in range(len(tuning_node_inputs)):
        #    computed_tensor_shapes.append(tuning_node_inputs[i].shape)
        #for i in range(len(tuning_node_outputs)):
        #    computed_tensor_shapes.append(tuning_node_outputs[i].shape)
        session = ort.InferenceSession(half_model_path)
        outname = [output.name for output in session.get_outputs()]
        dummy_input = {}
        for gi in graph.inputs:
            dummy_input[gi.name] = (1 + np.random.random([int(i) for i in gi.shape])).astype(gi.dtype)
        dummy_output = session.run(outname, dummy_input)

        computed_tensor_shapes = []
        for i in range(len(tuning_node_inputs)):
            assert tuning_node_inputs[i].name == outname[i]
            computed_tensor_shapes.append(dummy_output[i].shape)
            # print(f"node output {tuning_node_inputs[i].name} with shape {dummy_output[i].shape}")
        for i in range(len(tuning_node_outputs)):
            assert tuning_node_outputs[i].name == outname[len(tuning_node_inputs) + i]
            computed_tensor_shapes.append(
                dummy_output[len(tuning_node_inputs) + i].shape
            )
        os.remove(half_model_path)
        return computed_tensor_shapes

    def extract_target_onnx_node(self, model):
        """
        Extra target operator from onnx model
        """
        inferred_model = shape_inference.infer_shapes(model)
        graph = gs.import_onnx(inferred_model)
        nodes = graph.nodes
        tensors = graph.tensors()

        tuning_node_list = [node for node in nodes if node.name == self._tuning_name]

        assert (
            len(tuning_node_list) != 0
        ), "Not get tuning node in onnx model, please check op name or onnx model"

        tuning_node = tuning_node_list[0]
        # self._tuning_node = tuning_node
        # self._onnx_op_type = tuning_node.op
        # print("cuda_kernel.py tuning node: ", self._tuning_node.inputs[4].name, self._plugin_name)
        tuning_node_inputs = [
            tensors[inp.name].to_variable(dtype=inp.dtype, shape=inp.shape)
            for inp in tuning_node.inputs
            if (inp.__class__ == gs.Variable and not inp.is_empty())
        ]
        tuning_node_outputs = [
            tensors[oup.name].to_variable(dtype=oup.dtype, shape=oup.shape)
            for oup in tuning_node.outputs
        ]
        computed_tensor_shapes = self.compute_tensor_shape(
            self._model_path
        )
        # enhanced shape calculation
        for i in range(len(tuning_node_inputs)):
            tuning_node_inputs[i].shape = computed_tensor_shapes[i]
        for i in range(len(tuning_node_outputs)):
            tuning_node_outputs[i].shape = computed_tensor_shapes[
                len(tuning_node_inputs) + i
            ]
        graph.inputs = tuning_node_inputs
        graph.outputs = tuning_node_outputs
        graph.cleanup()
        submodel = gs.export_onnx(graph)
        onnx.save(submodel, self._one_node_model)
        return submodel

    def tune(self, tasks, task_weights):
        """
        The Search Config for Tvm AutoScheduler
        """
        print("Begin tuning...", flush=True)
        measure_ctx = auto_scheduler.LocalRPCMeasureContext(
            repeat=10, min_repeat_ms=300, timeout=10
        )

        tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=200,  # change this to 20000 to achieve the best performance
            runner=measure_ctx.runner,
            measure_callbacks=[auto_scheduler.RecordToFile(self._log_file)],
        )
        tuner.tune(tune_option)

    def get_input_data_shape_dict(self, graph_def, input_data):
        """
        Get shape of input data.
        """
        if isinstance(input_data, list):
            input_names = {}
            shape_dict = {}
            for i, _ in enumerate(input_data):
                input_names[i] = graph_def.graph.input[i].name
                shape_dict[input_names[i]] = input_data[i].shape
        else:
            input_names = graph_def.graph.input[0].name
            shape_dict = {input_names: input_data.shape}
        return input_names, shape_dict

    # Cuda Kernel generated by tvm.
    @property
    def cuda_source_code(self):
        try:
            source_code = self._lib.get_lib().imported_modules[0].get_source()
            source_code = source_code.replace("signed char*", "int*")
            source_code = source_code.replace("uint64_t*", "int*")
            source_code = source_code.replace("long long", "int")
            source_code = source_code.replace("double", "float")
        except IndexError:
            return None
        return source_code

    # Tvm runtime Module.
    @property
    def runtime_module(self):
        return self._lib

    # Tvm Graph executor
    @property
    def graph_module(self):
        return self._module

    # Constant params in operator. such as weight in Matmul operator.
    @property
    def constant_param(self):
        return self._lib.get_constant_params()

    # Tvm executor the order of device functions.
    @property
    def device_funcs_inorder(self):
        return self._lib.get_device_function_list()

    # The config of Grid. Block. Thread.
    @property
    def device_funcs_thread_config(self):
        return self._lib.get_grid_block_thread_config()

    # Independently allocated memory on the device side.
    @property
    def device_allocate_global_memory(self):
        return self._lib.get_device_memory_size()

    # The number of inputs.
    @property
    def num_inputs(self):
        return self._module.get_num_inputs()

    # The number of output.
    @property
    def num_outputs(self):
        return self._module.get_num_outputs()

    # The dtype of variables which are stored in memory.
    @property
    def workspace_dtype(self):
        return self._module.get_workspace_dtype()

    # The size of variables which are stored in memory.
    @property
    def workspace_size(self):
        return self._module.get_workspace_size()

    # Tvm executor the order of host functions.
    @property
    def func_inorder(self):
        return self._module.get_func_inorder()

    # The storage index in memory for each variable.
    @property
    def storageid(self):
        return self._module.get_storageid()

    # Generated plugin name
    @property
    def plugin_name(self):
        return self._plugin_name

    # Tuning op type.
    @property
    def onnx_op_type(self):
        return self._onnx_op_type

    # Tuning op name.
    @property
    def tuning_name(self):
        return self._tuning_name
