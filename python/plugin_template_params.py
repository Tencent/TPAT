##############################
# author : qianqiu
# email : qianqiu@tencent.com
# time : 2022.1.7
##############################
import os
import onnx
from onnx import shape_inference
import onnx_graphsurgeon as gs
import onnxruntime as ort
import numpy as np
from type_mapping import (
    tvm_to_c_type_mapping,
    python_to_trt_type_mapping,
    plugin_type_size,
)


class PluginTemplateParams(object):
    """
    Generate useable params for tensorRT plugin.
    """

    def __init__(
        self,
        cuda_kernel,
        onnx_path,
        tuning_name,
        one_node_model="submodel.onnx",
    ):
        self._kernel_generate = cuda_kernel
        self._onnx_path = onnx_path
        self._one_node_model = one_node_model
        self._tuning_name = tuning_name

        self._onnx_input_order = []
        self._input_dict = {}
        self._tvm_executor_order = {}
        self._allocate_size = []
        self._data_type = []
        self._cuda_kernel_order = {}
        self._gpu_thread_config = {}
        self._tvm_func_order = []
        self._nums_input = 0
        self._nums_output = 0
        self._workspace_size = 0
        self._output_type = []
        self._cuda_func_order = []
        self._tvm_constant = {}
        self._tvm_workspace_constant = {}
        self._onnx_input_shape = []
        self._onnx_output_shape = []
        self._onnx_weight_input_index = []
        self._onnx_tensor_input_index = []
        self._onnx_tensor_type = []
        self._onnx_input_python_type = []
        self._onnx_output_python_type = []
        self._storage_id = []
        self._allocate_global_memory = {}
        self._plugin_config = None

        self.infer_for_output_shape()
        self.input_weight_and_tensor_index()
        self.parse()
        self.align_onnx_and_tvm_input(self._one_node_model)
        self.match_address_for_eid()
        self.cuda_kernel_config()
        os.remove(self._one_node_model)

    # Parse Constant.
    def parse_constant_params(self, constant_params):
        tvm_constant = {}
        for key, value in constant_params.items():
            tvm_constant[key] = value.flatten()
        return tvm_constant

    # Parse device functions params order.
    def parse_device_funcs_params(self, device_funcs_inorder):
        cuda_kernel_order = {}
        for device_func_inorder in device_funcs_inorder:
            if len(device_func_inorder) == 0:
                continue
            tvm_device_func = device_func_inorder.split()
            cuda_kernel_order[tvm_device_func[0]] = tvm_device_func[1:]
        return cuda_kernel_order

    # Parse device functions thread config.
    def parse_device_funcs_thread_config(self, device_funcs_thread_config):
        gpu_thread_config = {}
        cuda_func_order = []
        for device_func_thread_config in device_funcs_thread_config:
            if len(device_func_thread_config) == 0:
                continue
            config = device_func_thread_config.split()
            cuda_func_name = config[0]
            gpu_thread_config[cuda_func_name] = config[1:]
            cuda_func_order.append(cuda_func_name)
        return gpu_thread_config, cuda_func_order

    # Parse global memory allocated in device side.
    def parse_device_allocate_global_memory(self, device_allocate_global_memory):
        allocate_global_memory = {}
        for allocate_memory in device_allocate_global_memory:
            if len(allocate_memory) == 0:
                continue
            allocate = allocate_memory.split()
            allocate_global_memory[allocate[0]] = allocate[1:]
        return allocate_global_memory

    # Parse variables storage index.
    def parse_storageid(self, storageid):
        storage_id = []
        storage_slot = {}
        for sid in storageid:
            if len(sid) == 0:
                continue
            storage_id = sid.split()
            storage_slot = {}.fromkeys(sid).keys()
        return storage_id, storage_slot

    # Parse numbers of input.
    def parse_nums_input(self, nums_input):
        real_nums_input = int(nums_input) - int(len(self._tvm_constant))
        return real_nums_input

    # Parse numbers of output.
    def parse_nums_output(self, nums_output):
        real_nums_output = int(nums_output)
        return real_nums_output

    # Parse datatype of variables in memory.
    def parse_workspace_dtype(self, workspaces_dtype):
        return workspaces_dtype.split()

    # Parse size of variables in memory.
    def parse_workspace_size(self, workspace_size):
        return workspace_size.split()

    def parse_func_inorder(self, funcs_inorder):
        """
        Parse the order of host functions.
        """
        func_call = {}
        tvm_executor_order = {}
        tvm_func_order = []
        for host_func_inorder in funcs_inorder:
            if len(host_func_inorder) == 0:
                continue
            tvm_host_func = host_func_inorder.split()
            if tvm_host_func[0] not in tvm_executor_order.keys():
                tvm_executor_order[tvm_host_func[0]] = tvm_host_func[1:]
                tvm_func_order.append(tvm_host_func[0])
                func_call[tvm_host_func[0]] = 0
            else:
                func_call[tvm_host_func[0]] += 1
                func_name = tvm_host_func[0] + "_" + str(func_call[tvm_host_func[0]])
                tvm_executor_order[func_name] = tvm_host_func[1:]
                tvm_func_order.append(func_name)
        return tvm_executor_order, tvm_func_order

    # Parse
    def parse(self):
        constant_params = self._kernel_generate.constant_param
        device_funcs_inorder = self._kernel_generate.device_funcs_inorder.split("\n")
        device_funcs_thread_config = (
            self._kernel_generate.device_funcs_thread_config.split("\n")
        )
        device_allocate_global_memory = (
            self._kernel_generate.device_allocate_global_memory.split("\n")
        )
        num_inputs = self._kernel_generate.num_inputs
        num_outputs = self._kernel_generate.num_outputs
        workspace_dtype = self._kernel_generate.workspace_dtype
        workspace_size = self._kernel_generate.workspace_size
        funcs_inorder = self._kernel_generate.func_inorder.split("\n")
        storage_id = self._kernel_generate.storageid.split("\n")

        self._tvm_constant = self.parse_constant_params(constant_params)
        self._cuda_kernel_order = self.parse_device_funcs_params(device_funcs_inorder)
        (
            self._gpu_thread_config,
            self._cuda_func_order,
        ) = self.parse_device_funcs_thread_config(device_funcs_thread_config)

        self._nums_input = self.parse_nums_input(num_inputs)
        self._nums_output = self.parse_nums_output(num_outputs)
        self._data_type = self.parse_workspace_dtype(workspace_dtype)

        self._allocate_size = self.parse_workspace_size(workspace_size)
        self._tvm_executor_order, self._tvm_func_order = self.parse_func_inorder(
            funcs_inorder
        )
        self._cuda_source_code = self._kernel_generate.cuda_source_code
        self._storage_id, self.storage_slot = self.parse_storageid(storage_id)
        self._allocate_global_memory = self.parse_device_allocate_global_memory(
            device_allocate_global_memory
        )
        self._input_workspace_size = self._allocate_size[0:self._nums_input]
        self._output_workspace_size = self._allocate_size[-self._nums_output:]

    def infer_for_output_shape(self):
        """
        infer for output shape
        """
        model = onnx.load(self._onnx_path)
        inferred_model = shape_inference.infer_shapes(model)
        graph = gs.import_onnx(inferred_model)
        nodes = graph.nodes
        tensors = graph.tensors()
        tuning_nodes = [node for node in nodes if node.name == self._tuning_name]
        assert len(tuning_nodes) != 0
        tuning_node = tuning_nodes[0]
        for inp in tuning_node.inputs:
            if inp.__class__ == gs.Constant:
                self._onnx_input_python_type.append(tvm_to_c_type_mapping[inp.dtype.__name__])
                self._onnx_tensor_type.append(
                    python_to_trt_type_mapping[inp.dtype.__name__]
                )
            elif not inp.is_empty():
                self._onnx_input_python_type.append(tvm_to_c_type_mapping[inp.dtype.name])
                self._onnx_tensor_type.append(
                    python_to_trt_type_mapping[inp.dtype.name]
                )

        for oup in tuning_node.outputs:
            self._onnx_output_python_type.append(tvm_to_c_type_mapping[oup.dtype.name])
            self._onnx_tensor_type.append(python_to_trt_type_mapping[oup.dtype.name])

        graph.outputs = [
            tensors[oup.name].to_variable(dtype=oup.dtype, shape=oup.shape)
            for oup in tuning_node.outputs
        ]
        graph.cleanup()
        self._onnx_output_shape = self.dummy_onnx_ort_output_shape(graph)

        #graph.outputs = [
        #    tensors[inp.name].to_variable(dtype=inp.dtype, shape=inp.shape)
        #    for k, inp in enumerate(tuning_node.inputs) if (inp.__class__ == gs.Variable and not (len(inp.inputs) == 1 and tuning_node.i(k, 0).op == "Constant"))
        #]
        #print(graph.outputs)
        graph.outputs = [
            tensors[inp.name].to_variable(dtype=inp.dtype, shape=inp.shape)
            for inp in tuning_node.inputs if (inp.__class__ == gs.Variable and 'inserted_const_for_' not in inp.name and not inp.is_empty())
        ]
        ### for debug
        # graph.outputs = []
        # for k, inp in enumerate((tuning_node.inputs)):
        #     if inp.__class__ == gs.Variable:
        #         if not (len(inp.inputs) == 1 and tuning_node.i(k, 0).op == "Constant"): 
        #             graph.outputs.append(tensors[inp.name].to_variable(dtype=inp.dtype, shape=inp.shape))
        #         else:
        #             print("tuning_node inputs: ", tuning_node.i(k, 0))
        ### for debug
        graph.cleanup()
        self._onnx_input_shape = self.dummy_onnx_ort_output_shape(graph)

    def dummy_onnx_ort_output_shape(self, graph):
        onnx_output_shape = []
        submodel = gs.export_onnx(graph)
        dummy_model = "dummy_model.onnx"
        onnx.save(submodel, dummy_model)
        session = ort.InferenceSession(dummy_model)
        outname = [output.name for output in session.get_outputs()]
        dummy_input = {}
        for gi in graph.inputs:
            dummy_input[gi.name] = (np.random.random(gi.shape) + 1).astype(gi.dtype)
        dummy_output = session.run(outname, dummy_input)
        for i in range(len(dummy_output)):
            onnx_output_shape.append(dummy_output[i].shape)
        os.remove(dummy_model)
        return onnx_output_shape

    def input_weight_and_tensor_index(self):
        """
        calculate the index of weight input and tensor input
        """
        graph = gs.import_onnx(onnx.load(self._onnx_path))
        tuning_nodes = [node for node in graph.nodes if node.name == self._tuning_name]
        assert len(tuning_nodes) != 0
        tuning_node = tuning_nodes[0]
        self._onnx_tensor_input_index = [k for k, inp in enumerate(tuning_node.inputs) 
            if (inp.__class__ == gs.Variable and not (len(inp.inputs) == 1 and tuning_node.i(k, 0).op == "Constant"))]
        self._onnx_weight_input_index = [k for k, inp in enumerate(tuning_node.inputs) 
            if (inp.__class__ == gs.Constant or (len(inp.inputs) == 1 and tuning_node.i(k, 0).op == "Constant"))]

    def align_onnx_and_tvm_input(self, onnx_path):
        """
        align onnx and tvm input. Because tvm let constants in the after of variables params.
        """
        model = onnx.load(onnx_path)
        graph = model.graph
        nodes = graph.node
        onnx_inputs = graph.input
        init_order = {}
        for node in nodes:
            op_inputs = node.input
            for i in range(len(op_inputs)):
                init_order[op_inputs[i]] = i
        for i in onnx_inputs:
            self._onnx_input_order.append(init_order[i.name])

    def match_address_for_eid(self):
        """
        The memory address used by functions params.
        """
        workspace = 0
        input_slot_dict = {}
        for i in range(self._nums_output):
            eid = self._kernel_generate.graph_module.get_output_eid(i)
            idx = int(self._storage_id[eid])
            self._output_type.append(python_to_trt_type_mapping[self._data_type[eid]])
            self._input_dict[str(eid)] = "outputs[" + str(i) + "]"
            input_slot_dict[idx] = self._input_dict[str(eid)]
        duplicate_allocate = {}
        for i in range(len(self._allocate_size)):
            idx = int(self._storage_id[i])
            if idx not in duplicate_allocate.keys():
                duplicate_allocate[idx] = 0
            duplicate_allocate[idx] = max(
                int(self._allocate_size[i]), int(duplicate_allocate[idx])
            )
        for i in range(len(self._allocate_size)):
            idx = int(self._storage_id[i])
            if idx in input_slot_dict.keys():
                self._input_dict[str(i)] = input_slot_dict[idx]
                continue
            if i < self._nums_input:
                self._input_dict[str(i)] = (
                    "inputs[" + str(self._onnx_input_order[i]) + "]"
                )
            elif i < len(self._allocate_size) - self._nums_output:
                if i == self._nums_input:
                    self._input_dict[str(i)] = "workspace"
                else:
                    self._input_dict[str(i)] = "(workspace + " + str(workspace) + ")"
                workspace += int(duplicate_allocate[idx])
                self._workspace_size = workspace
                if (
                    self._input_dict[str(i)] not in self._tvm_workspace_constant.keys()
                    and str(idx) in self._tvm_constant.keys()
                ):
                    # self._tvm_workspace_constant[self._input_dict[str(i)]] = None
                    self._tvm_workspace_constant[self._input_dict[str(i)]] = (
                        self._tvm_constant[str(idx)],
                        tvm_to_c_type_mapping[self._data_type[i]],
                        int(i),
                    )
            input_slot_dict[idx] = self._input_dict[str(i)]
        if len(self._allocate_global_memory) != 0:
            for key, value in self._allocate_global_memory.items():
                self._input_dict[key] = (
                    "("
                    + tvm_to_c_type_mapping[value[0]]
                    + "*)(workspace + "
                    + str(workspace)
                    + ")"
                )
                workspace += int(value[1]) * plugin_type_size[value[0]]
                self._workspace_size = workspace

    def cuda_kernel_config(self):
        """
        Grid. Block. Thread. size.
        """
        output = ""
        output_json = {}
        cuda_func_call = {}
        for i in range(len(self._cuda_func_order)):
            cuda_func_name = self._cuda_func_order[i]
            import re

            func_name = re.sub("_kernel\d+", "", cuda_func_name, count=1)
            if cuda_func_name not in output_json.keys():
                output_json[cuda_func_name] = {}
                cuda_func_call[cuda_func_name] = 0
                multi_cuda_func_name = cuda_func_name
            else:
                cuda_func_call[cuda_func_name] += 1
                func_name = func_name + "_" + str(cuda_func_call[cuda_func_name])
                multi_cuda_func_name = (
                    cuda_func_name + "_" + str(cuda_func_call[cuda_func_name])
                )
                output_json[multi_cuda_func_name] = {}

            output_json[multi_cuda_func_name]["grid_dim"] = self._gpu_thread_config[
                cuda_func_name
            ][0].strip("grid=")
            output_json[multi_cuda_func_name]["block_dim"] = self._gpu_thread_config[
                cuda_func_name
            ][1].strip("block=")
            output += (
                cuda_func_name
                + "\n"
                + str(self._gpu_thread_config[cuda_func_name])
                + "\n"
            )

            kernel_param_order = self._cuda_kernel_order[cuda_func_name]

            tvm_param_order = self._tvm_executor_order[func_name]
            enqueue_params = ""
            for j in range(len(kernel_param_order)):
                # output += self._input_dict[int(tvm_param_order[int(kernel_param_order[j])])]
                if kernel_param_order[j].isdigit():
                    # enqueue_params += self._input_dict[str(tvm_param_order[int(kernel_param_order[j])])]
                    output += self._input_dict[
                        str(tvm_param_order[int(kernel_param_order[j])])
                    ]
                    eid = tvm_param_order[int(kernel_param_order[j])]
                    enqueue_params += (
                        "("
                        + tvm_to_c_type_mapping[self._data_type[int(eid)]]
                        + "*)"
                        + self._input_dict[str(eid)]
                    )
                else:
                    if kernel_param_order[j] in self._input_dict.keys():
                        enqueue_params += self._input_dict[kernel_param_order[j]]
                if j == len(kernel_param_order) - 1:
                    output += "\n"
                else:
                    output += ", "
                    enqueue_params += ", "
            output_json[multi_cuda_func_name]["enqueue_params"] = enqueue_params
        self._plugin_config = output_json

    @property
    def host_func_order(self):
        return self._tvm_func_order

    @property
    def kernel_order(self):
        return self._cuda_func_order

    @property
    def plugin_config(self):
        return self._plugin_config

    @property
    def workspace_size(self):
        return self._workspace_size

    @property
    def output_num(self):
        return self._nums_output

    @property
    def output_type(self):
        return self._output_type

    @property
    def output_shape(self):
        return self._onnx_output_shape

    @property
    def input_shape(self):
        return self._onnx_input_shape

    @property
    def onnx_weight_input_index(self):
        return self._onnx_weight_input_index

    @property
    def onnx_tensor_input_index(self):
        return self._onnx_tensor_input_index

    @property
    def tensor_type(self):
        return self._onnx_tensor_type

    @property
    def workspace_init(self):
        return self._tvm_workspace_constant

    @property
    def cuda_source_code(self):
        return self._cuda_source_code

    @property
    def plugin_name(self):
        return self._kernel_generate.plugin_name

    @property
    def onnx_op_type(self):
        return self._kernel_generate.onnx_op_type

    @property
    def storage_id(self):
        return self._storage_id

    @property
    def onnx_input_python_type(self):
        return self._onnx_input_python_type

    @property
    def onnx_output_python_type(self):
        return self._onnx_output_python_type

    @property
    def input_workspace_size(self):
        return self._input_workspace_size

    @property
    def output_workspace_size(self):
        return self._output_workspace_size
   
    @property
    def total_workspace_size(self):
        allocate_size = 0
        for size in self._allocate_size:
            allocate_size += int(size) 
        return allocate_size 
