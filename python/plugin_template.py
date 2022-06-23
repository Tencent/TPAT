##############################
# author : qianqiu
# email : qianqiu@tencent.com
# time : 2022.1.7
##############################
import os
import contextlib
import re
import onnx
import onnx_graphsurgeon as gs
from onnx import shape_inference
from jinja2 import FileSystemLoader, Environment


@contextlib.contextmanager
def pushd(new_dir):
    pre_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(pre_dir)

def rm_part_define(source_code):
    m = re.search('extern "C"', source_code.strip())  
    return source_code[m.start() :]

class PluginTemplate(object):
    def __init__(self, template_params):
        self._template_params = template_params
        self._plugin_name = template_params.plugin_name
        self._plugin_config = template_params.plugin_config
        with pushd(os.path.normpath(os.path.dirname(__file__))):
            template_loader = FileSystemLoader(searchpath="./")
        self._template_env = Environment(loader=template_loader)
        self._plugin_output_number = template_params.output_num
        self._plugin_output_type = template_params.output_type
        self._plugin_workspace_size = template_params.workspace_size
        self._plugin_total_workspace_size = template_params.total_workspace_size
        onnx_output_shape = template_params.output_shape
        onnx_input_shape = template_params.input_shape
        self._plugin_output_shape = self.parse_plugin_output_shape(onnx_output_shape)
        self._plugin_input_shape = self.parse_plugin_input_shape(onnx_input_shape)
        self._plugin_tensor_input_index = template_params.onnx_tensor_input_index
        onnx_tensor_type = template_params.tensor_type
        self._plugin_tensor_format = self.parse_plugin_tensor_format(onnx_tensor_type)
        kernel_order = template_params.kernel_order
        workspace_init = template_params.workspace_init
        self._plugin_kernels_params = self.parse_plugin_kernels_params(kernel_order)
        self._plugin_constant_init = self.parse_plugin_workspace_init(workspace_init)
        self._plugin_kernels_body = template_params.cuda_source_code
        self._onnx_input_python_type = template_params.onnx_input_python_type
        self._onnx_output_python_type = template_params.onnx_output_python_type
        self._input_workspace_size = template_params.input_workspace_size
        self._output_workspace_size = template_params.output_workspace_size

    @property
    def plugin_name(self):
        return self._plugin_name

    class TensorDims:
        def __init__(self, nbdims, shape, dynamic_dim=None):
            self.nbdims = nbdims
            self.shape = tuple(shape)
            self.dynamic_dim = dynamic_dim

    class TensorFormat:
        def __init__(self, format, type):
            self.format = format
            self.type = type

    class Kernel:
        def __init__(
            self,
            name,
            grid_dim,
            block_dim,
            enqueue_params,
            kernel_params=None,
            code=None,
        ):
            self.name = name
            self.grid_dim = grid_dim
            self.block_dim = block_dim
            self.enqueue_params = enqueue_params
            self.kernel_params = kernel_params
            self.code = code

    class Constant:
        def __init__(self, pos, value, type, index, length):
            self.pos = pos
            self.value = value
            self.type = type
            self.index = index
            self.length = length

    class Case:
        def __init__(self, batch_size, plugin_template, dy_plugin_input_size_type_without_bs=None, dy_plugin_output_size_type_without_bs=None):
            self.batch_size = batch_size
            self.plugin_template = plugin_template
            self.dy_plugin_input_size_type_without_bs = dy_plugin_input_size_type_without_bs
            self.dy_plugin_output_size_type_without_bs = dy_plugin_output_size_type_without_bs

    class Shape:
        def __init__(self, size, dtype):
            self.size = size
            self.dtype = dtype

    def parse_plugin_input_shape(self, onnx_input_shape):
        plugin_input_shape = []
        for s in onnx_input_shape:
            nbdims = len(s)
            shape = s
            plugin_input_shape.append(self.TensorDims(nbdims, shape))
        return plugin_input_shape

    def parse_plugin_output_shape(self, onnx_output_shape):
        plugin_output_shape = []
        for s in onnx_output_shape:
            nbdims = len(s)
            shape = s
            plugin_output_shape.append(self.TensorDims(nbdims, shape))
        return plugin_output_shape

    def parse_plugin_tensor_format(self, onnx_tensor_type):
        plugin_tensor_format = []
        for dtype in onnx_tensor_type:
            plugin_tensor_format.append(self.TensorFormat("LINEAR", dtype))
        return plugin_tensor_format

    def parse_plugin_kernels_params(self, kernel_order):
        kernel_call = {}
        plugin_kernels_params = []
        for func_name in kernel_order:
            if func_name not in kernel_call.keys():
                kernel_call[func_name] = 0
                key_name = func_name
            else:
                kernel_call[func_name] += 1
                key_name = func_name + "_" + str(kernel_call[func_name])
            plugin_kernels_params.append(
                self.Kernel(
                    func_name,
                    self._plugin_config[key_name]["grid_dim"],
                    self._plugin_config[key_name]["block_dim"],
                    self._plugin_config[key_name]["enqueue_params"],
                )
            )
        return plugin_kernels_params

    def parse_plugin_workspace_init(self, workspace_init):
        plugin_constant_init = []
        for init_constant in workspace_init.items():
            value_str = ""
            for ele in init_constant[1][0]:
                value_str = value_str + str(ele) + " ,"
            value_str = value_str.strip(",")
            plugin_constant_init.append(
                self.Constant(
                    init_constant[0],
                    # init_constant[1][0],
                    value_str,
                    init_constant[1][1],
                    init_constant[1][2],
                    len(init_constant[1][0]),
                )
            )
        return plugin_constant_init

    def generate_header_file(self):
        raise Exception("not implement method")
        
    def generate_source_file(self):
        raise Exception("not implement method")

    def fill(self):
        plugin_header_path = f"./trt_plugin/src/{self.plugin_name}.h"
        plugin_source_path = f"./trt_plugin/src/{self.plugin_name}.cu"
        if os.path.isfile(plugin_header_path):
            os.remove(plugin_header_path)
        if os.path.isfile(plugin_source_path):
            os.remove(plugin_source_path)
        with pushd(os.path.normpath(os.path.dirname(__file__))): 
            self.generate_header_file()
            self.generate_source_file()
            self.build_plugin()
    
    def build_plugin(self):
        os.chdir("./trt_plugin")
        os.system(f"make clean plugin_name={self.plugin_name}")
        os.system(f"make plugin_name={self.plugin_name}")
        os.chdir("../")

class StaticBatchPluginTemplate(PluginTemplate):
    """
    Fill in the useable params which generated by PluginTemplateParams to plugin template.
    The plugin template is compatible with TensorRT-8.0.
    """

    def __init__(
        self,
        template_params,
        TEMPLATE_HEADER_FILE="./trt_plugin/trt8.0_plugin_h.template",
        TEMPLATE_SOURCE_FILE="./trt_plugin/trt8.0_plugin_cu.template",
    ):
        super(StaticBatchPluginTemplate, self).__init__(template_params)
        self._template_header_file = TEMPLATE_HEADER_FILE
        self._template_source_file = TEMPLATE_SOURCE_FILE
    
    def generate_header_file(self):
        template = self._template_env.get_template(self._template_header_file)
        output_text = template.render(
            plugin_name=self._plugin_name,
            plugin_output_number=self._plugin_output_number,
            plugin_output_shape=self._plugin_output_shape,
            plugin_output_type=self._plugin_output_type,
            plugin_workspace_size=self._plugin_workspace_size,
            plugin_tensor_format=self._plugin_tensor_format,
        )
        with open("./trt_plugin/src/{}.h".format(self._plugin_name), "w") as f:
            f.write(output_text)

    def generate_source_file(self):
        template = self._template_env.get_template(self._template_source_file)
        output_text = template.render(
            plugin_name=self._plugin_name,
            plugin_kernels_params=self._plugin_kernels_params,
            plugin_kernels_body=self._plugin_kernels_body,
            plugin_constant_init=self._plugin_constant_init,
        )
        with open("./trt_plugin/src/{}.cu".format(self._plugin_name), "w") as f:
            f.write(output_text)

class DynamicBatchPluginTemplate(PluginTemplate):
    def __init__(
        self,
        template_params,
        naive_onnx_model_path,
        TEMPLATE_HEADER_FILE="./trt_plugin/trt8.0_plugin_h_dynamic.template",
        TEMPLATE_SOURCE_FILE="./trt_plugin/trt8.0_plugin_cu_dynamic.template",
    ):
        super(DynamicBatchPluginTemplate, self).__init__(template_params)
        self._template_env.filters['rm_part_define'] = rm_part_define 
        self._template_header_file = TEMPLATE_HEADER_FILE
        self._template_source_file = TEMPLATE_SOURCE_FILE
        self._plugin_template_list = list()
        # {0 : 1} : input_0 index1 is batchSize
        self._batch_dim_in_inputs, self._batch_dim_in_outputs, self._input_dim_shape_without_bs, self._output_dim_shape_without_bs = self.get_batch_dim_in_input_output(naive_onnx_model_path)
        self._dy_plugin_output_shape = self.get_dynamic_output_shape(self._plugin_output_shape)
        self._dy_plugin_output_size_type = list()
        self._dy_plugin_input_size_type = list()
        self._dy_plugin_output_workspace_size_type = list()
        self.first_push = True

    # get dim of batch size in inputs and outputs
    def get_batch_dim_in_input_output(self, naive_onnx_model_path):
        inferred_model = shape_inference.infer_shapes(onnx.load(naive_onnx_model_path))
        graph = gs.import_onnx(inferred_model)
        tuning_nodes = [node for node in graph.nodes if node.name == self._template_params._tuning_name] 
        tuning_node = tuning_nodes[0]
        batch_dim_in_inputs, batch_dim_in_outputs = {}, {}
        input_dim_shape_without_bs, output_dim_shape_without_bs = [],[]
        input_dynamic_shape_num , output_dynamic_shape_num = 0, 0
        for idx, inp in enumerate(tuning_node.inputs):
            one_input_shape_without_bs = []
            if inp.__class__ == gs.Variable and not inp.is_empty() and inp.shape is not None:
                for i, dim in enumerate(inp.shape):
                    if not str(dim).isdigit():
                        batch_dim_in_inputs[idx] = i 
                        input_dynamic_shape_num += 1
                    else:
                        one_input_shape_without_bs.append(dim)
            input_dim_shape_without_bs.append(one_input_shape_without_bs)
        for idx, oup in enumerate(tuning_node.outputs):
            one_output_shape_without_bs = []
            if oup.__class__ == gs.Variable and not oup.is_empty() and oup.shape is not None:
                for i, dim in enumerate(oup.shape):
                    if not str(dim).isdigit():
                        batch_dim_in_outputs[idx] = i 
                        output_dynamic_shape_num += 1
                    else:
                        one_output_shape_without_bs.append(dim)
            output_dim_shape_without_bs.append(one_output_shape_without_bs)
        if not batch_dim_in_inputs or input_dynamic_shape_num > 1:
            tensors = graph.tensors()
            graph.outputs = [
                tensors[inp.name].to_variable(dtype=inp.dtype, shape=inp.shape)
                for inp in tuning_node.inputs if (inp.__class__ == gs.Variable and not inp.is_empty())
            ] 
            batch_dim_in_inputs, input_dim_shape_without_bs = self.onnx_runtime_get_input_output_shape(graph)
        if not batch_dim_in_outputs or output_dynamic_shape_num > 1:
            tensors = graph.tensors()
            graph.outputs = [
                tensors[oup.name].to_variable(dtype=oup.dtype, shape=oup.shape)
                for oup in tuning_node.outputs
            ] 
            batch_dim_in_outputs, output_dim_shape_without_bs = self.onnx_runtime_get_input_output_shape(graph)

        return batch_dim_in_inputs, batch_dim_in_outputs, input_dim_shape_without_bs, output_dim_shape_without_bs

    def onnx_runtime_get_input_output_shape(self, graph): 
        batch_dim_in_inputs = {}
        input_dim_shape_without_bs = []
        submodel = gs.export_onnx(graph)
        dummy_submodel = 'dummy_submodel.onnx'
        onnx.save(submodel, dummy_submodel)
        import onnxruntime as ort
        import numpy as np
        EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(dummy_submodel, providers=EP_list)
        outname = [output.name for output in session.get_outputs()]
        dummy_input = {}
        dummy_input_bak = {}
        batch = np.random.randint(1, 128)
        batch_bak = batch + 1
        for gi in graph.inputs:
            input_shape = gi.shape
            dummy_input_shape = []
            dummy_input_shape_bak = []
            for dim in input_shape:
                if not str(dim).isdigit():
                    dummy_input_shape.append(batch)
                    dummy_input_shape_bak.append(batch_bak)
                else:
                    dummy_input_shape.append(dim) 
                    dummy_input_shape_bak.append(dim) 
            dummy_input[gi.name] = (np.random.random(dummy_input_shape)).astype(gi.dtype) 
            dummy_input_bak[gi.name] = (np.random.random(dummy_input_shape_bak)).astype(gi.dtype) 
        dummy_output = session.run(outname, dummy_input)
        dummy_output_bak = session.run(outname, dummy_input_bak)
        for i, oup in enumerate(dummy_output):
            one_output_shape_without_bs = []
            for j, dim in enumerate(oup.shape):
                if dim != dummy_output_bak[i].shape[j]:
                    batch_dim_in_inputs[i] = j 
                else:
                    one_output_shape_without_bs.append(dim)
            input_dim_shape_without_bs.append(one_output_shape_without_bs) 
        os.remove(dummy_submodel)
        return batch_dim_in_inputs, input_dim_shape_without_bs

    def get_dynamic_output_shape(self, plugin_output_shape):
        dy_plugin_output_shape = []
        for idx, output_shape in enumerate(plugin_output_shape):
            dy_plugin_output_shape.append(
                self.TensorDims(
                    dynamic_dim=self._batch_dim_in_outputs[idx],
                    nbdims=output_shape.nbdims,
                    shape=output_shape.shape
                )
            )
        return dy_plugin_output_shape

    def get_dynamic_shape_size(self, shape_dims):
        from functools import reduce
        from operator import mul
        dy_plugin_shape_size = []
        for dims in shape_dims:
            dy_plugin_shape_size.append(reduce(mul, dims.shape))
        return dy_plugin_shape_size

    def get_dynamic_shape_size_type(self, dy_plugin_shape_size, onnx_python_type):
        dynamic_shape_size_type = list()
        for i in range(len(dy_plugin_shape_size)):
            dynamic_shape_size_type.append(
                self.Shape(
                    size=int(dy_plugin_shape_size[i]),
                    dtype=onnx_python_type[i]
                )
            )
        return dynamic_shape_size_type

    def get_dynamic_shape_size_type_without_bs(self, dy_shape_size_type, batch_size):
        dy_shape_size_type_without_bs = list()
        for shape in dy_shape_size_type:
            dy_shape_size_type_without_bs.append(
                self.Shape(
                    size=shape.size / batch_size,
                    dtype=shape.dtype
                )
            )
        return dy_shape_size_type_without_bs

    def push_plugin_template(self, batch_size, plugin_template):
        duplicate_list = set()
        _dy_plugin_input_size = self.get_dynamic_shape_size(plugin_template._plugin_input_shape)
        _dy_plugin_output_size = self.get_dynamic_shape_size(plugin_template._plugin_output_shape)

        for kernel in plugin_template._plugin_kernels_params:
            func_name = kernel.name
            func_name_with_bs = func_name + '_bs' + str(batch_size)
            kernel.name = func_name_with_bs
            if kernel.name in duplicate_list:
                continue
            duplicate_list.add(kernel.name)
            plugin_template._plugin_kernels_body = plugin_template._plugin_kernels_body.replace(func_name + '(', func_name_with_bs + '(')
            
        for kernel in plugin_template._plugin_kernels_params:
            #print("_plugin_tensor_input_index: ", self._plugin_tensor_input_index)
            assert(len(self._plugin_tensor_input_index) > 0, "incorrect _plugin_tensor_input_index")
            if not self.first_push:            
                for i in range(len(_dy_plugin_input_size)):
                    index = self._plugin_tensor_input_index[i]
                    kernel.enqueue_params = kernel.enqueue_params.replace("inputs[{}]".format(index), "(workspace + offset_input_{})".format(i))
                for i in range(len(_dy_plugin_output_size)):
                    kernel.enqueue_params = kernel.enqueue_params.replace("outputs[{}]".format(i), "(workspace + offset_output_{})".format(i))
        self.first_push = False
        
        kernels_body = plugin_template._plugin_kernels_body.split('extern "C"')
        real_kernels_body = []
        real_kernels_body.append(kernels_body[0])
        for func_name in duplicate_list:
            for func_body in kernels_body:
                if func_name in func_body:
                    real_kernels_body.append('extern "C"' + func_body)
        plugin_template._plugin_kernels_body = '\n'.join(real_kernels_body)

        self._plugin_workspace_size = max(self._plugin_workspace_size, plugin_template._plugin_workspace_size)
        self._plugin_total_workspace_size = max(self._plugin_total_workspace_size, plugin_template._plugin_total_workspace_size)
        _dy_plugin_input_size = self.get_dynamic_shape_size(plugin_template._plugin_input_shape)
        _dy_plugin_output_size = self.get_dynamic_shape_size(plugin_template._plugin_output_shape)
        self._dy_plugin_input_size_type = self.get_dynamic_shape_size_type(_dy_plugin_input_size, self._onnx_input_python_type)
        self._dy_plugin_output_size_type = self.get_dynamic_shape_size_type(_dy_plugin_output_size, self._onnx_output_python_type)
        self._dy_plugin_output_workspace_size_type = self.get_dynamic_shape_size_type(plugin_template._output_workspace_size, self._onnx_output_python_type)
        self._dy_plugin_input_size_type_without_bs = self.get_dynamic_shape_size_type_without_bs(self._dy_plugin_input_size_type, batch_size)
        self._dy_plugin_output_size_type_without_bs = self.get_dynamic_shape_size_type_without_bs(self._dy_plugin_output_size_type, batch_size)

        self._plugin_template_list.append(
            self.Case(
                batch_size,
                plugin_template,
                dy_plugin_input_size_type_without_bs=self._dy_plugin_input_size_type_without_bs,
                dy_plugin_output_size_type_without_bs=self._dy_plugin_output_size_type_without_bs
            )
        )

    def get_shape_size(self, shape_dims):
        shape_size_list = []
        for dims in shapes_dims:
            shape_size = 1
            for dim in dims:
                shape_size = shape_size * dim
            shape_size_list.append(shape_size)
        return shape_size_list

    def generate_source_file(self):
        template = self._template_env.get_template(self._template_source_file)
        output_text = template.render(
            plugin_name=self._plugin_name,
            plugin_tensor_input_index=self._plugin_tensor_input_index,
            cases=self._plugin_template_list,
            plugin_input_dy_dim=self._batch_dim_in_inputs[self._plugin_tensor_input_index[0]],
            output_dim_shape_without_bs=self._output_dim_shape_without_bs,
            input_dim_shape_without_bs=self._input_dim_shape_without_bs
        )
        with open("./trt_plugin/src/{}.cu".format(self._plugin_name), "w") as f:
            f.write(output_text)    

    def generate_header_file(self):
        template = self._template_env.get_template(self._template_header_file)
        output_text = template.render(
            plugin_name=self._plugin_name,
            plugin_output_number=self._plugin_output_number,
            plugin_output_shape=self._dy_plugin_output_shape,
            plugin_output_type=self._plugin_output_type,
            plugin_workspace_size=self._plugin_workspace_size,
            plugin_tensor_format=self._plugin_tensor_format,
            plugin_input_size_type=self._dy_plugin_input_size_type,
            plugin_output_size_type=self._dy_plugin_output_size_type,
            plugin_output_workspace_size_type=self._dy_plugin_output_workspace_size_type,
            plugin_tensor_input_index=self._plugin_tensor_input_index,
            plugin_input_dy_dim=self._batch_dim_in_inputs[self._plugin_tensor_input_index[0]]
        )
        with open("./trt_plugin/src/{}.h".format(self._plugin_name), "w") as f:
            f.write(output_text)
