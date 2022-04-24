##############################
# author : qianqiu
# email : qianqiu@tencent.com
# time : 2022.1.7
##############################
import os
import contextlib
import re
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
        onnx_output_shape = template_params.output_shape
        onnx_input_shape = template_params.input_shape
        self._plugin_output_shape = self.parse_plugin_output_shape(onnx_output_shape)
        self._plugin_input_shape = self.parse_plugin_input_shape(onnx_input_shape)
        onnx_tensor_type = template_params.tensor_type
        self._plugin_tensor_format = self.parse_plugin_tensor_format(onnx_tensor_type)
        kernel_order = template_params.kernel_order
        workspace_init = template_params.workspace_init
        self._plugin_kernels_params = self.parse_plugin_kernels_params(kernel_order)
        self._plugin_constant_init = self.parse_plugin_workspace_init(workspace_init)
        self._plugin_kernels_body = template_params.cuda_source_code
        self._onnx_input_python_type = template_params.onnx_input_python_type
        self._onnx_output_python_type = template_params.onnx_output_python_type
        self._plugin_input_size = self.get_mul_shape_size(self._plugin_input_shape)
        self._plugin_output_size = self.get_mul_shape_size(self._plugin_output_shape)
        self._plugin_input_size_type = self.get_mul_shape_size_type(self._plugin_input_size, self._onnx_input_python_type)
        self._plugin_output_size_type = self.get_mul_shape_size_type(self._plugin_output_size, self._onnx_output_python_type)

    @property
    def plugin_name(self):
        return self._plugin_name

    class TensorDims:
        def __init__(self, nbdims, shape):
            self.nbdims = nbdims
            self.shape = tuple(shape)

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
        def __init__(self, batch_size, plugin_template):
            self.batch_size = batch_size
            self.plugin_template = plugin_template

    class Shape:
        def __init__(self, size, dtype):
            self.size = size
            self.dtype = dtype

    def get_mul_shape_size(self, shape_dims):
        from functools import reduce
        from operator import mul
        dy_plugin_shape_size = []
        for dims in shape_dims:
            dy_plugin_shape_size.append(reduce(mul, dims.shape))
        return dy_plugin_shape_size

    def get_mul_shape_size_type(self, dy_plugin_shape_size, onnx_python_type):
        dynamic_shape_size_type = list()
        for i in range(len(dy_plugin_shape_size)):
            dynamic_shape_size_type.append(
                self.Shape(
                    size=dy_plugin_shape_size[i],
                    dtype=onnx_python_type[i]
                )
            )
        return dynamic_shape_size_type

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
        TEMPLATE_HEADER_FILE="./trt_plugin/trt8.0_plugin_h_dynamic.template",
        TEMPLATE_SOURCE_FILE="./trt_plugin/trt8.0_plugin_cu_dynamic.template",
    ):
        super(DynamicBatchPluginTemplate, self).__init__(template_params)
        self._template_env.filters['rm_part_define'] = rm_part_define 
        self._template_header_file = TEMPLATE_HEADER_FILE
        self._template_source_file = TEMPLATE_SOURCE_FILE
        self._plugin_template_list = list()
        self._dy_plugin_output_shape = self.get_dynamic_output_shape(self._plugin_output_shape)
        self._dy_plugin_output_size_type = list()
        self._dy_plugin_input_size_type = list()

    def get_dynamic_output_shape(self, plugin_output_shape):
        dy_plugin_output_shape = []
        for output_shape in plugin_output_shape:
            dy_plugin_output_shape.append(
                self.TensorDims(
                    nbdims=output_shape.nbdims - 1,
                    shape=output_shape.shape[1:]
                )
            )
        return dy_plugin_output_shape

    def push_plugin_template(self, batch_size, plugin_template):
        duplicate_list = set()
        for kernel in plugin_template._plugin_kernels_params:
            func_name = kernel.name
            func_name_with_bs = func_name + '_bs' + str(batch_size)
            kernel.name = func_name_with_bs
            if kernel.name in duplicate_list:
                continue
            duplicate_list.add(kernel.name)
            plugin_template._plugin_kernels_body = plugin_template._plugin_kernels_body.replace(func_name, func_name_with_bs)

        kernels_body = plugin_template._plugin_kernels_body.split('extern "C"')
        real_kernels_body = []
        real_kernels_body.append(kernels_body[0])
        for func_name in duplicate_list:
            for func_body in kernels_body:
                if func_name in func_body:
                    real_kernels_body.append('extern "C"' + func_body)
        plugin_template._plugin_kernels_body = '\n'.join(real_kernels_body)

        self._plugin_workspace_size = max(self._plugin_workspace_size, plugin_template._plugin_workspace_size)
        self._dy_plugin_input_size_type = plugin_template._plugin_input_size_type
        self._dy_plugin_output_size_type = plugin_template._plugin_output_size_type

        self._plugin_template_list.append(
            self.Case(
                batch_size,
                plugin_template
            )
        )

    def generate_source_file(self):
        template = self._template_env.get_template(self._template_source_file)
        output_text = template.render(
            plugin_name=self._plugin_name,
            cases=self._plugin_template_list,
            plugin_input_size_type=self._dy_plugin_input_size_type,
            plugin_output_size_type=self._dy_plugin_output_size_type
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
        )
        with open("./trt_plugin/src/{}.h".format(self._plugin_name), "w") as f:
            f.write(output_text)
