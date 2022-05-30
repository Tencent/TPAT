##############################
# author : qianqiu
# email : qianqiu@tencent.com
# time : 2022.1.7
##############################
import argparse
import json
import onnx
import os
import onnx_graphsurgeon as gs
from cuda_kernel import CudaKernel
from plugin_template_params import PluginTemplateParams
from plugin_template import StaticBatchPluginTemplate, DynamicBatchPluginTemplate
from onnx_modified import OnnxModified


def generate_plugin_library(input_model_path, nodes, plugin_name_dict=None, dynamic_bs=False, tune_bs_list=None, naive_onnx_model_path=None):
    onnx_name_mapping_trt_plugin = {}
    trt_plugin_mapping_onnx_node = {}

    for node in nodes:
        tuning_name = node.name
        if plugin_name_dict is not None and tuning_name in plugin_name_dict.keys():
            plugin_name = plugin_name_dict[tuning_name]
        else:
            plugin_name = "tpat_" + str(node.name).replace('/', '_')
        assert (
            node.op != plugin_name
        ), "Please make sure your plugin name is different from op type in TensorRT, \
            otherwise, the native kernel of tensorrt will be preferred for execution."
        if dynamic_bs == False:
            cuda_kernel = CudaKernel(input_model_path, node, plugin_name)
            reusable_plugin = cuda_kernel.check_existing_plugins(
                trt_plugin_mapping_onnx_node
            )
            if reusable_plugin is None:
                print(
                    "Couldn't find reusable plugin for node {}\nStart auto-tuning!".format(
                        cuda_kernel.tuning_name
                    )
                )
                cuda_kernel.run()
                template_params = PluginTemplateParams(
                    cuda_kernel, input_model_path, tuning_name
                )
                plugin_template = StaticBatchPluginTemplate(template_params)
                plugin_template.fill()
                onnx_name_mapping_trt_plugin[
                    cuda_kernel.tuning_name
                ] = template_params.plugin_name
                trt_plugin_mapping_onnx_node[
                    template_params.plugin_name
                ] = cuda_kernel._tuning_node
            else:
                print(
                    "Find existing plugin {} which could be reused for node {}".format(
                        reusable_plugin, cuda_kernel.tuning_name
                    )
                )
                onnx_name_mapping_trt_plugin[cuda_kernel.tuning_name] = reusable_plugin
        else:
            assert (
                dynamic_bs and isinstance(input_model_path, list)
            ), "[Debug] Input model path should be a list"

            template_params_list = []
            for i, explicit_bs_input_model_path in enumerate(input_model_path):
                cuda_kernel = CudaKernel(explicit_bs_input_model_path, node, plugin_name)
                resable_plugin = cuda_kernel.check_existing_plugins(
                    trt_plugin_mapping_onnx_node
                )
                if resable_plugin is not None:
                    print(
                        "[Dynamic Batch]Find existing plugin {} which could be reused for node {}".format(
                            reusable_plugin, cuda_kernel.tuning_name
                        )
                    )
                    onnx_name_mapping_trt_plugin[cuda_kernel.tuning_name] = reusable_plugin
                    continue
                print(
                    "[Dynamic Batch] Couldn't find reusable plugin for node {}\nStart auto-tuning!".format(
                        cuda_kernel.tuning_name
                    )
                )
                cuda_kernel.run()
                template_params_list.append(PluginTemplateParams(
                    cuda_kernel, explicit_bs_input_model_path, tuning_name
                ))
            dynamic_template_params = DynamicBatchPluginTemplate(template_params_list[0], naive_onnx_model_path)
            for i, template_params in enumerate(template_params_list):
                dynamic_template_params.push_plugin_template(tune_bs_list[i], StaticBatchPluginTemplate(template_params))
            dynamic_template_params.fill()
            onnx_name_mapping_trt_plugin[
                cuda_kernel.tuning_name
            ] = dynamic_template_params.plugin_name
            trt_plugin_mapping_onnx_node[
                dynamic_template_params.plugin_name
            ] = cuda_kernel._tuning_node
    return onnx_name_mapping_trt_plugin


            
def add_explicit_bs(model, explicit_bs, ):
    inputs = model.graph.input
    for input in inputs:
        dy_dims = input.type.tensor_type.shape.dim
        for dy_dim in dy_dims:
            if dy_dim.dim_value == 0:
                dy_dim.dim_value = explicit_bs
    output_file_path = 'dynamicBatch_{}.onnx'.format(explicit_bs)
    onnx.save(model, output_file_path)
    return output_file_path

def convert_node_weights(input_model_path, tuning_nodes):
    tuning_node_names = [node.name for node in tuning_nodes]
    graph = gs.import_onnx(onnx.load(input_model_path))
    new_tuning_nodes = [node for node in graph.nodes if node.name in tuning_node_names]
    tensors = graph.tensors()
    for tuning_node in new_tuning_nodes:
        for i, inp in enumerate(tuning_node.inputs):
            if isinstance(inp, gs.ir.tensor.Constant):
                const_input = tensors[inp.name]
                print("const_input: ", const_input, "\nvalues: ",const_input.values)
                if const_input.values.size < 10:
                    continue
                print("Warning: the initializer input will be converted to Constant node due to its large size")
                const_node = gs.Node(
                                    op="Constant",
                                    inputs=[],
                                    name="inserted_const_for_" + const_input.name.split(":")[0],
                                    attrs={"value": gs.ir.tensor.Constant(const_input.name, const_input.values)},
                                )  # INT32
                const_node_out = gs.Variable(const_node.name + ":0")
                const_node.outputs = [const_node_out]
                tuning_node.inputs[i] = const_node_out
                const_input.outputs.clear()
                print("inserted const node: ", const_node)
                graph.nodes.append(const_node)
                # graph.cleanup()
    output_file_path = input_model_path.replace('.onnx', '_convertedWeights.onnx')
    onnx.save(gs.export_onnx(graph), output_file_path)
    return output_file_path, new_tuning_nodes

def onnx2plugin(
    input_model_path,
    output_model_path,
    node_names=None,
    node_types=None,
    plugin_name_dict=None,
    dynamic_bs=False,
    min_bs=None,
    max_bs=None,
    opt_bs=None
):
    assert (
        node_names is not None or node_types is not None or plugin_name_dict is not None
    ), "Please input at least one of node name、node type and dict of plugin"
    
    assert (
        dynamic_bs is not True or (dynamic_bs is True and not (min_bs is None or max_bs is None or opt_bs is None))
    ), "Dynamic Batch Size need to input of minium batch size, maxium batch size and optimize batch size"

    input_onnx_model = onnx.load(input_model_path)
    input_model = gs.import_onnx(input_onnx_model)
    nodes = []
    if node_names is not None:
        for node_name in node_names:
            nodes.extend([node for node in input_model.nodes if node.name == node_name])
    if node_types is not None:
        for node_type in node_types:
            nodes.extend([node for node in input_model.nodes if node.op == node_type])
    if plugin_name_dict is not None:
        for one_plugin_name in plugin_name_dict.keys():
            nodes.extend(
                [node for node in input_model.nodes if node.name == one_plugin_name]
            )
    assert (
        len(nodes) != 0
    ), "Not get tuning node in onnx model, please check op name or onnx model"

    input_model_path, nodes = convert_node_weights(input_model_path, nodes)

    if dynamic_bs == True:
        dy_input_model_path = []
        tune_bs_list = [min_bs, opt_bs, max_bs]
        for cur_bs in tune_bs_list:
            input_onnx_model = onnx.load(input_model_path)
            explicit_bs_onnx_file = add_explicit_bs(input_onnx_model, cur_bs)
            dy_input_model_path.append(explicit_bs_onnx_file)    
        onnx_name_mapping_trt_plugin = generate_plugin_library(
            dy_input_model_path, nodes, plugin_name_dict, dynamic_bs, tune_bs_list, input_model_path
        )
        for dy_input_model in dy_input_model_path:
            os.remove(dy_input_model)
    else:
        onnx_name_mapping_trt_plugin = generate_plugin_library(
            input_model_path, nodes, plugin_name_dict 
        )
    print("Onnx_name_mapping_trt_plugin: {}".format(onnx_name_mapping_trt_plugin))
    OnnxModified(
        input_model_path, output_model_path, nodes, onnx_name_mapping_trt_plugin
    )
    return onnx_name_mapping_trt_plugin.values()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_model_path",
        type=str,
        required=True,
        help="Please provide input onnx model path",
    )
    parser.add_argument(
        "-o",
        "--output_model_path",
        type=str,
        required=True,
        help="Please provide output onnx model path which used for tensorrt",
    )
    parser.add_argument(
        "-n",
        "--node_names",
        type=str,
        nargs="*",
        help="Please provide the operator name that needed to generate tensorrt-plugin",
    )
    parser.add_argument(
        "-t",
        "--node_types",
        type=str,
        nargs="*",
        help="Please provide the operator type that needed to generate tensorrt-plugin",
    )
    parser.add_argument(
        "-p",
        "--plugin_name_dict",
        type=str,
        help='Please provide the dict of op name and plugin name that \
            will be generated by TPAT, such as : {"op_name" : "plugin_name"}',
    )
    parser.add_argument(
        "-dynamic",
        "--dynamic_batchsize",
        type=str,
        help='',
    )
    parser.add_argument(
        "-min",
        "--min_batchsize",
        type=int,
        help='',
    )
    parser.add_argument(
        "-max",
        "--max_batchsize",
        type=int,
        help='',
    )
    parser.add_argument(
        "-opt",
        "--optimize_batchsize",
        type=int,
        help='',
    )


    args = parser.parse_args()
    input_model_path = args.input_model_path
    output_model_path = args.output_model_path
    node_names, node_types, plugin_name_dict = None, None, None
    dynamic_bs, min_bs, max_bs, opt_bs = False, None, None, None
    if args.node_names:
        node_names = args.node_names
    if args.node_types:
        node_types = args.node_types
    if args.plugin_name_dict:
        plugin_name_dict = json.loads(args.plugin_name_dict)
    if args.dynamic_batchsize:
        dynamic_bs = bool(args.dynamic_batchsize)
    if args.min_batchsize:
        min_bs = int(args.min_batchsize)
    if args.max_batchsize:
        max_bs = int(args.max_batchsize)
    if args.optimize_batchsize:
        opt_bs = int(args.optimize_batchsize)
    onnx2plugin(
        input_model_path,
        output_model_path,
        node_names=node_names,
        node_types=node_types,
        plugin_name_dict=plugin_name_dict,
        dynamic_bs=dynamic_bs,
        min_bs=min_bs,
        max_bs=max_bs,
        opt_bs=opt_bs
    )
