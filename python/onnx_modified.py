##############################
# author : qianqiu
# email : qianqiu@tencent.com
# time : 2022.1.7
##############################
import onnx
import onnx_graphsurgeon as gs
from onnx import shape_inference
from type_mapping import onnx_type_mapping


class OnnxModified(object):
    """
    Insert cast operator for operators which inputs or outputs has bool type.
    Modify operator type in onnx model for tensorRT can run plugin.
    """

    def __init__(
        self,
        input_model_path,
        output_model_path,
        tuning_nodes,
        onnx_layer_name_mapping_trt_plugin_name,
    ):
        self._graph = gs.import_onnx(onnx.load(input_model_path))
        self._onnx_original_tensor_type = self.compute_tensor_type(
            input_model_path, tuning_nodes
        )
        self.handle_trt_not_support_type(
            input_model_path,
            output_model_path,
            onnx_layer_name_mapping_trt_plugin_name,
            self._onnx_original_tensor_type,
        )

    def handle_trt_not_support_type(
        self,
        input_model_path,
        output_model_path,
        onnx_layer_name_mapping_trt_plugin_name,
        onnx_original_tensor_type,
    ):
        count = 0
        insert_cast_nodes = False
        for node in self._graph.nodes:
            if node.name in onnx_layer_name_mapping_trt_plugin_name:
                node.op = onnx_layer_name_mapping_trt_plugin_name[node.name]
                # print("ONNX node for modifying: {}".format(node))
                for i, inp in enumerate(node.inputs):
                    if inp.is_empty():
                        node.inputs.remove(inp)
                        print("remove empty input tensor: ", node)
                        self._graph.cleanup()
                        continue
                    if onnx_original_tensor_type[inp.name] in onnx_type_mapping:
                        cast_node = gs.Node(
                            op="Cast",
                            name="cast_to_int32_for_" + inp.name.split(":")[0],
                            attrs={"to": 6},
                        )  # INT32
                        cast_node.inputs = [inp]
                        cast_node_out = gs.Variable(cast_node.name + ":0")
                        cast_node.outputs = [cast_node_out]
                        node.inputs[i] = cast_node_out
                        self._graph.nodes.append(cast_node)
                        self._graph.cleanup()
                        insert_cast_nodes = True
                for i, oup in enumerate(node.outputs):
                    if onnx_original_tensor_type[oup.name] in onnx_type_mapping:
                        dtype = onnx_type_mapping[onnx_original_tensor_type[oup.name]]
                        cast_node = gs.Node(
                            op="Cast",
                            name="cast_back_for_" + oup.name.split(":")[0],
                            attrs={"to": dtype},
                        )
                        cast_node.outputs = [oup]
                        cast_node_out = gs.Variable(cast_node.name + ":0")
                        cast_node.inputs = [cast_node_out]
                        node.outputs[i] = cast_node_out
                        self._graph.nodes.append(cast_node)
                        self._graph.cleanup()
                        insert_cast_nodes = True
                # print("Graph after inserting Cast nodes", self._graph)
                count = count + 1
        assert count == len(onnx_layer_name_mapping_trt_plugin_name)
        if insert_cast_nodes:
            self.remove_unnecessary_cast_nodes()
        onnx.save(gs.export_onnx(self._graph), output_model_path)

    def remove_unnecessary_cast_nodes(self):
        self._graph.toposort()
        cast_nodes = [
            node
            for node in self._graph.nodes
            if (
                node.op == "Cast"
                and node.outputs[0] not in self._graph.outputs
                and node.o().op == "Cast"
            )
        ]
        for node in cast_nodes:
            if (
                node.attrs["to"] == 13
                and len(node.inputs[0].inputs) <= 1
                and len(node.outputs[0].outputs) <= 1
            ):
                # print("[WARNING] remove unnecessary cast node: ", node)
                node.o().inputs = node.inputs
                node.inputs.clear()
                self._graph.cleanup()

    def compute_tensor_type(self, input_model_path, tuning_nodes):
        model = onnx.load(input_model_path)
        inferred_model = shape_inference.infer_shapes(model)
        graph = gs.import_onnx(inferred_model)
        onnx_original_tensor_type = {}

        for tuning_node in tuning_nodes:
            # print("Tuning_node: {}\n".format(tuning_node))
            inferred_tuning_node = [
                node for node in graph.nodes if node.name == tuning_node.name
            ][0]
            for inp in inferred_tuning_node.inputs:
                if inp.__class__ == gs.Constant:
                    onnx_original_tensor_type[inp.name] = inp.dtype.__name__
                elif not inp.is_empty():
                    onnx_original_tensor_type[inp.name] = inp.dtype.name
            [
                onnx_original_tensor_type.update({oup.name: oup.dtype.name})
                for oup in inferred_tuning_node.outputs
            ]
        # print("Onnx_original_tensor_type: {}".format(onnx_original_tensor_type))
        return onnx_original_tensor_type
