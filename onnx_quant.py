import numpy as np
import onnxruntime as onnxrt
import onnx
import math
import io
import onnx.helper as helper

def fake_quantize(x, scale):
    return (x / scale.clip(1e-7)).round().clip(-127, 127) * scale

class Quantizer(object):
    def __init__(self):
        super().__init__()
        self.amax = None
        self.collect_data = None

    def __repr__(self) -> str:
        return f"quant={self.do_quant}, collect={self.do_collect}"
        
    def post_compute_amax(self):
        if self.collect_data is None:
            return
        
        amax, bin_width, hist = self.collect_data
        number_of_bins = len(hist)
        centers = np.linspace(bin_width*0.5, amax-bin_width*0.5, number_of_bins, dtype=np.float32)
        number_of_centers = len(centers)
        centers = centers.reshape(1, number_of_centers)
        candidates = centers
        number_of_candidates = candidates.shape[1]
        candidates = candidates.reshape(number_of_candidates, 1)
        scales = candidates / 127
        reproject = fake_quantize(centers, scales)
        differences = ((reproject-centers) **2 * hist).sum(1)
        selected_index = differences.argmin()
        optimal_amax = candidates[selected_index, 0]
        self.scale = (optimal_amax / 127).clip(1e-7).astype(np.float32).reshape(1)
        self.collect_data = None
    
    def forward(self, x):
        if self.collect_data is None:
            number_of_bins = 2048
            absx = np.abs(x.astype(np.float32))
            amax = absx.max().item()
            bin_width = amax / number_of_bins
            hist = np.histogram(absx, bins=number_of_bins, range=(0, amax))[0]
            self.collect_data = amax, bin_width, hist
        else:
            prev_amax, bin_width, prev_hist = self.collect_data
            absx = np.abs(x.astype(np.float32))

            # round amax
            amax = int((max(prev_amax, absx.max().item()) + bin_width) / bin_width) * bin_width
            number_of_bins = math.ceil(amax / bin_width)
            hist = np.histogram(absx, bins=number_of_bins, range=(0, amax))[0]
            hist[:len(prev_hist)] += prev_hist
            self.collect_data = amax, bin_width, hist
        return x

class Node(object):
    def __init__(self, obj, tensors, output_indexs, inode):
        self.name = obj.name
        self.op_type = obj.op_type
        self.obj = obj
        self.tensors = tensors
        self.output_indexs = output_indexs
        self.inode = inode
        self.qs = [Quantizer() for i in tensors]
        self.disable = False

    def __repr__(self) -> str:
        return f"{{Node({self.name}, {self.op_type}): {self.tensors}}}"
    

class Tensor(object):
    def __init__(self, name, parent):
        self.name = name
        self.next = []
        self.parent = parent
        self.friends = []

    def __repr__(self):
        return f"{{Tensor({self.name}): friends{[item.name for item in self.friends]}, next{[item.name for item in self.next]}}}"

class CollectedModel(object):
    def __init__(self, file):
        self.file = file
        self.onnxmodel = onnx.load_model(file)
        self.input_tensors = [item.name for item in self.onnxmodel.graph.input]
        self.old_num_outputs = len(self.onnxmodel.graph.output)
        self.mark_all_param_layers()
        self.link_frient_quantizers()
        self.create_ort_runtime()

    def mark_all_param_layers(self):
        keeped_tensors = []
        keeped_nodes   = []
        initialize_names = set([item.name for item in self.onnxmodel.graph.initializer])
        constant_names = set([item.output[0] for item in self.onnxmodel.graph.node if item.op_type == "Constant"])
        self.ignore_tensor_names = initialize_names | constant_names

        for inode, node in enumerate(self.onnxmodel.graph.node):
            if node.op_type in ["Conv", "Concat", "Add", "Resize"]:
                select_tensors = [tensor_name for tensor_name in node.input if tensor_name not in self.ignore_tensor_names and tensor_name != ""]

                base = len(keeped_tensors)
                select_output_indexs = [base + i for i in range(len(select_tensors))]
                keeped_tensors.extend(select_tensors)
                keeped_nodes.append(Node(node, select_tensors, select_output_indexs, inode))
        
        self.onnxmodel.graph.output.extend(
            [onnx.ValueInfoProto(name=tensor_name) for tensor_name in keeped_tensors]
        )

        self.keeped_tensors = keeped_tensors
        self.keeped_nodes   = keeped_nodes
        self.keeped_node_map = {node.name: node for node in keeped_nodes}

    def link_frient_quantizers(self):
        graph = dict()
        ignore_tensor_names = self.ignore_tensor_names
        model = self.onnxmodel
        for node in model.graph.node:
            if node.op_type == "Constant":
                continue
            
            for item in node.output:
                graph[item] = Tensor(item, node)

            for item in node.input:
                if item in ignore_tensor_names or item == "":
                    continue
                
                if item not in graph:
                    graph[item] = Tensor(item, None)

                graph[item].next.append(node)

        for tname in graph:
            tensor = graph[tname]
            parent = tensor.parent

            if parent is not None and parent.op_type in ["Concat", "Resize"]:
                tensor.friends.extend([graph[i] for i in parent.input if i != tname and i in graph])

            for item in tensor.next:
                if item.op_type in ["Concat", "Add"]:
                    tensor.friends.extend([graph[i] for i in item.input if i != tname and i in graph])

            flags = set()
            all_friends = set([tensor])
            params = tensor.friends.copy()
            while len(params) > 0:
                f = params.pop()
                if f.name in flags:
                    continue
                    
                flags.add(f.name)
                params.extend(f.friends)
                all_friends.add(f)

            for f in all_friends:
                f.friends = list(all_friends)

        self.graph = graph
        enable_nodes = {key: item for key, item in self.keeped_node_map.items() if not item.disable}
        for tname in graph:
            tensor = graph[tname]
            if len(tensor.friends) < 2:
                continue
            
            if len(tensor.next) == 0 or tensor.next[0].name not in enable_nodes:
                continue

            pairs = [tname]
            major = enable_nodes[tensor.next[0].name]
            for node in tensor.next[1:]:
                if node.name in enable_nodes:
                    fnext = enable_nodes[node.name]
                    idxf = fnext.tensors.index(tname)
                    fnext.qs[idxf] = major.qs[major.tensors.index(tname)]

            if major is not None:
                for f in tensor.friends:
                    if len(f.next) == 0 or f.next[0].name not in enable_nodes or f.name == tname:
                        continue

                    fnext = enable_nodes[f.next[0].name]
                    idxf = fnext.tensors.index(f.name)
                    fnext.qs[idxf] = major.qs[major.tensors.index(tname)]
                    pairs.append(f.name)
            # print(f"Connect: {pairs}")

    def create_ort_runtime(self):
        bio = io.BytesIO()
        onnx.save_model(self.onnxmodel, bio)
        bio.seek(0)
        self.model = onnxrt.InferenceSession(bio.read())

        for i in range(len(self.onnxmodel.graph.output) - 1, self.old_num_outputs - 1, -1):
            del self.onnxmodel.graph.output[i]

    def collect(self, *inputs):
        outputs = self.model.run(self.keeped_tensors, {self.input_tensors[i]:inputs[i] for i in range(len(inputs))})

        for node in self.keeped_nodes:
            if node.disable:
                continue

            for output_index, q in zip(node.output_indexs, node.qs):
                q.forward(outputs[output_index])

    def end_collect(self):
        for node in self.keeped_nodes:
            if node.disable:
                continue

            for q in node.qs:
                q.post_compute_amax()

    def save(self, file):
        onnx.save_model(self.onnxmodel, file)

    def disable(self, *ops):
        for n in self.keeped_nodes:
            if n.name in ops:
                n.disable = True

    def disable_afters(self, *tensors):

        flags = set()
        params = list(tensors)
        while len(params) > 0:
            tname = params[0]
            del params[0]

            if tname in flags:
                continue

            flags.add(tname)
            tensor = self.graph[tname]
            for node in tensor.next:
                if node.name in self.keeped_node_map:
                    print(f"Disable: {node.name}")
                    self.keeped_node_map[node.name].disable = True
                params.extend([item for item in node.output if item not in self.ignore_tensor_names])

    def make_qdq(self, input, i_qdq_node, scale):
        zeros = np.zeros_like(scale).astype(np.int8)
        qname = f"QuantizeLinear_{i_qdq_node}"
        y_scale = helper.make_tensor(f"QuantizeLinear_y_scale_{i_qdq_node}", onnx.TensorProto.DataType.FLOAT, [len(scale)], scale)
        y_zero_point = helper.make_tensor(f"QuantizeLinear_y_zero_point_{i_qdq_node}", onnx.TensorProto.DataType.INT8, [len(scale)], zeros)
        qnode = helper.make_node("QuantizeLinear", [input, y_scale.name, y_zero_point.name], [qname], qname)

        dqname = f"DequantizeLinear_{i_qdq_node}"
        x_scale = helper.make_tensor(f"QuantizeLinear_x_scale_{i_qdq_node}", onnx.TensorProto.DataType.FLOAT, [len(scale)], scale)
        x_zero_point = helper.make_tensor(f"QuantizeLinear_x_zero_point_{i_qdq_node}", onnx.TensorProto.DataType.INT8, [len(scale)], zeros)
        dqnode = helper.make_node("DequantizeLinear", [qname, x_scale.name, x_zero_point.name], [dqname], dqname)

        if len(scale) == 1:
            del y_scale.dims[0]
            del y_zero_point.dims[0]
            del x_scale.dims[0]
            del x_zero_point.dims[0]
        return [qnode, dqnode], [y_scale, y_zero_point, x_scale, x_zero_point]

    def add_qdq(self):
        nodes = []
        initializers = []
        i_qdq_node = 0
        initializers_names = [item.name for item in self.onnxmodel.graph.initializer]
        for node in self.keeped_nodes:
            if node.disable:
                continue

            if node.op_type == "Conv":
                weight_name = node.obj.input[1]
                iweight   = initializers_names.index(weight_name)
                weight    = self.onnxmodel.graph.initializer[iweight]
                np_weight = np.frombuffer(weight.raw_data, dtype=np.float32).reshape(*weight.dims)
                scale = (np.abs(np_weight).reshape(np_weight.shape[0], -1).max(1) / 127).clip(1e-7).astype(np.float32)

                i_qdq_node += 1
                qdq, scales = self.make_qdq(weight_name, i_qdq_node, scale)
                node.obj.input[1] = qdq[1].name
                nodes.append([qdq, node.inode])
                initializers.extend(scales)

            for i, q in enumerate(node.qs):
                i_qdq_node += 1
                qdq, scales = self.make_qdq(node.tensors[i], i_qdq_node, q.scale)

                iinput = list(node.obj.input).index(node.tensors[i])
                node.obj.input[iinput] = qdq[1].name
                nodes.append([qdq, node.inode])
                initializers.extend(scales)

        nodes = sorted(nodes, key=lambda x:x[1], reverse=True)
        self.onnxmodel.graph.initializer.extend(initializers)

        # topsort
        for (q, dq), inode in nodes:
            self.onnxmodel.graph.node.insert(inode, q)
            self.onnxmodel.graph.node.insert(inode + 1, dq)


class quantonnx(object):
    def __init__(self, model, save):
        super().__init__()
        print(f"Load onnx from: {model}")
        self.save = save
        self.model = CollectedModel(model)

    def disable(self, *ops):
        self.model.disable(*ops)

    def disable_afters(self, *tensors):
        self.model.disable_afters(*tensors)

    def collect(self, *inputs):
        self.model.collect(*inputs)
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_value is not None:
            raise exc_value
        
        self.model.end_collect()
        self.model.add_qdq()
        self.model.save(self.save)
        print(f"Save int8 onnx to {self.save}")
