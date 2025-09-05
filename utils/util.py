from collections import namedtuple
from io import BytesIO

import cv2
import onnx
import onnxsim
import tensorrt
import torch

tensorrt_version = tensorrt.__version__
major_version = int(tensorrt_version.split('.')[0])
minor_version = int(tensorrt_version.split('.')[1])
total_memory = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory


def resize(x, h, w):
    # Resize and pad image while meeting stride-multiple constraints
    shape = x.shape[:2]  # current shape [height, width]

    # Scale ratio (new / old)
    r = min(w / shape[1], h / shape[0])
    # Compute padding [width, height]
    pad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = w - pad[0], h - pad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != pad:  # resize
        x = cv2.resize(x, pad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    x = cv2.copyMakeBorder(x,
                           top,
                           bottom,
                           left,
                           right,
                           cv2.BORDER_CONSTANT)  # add border
    return x, r, (dw, dh)


class EngineBuilder:
    def __init__(self, onnx_path, engine_path, device):
        self.device = device
        self.onnx_path = onnx_path
        self.engine_path = engine_path

    def build(self, fp16=True, with_profiling=True):
        logger = tensorrt.Logger(tensorrt.Logger.WARNING)
        tensorrt.init_libnvinfer_plugins(logger, namespace='')
        builder = tensorrt.Builder(logger)
        config = builder.create_builder_config()

        if major_version >= 10:
            config.set_memory_pool_limit(tensorrt.MemoryPoolType.WORKSPACE, total_memory // 2)
            config.set_memory_pool_limit(tensorrt.MemoryPoolType.DLA_MANAGED_SRAM, total_memory // 4)
        elif major_version >= 8:
            config.set_memory_pool_limit(tensorrt.MemoryPoolType.WORKSPACE, total_memory)
        else:
            config.max_workspace_size = total_memory

        flag = (1 << int(tensorrt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        network = builder.create_network(flag)

        self.logger = logger
        self.builder = builder
        self.network = network

        self.build_from_onnx()

        if fp16 and self.builder.platform_has_fast_fp16:
            config.set_flag(tensorrt.BuilderFlag.FP16)

        if with_profiling:
            if major_version <= 8:
                config.profiling_verbosity = tensorrt.ProfilingVerbosity.VERBOSE
            else:
                config.profiling_verbosity = tensorrt.ProfilingVerbosity.DETAILED

        if major_version >= 8:
            serialized_engine = builder.build_serialized_network(
                network, config)
            if serialized_engine is None:
                raise RuntimeError('Failed to build serialized engine')
            with tensorrt.Runtime(logger) as runtime:
                engine = runtime.deserialize_cuda_engine(serialized_engine)
        else:
            engine = builder.build_engine(network, config)
            if engine is None:
                raise RuntimeError('Failed to build engine')

        if engine is not None:
            with open(self.engine_path, 'wb') as f:
                f.write(engine.serialize())
            self.logger.log(
                tensorrt.Logger.WARNING, f'Build TensorRT engine finished.\n'
                                         f'Saved to {str(self.engine_path)}')
        else:
            raise RuntimeError('Engine creation failed')

    def build_from_onnx(self):
        parser = tensorrt.OnnxParser(self.network, self.logger)
        onnx_model = onnx.load(str(self.onnx_path))

        if not parser.parse(onnx_model.SerializeToString()):
            raise RuntimeError(f'Failed to load ONNX file: {str(self.onnx_path)}')
        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]

        for inp in inputs:
            self.logger.log(tensorrt.Logger.WARNING,
                            f'input "{inp.name}" with shape: {inp.shape} dtype: {inp.dtype}')
        for out in outputs:
            self.logger.log(tensorrt.Logger.WARNING,
                            f'output "{out.name}" with shape: {out.shape} dtype: {out.dtype}')


class TRTModule(torch.nn.Module):
    dtypeMapping = {tensorrt.bool: torch.bool,
                    tensorrt.int8: torch.int8,
                    tensorrt.int32: torch.int32,
                    tensorrt.float16: torch.float16,
                    tensorrt.float32: torch.float32}

    def __init__(self, engine_path, device):
        super().__init__()
        self.weight = engine_path
        self.device = device if device is not None else torch.device('cuda:0')
        self.stream = torch.cuda.Stream(device=self.device)
        self.__init_engine()
        self.__init_bindings()

    def __init_engine(self) -> None:
        logger = tensorrt.Logger(tensorrt.Logger.WARNING)
        tensorrt.init_libnvinfer_plugins(logger, namespace='')
        with tensorrt.Runtime(logger) as runtime:
            with open(self.weight, 'rb') as f:
                model = runtime.deserialize_cuda_engine(f.read())

        context = model.create_execution_context()

        if major_version >= 10:
            num_io_tensors = model.num_io_tensors
            names = [model.get_tensor_name(i) for i in range(num_io_tensors)]
            num_inputs = sum(1 for name in names if model.get_tensor_mode(name) == tensorrt.TensorIOMode.INPUT)
            num_outputs = num_io_tensors - num_inputs
        else:
            num_bindings = model.num_bindings
            names = [model.get_binding_name(i) for i in range(num_bindings)]
            num_inputs = sum(1 for i in range(num_bindings) if model.binding_is_input(i))
            num_outputs = num_bindings - num_inputs

        self.bindings = [0] * (num_inputs + num_outputs)
        self.num_bindings = num_inputs + num_outputs
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.model = model
        self.context = context
        self.input_names = names[:num_inputs]
        self.output_names = names[num_inputs:]
        self.idx = list(range(self.num_outputs))

    def __init_bindings(self) -> None:
        i_dynamic = o_dynamic = False
        Tensor = namedtuple('Tensor', ('name', 'dtype', 'shape'))
        inp_info = []
        out_info = []
        for i, name in enumerate(self.input_names):
            if major_version >= 10:
                dtype = self.dtypeMapping[self.model.get_tensor_dtype(name)]
                shape = tuple(self.model.get_tensor_shape(name))
            else:
                assert self.model.get_binding_name(i) == name
                dtype = self.dtypeMapping[self.model.get_binding_dtype(i)]
                shape = tuple(self.model.get_binding_shape(i))
            if -1 in shape:
                i_dynamic |= True
            inp_info.append(Tensor(name, dtype, shape))
        for i, name in enumerate(self.output_names):
            j = i + self.num_inputs
            if major_version >= 10:
                dtype = self.dtypeMapping[self.model.get_tensor_dtype(name)]
                shape = tuple(self.model.get_tensor_shape(name))
            else:
                assert self.model.get_binding_name(j) == name
                dtype = self.dtypeMapping[self.model.get_binding_dtype(j)]
                shape = tuple(self.model.get_binding_shape(j))
            if -1 in shape:
                o_dynamic |= True
            out_info.append(Tensor(name, dtype, shape))

        if not o_dynamic:
            self.output_tensor = [torch.empty(info.shape, dtype=info.dtype, device=self.device)
                                  for info in out_info]
        self.i_dynamic = i_dynamic
        self.o_dynamic = o_dynamic
        self.inp_info = inp_info
        self.out_info = out_info

    def set_profiler(self, profiler):
        self.context.profiler = profiler if profiler is not None else \
            tensorrt.Profiler()

    def set_desired(self, desired):
        if isinstance(desired, (list, tuple)) and len(desired) == self.num_outputs:
            self.idx = [self.output_names.index(i) for i in desired]

    def forward(self, *inputs):
        assert len(inputs) == self.num_inputs
        contiguous_inputs = [i.contiguous() for i in inputs]

        if major_version >= 10:
            for i, name in enumerate(self.input_names):
                self.context.set_tensor_address(name, contiguous_inputs[i].data_ptr())
                if self.i_dynamic:
                    self.context.set_input_shape(name, tuple(contiguous_inputs[i].shape))

            outputs = []
            for i, name in enumerate(self.output_names):
                if self.o_dynamic:
                    shape = tuple(self.context.get_tensor_shape(name))
                    output = torch.empty(size=shape,
                                         dtype=self.out_info[i].dtype,
                                         device=self.device)
                else:
                    output = self.output_tensor[i]
                self.context.set_tensor_address(name, output.data_ptr())
                outputs.append(output)

            success = self.context.execute_async_v3(self.stream.cuda_stream)
            if not success:
                raise RuntimeError('TensorRT execution failed')
        else:
            for i in range(self.num_inputs):
                self.bindings[i] = contiguous_inputs[i].data_ptr()
                if self.i_dynamic:
                    self.context.set_binding_shape(i, tuple(contiguous_inputs[i].shape))

            outputs = []
            for i in range(self.num_outputs):
                j = i + self.num_inputs
                if self.o_dynamic:
                    shape = tuple(self.context.get_binding_shape(j))
                    output = torch.empty(size=shape,
                                         dtype=self.out_info[i].dtype,
                                         device=self.device)
                else:
                    output = self.output_tensor[i]
                self.bindings[j] = output.data_ptr()
                outputs.append(output)

            self.context.execute_async_v2(self.bindings,
                                          self.stream.cuda_stream)

        self.stream.synchronize()

        return tuple(outputs[i] for i in self.idx) if len(outputs) > 1 else outputs[0]


def to_onnx(device, torch_path, onnx_path):
    b = 1
    size = 640

    model = torch.load(f=torch_path, weights_only=False, map_location='cuda')
    model = model['model'].float().fuse().export()
    model.to(device)

    x = torch.randn((b, 3, size, size)).to(device)
    for _ in range(2):
        model(x)
    with BytesIO() as f:
        torch.onnx.export(model,
                          x,
                          f,
                          opset_version=11,
                          input_names=['images'],
                          output_names=['num', 'boxes', 'scores', 'labels'])
        f.seek(0)
        onnx_model = onnx.load(f)
    onnx.checker.check_model(onnx_model)

    try:
        onnx_model, check = onnxsim.simplify(onnx_model)
        assert check, 'assert check failed'
    except Exception as e:
        print(f'Simplifier failure: {e}')

    onnx.save(onnx_model, onnx_path)
    print(f'ONNX export success, saved as {onnx_path}')


def to_engine(device, onnx_path, engine_path):
    builder = EngineBuilder(onnx_path, engine_path, device)
    builder.build(fp16=True)


def make_anchors(x, strides, offset=0.5):
    assert x is not None
    anchor_tensor, stride_tensor = [], []
    dtype, device = x[0].dtype, x[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = x[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + offset  # shift y
        sy, sx = torch.meshgrid(sy, sx)
        anchor_tensor.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_tensor), torch.cat(stride_tensor)
