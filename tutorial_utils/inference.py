from typing import Callable
from typing import Tuple

import numpy as np
import onnxruntime
import torch
from onnx.onnx_pb import ModelProto  # pylint: disable=no-name-in-module


def create_onnxruntime_session(  # pylint: disable=missing-function-docstring
    proto: ModelProto,
    input_sample: torch.Tensor,
    output_shape: Tuple,
) -> Callable[[torch.Tensor], torch.Tensor]:
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

    sess = onnxruntime.InferenceSession(
        path_or_bytes=proto.SerializeToString(),
        sess_options=sess_options,
        providers=['TensorrtExecutionProvider'],
        provider_options=[{'trt_fp16_enable': 'True', 'trt_int8_enable': 'True'}],
    )
    sess.disable_fallback()

    bindings = sess.io_binding()
    device_type = input_sample.device.type
    device_id = input_sample.device.index
    inputs_shape = tuple(input_sample.shape)
    output_tensor = torch.empty(output_shape, dtype=torch.float32, device=input_sample.device).contiguous()

    bindings.bind_output(
        name='output',
        device_type=output_tensor.device.type,
        device_id=output_tensor.device.index,
        element_type=np.float32,
        shape=tuple(output_tensor.shape),
        buffer_ptr=output_tensor.data_ptr(),
    )

    def _sess(inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs.contiguous()
        bindings.bind_input(
            name='input',
            device_type=device_type,
            device_id=device_id,
            element_type=np.float32,
            shape=inputs_shape,
            buffer_ptr=inputs.data_ptr(),
        )
        sess.run_with_iobinding(bindings)
        return output_tensor

    _sess(input_sample)  # build engine

    return _sess
