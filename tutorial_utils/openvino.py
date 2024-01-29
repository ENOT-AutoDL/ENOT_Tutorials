import os
import subprocess
from typing import Iterable


def benchmark(model_path: str, shape: Iterable) -> None:
    """
    Measure FPS for OpenVINO model.

    Parameters
    ----------
    model_path : str
        Path to OpenVINO model.xml.
    shape : Iterable
        Shape for OpenVINO model with dynamic inputs (B, C, H, W).

    """
    print("Start benchmark")

    shape = str(list(shape))
    shape = shape.replace(" ", "")

    result = subprocess.run(
        ["benchmark_app", "--path_to_model", model_path, "-shape", shape, "-d", "CPU"], capture_output=True, text=True
    ).stdout

    # 10 last valuable lines including speed
    for line in result.split('\n')[-10:]:
        print(line)


def convert_model(model_path: str, output_name: str) -> None:
    """
    Convert ONNX model to OpenVINO.

    Parameters
    ----------
    model_path : str
        Path to model.onnx.
    output_name : str
        Name for openvino model.

    """
    subprocess.run(
        [
            "mo",
            "--input_model",
            model_path,
            "--model_name",
            output_name,
            "--output_dir",
            output_name + '_openvino_model',
        ]
    )
