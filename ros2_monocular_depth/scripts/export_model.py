import argparse
from pathlib import Path

import torch

import ros2_monocular_depth.config as config
from ros2_monocular_depth.models import DepthAnythingV2
import ros2_monocular_depth.transforms as transforms


def list_weights_available():
    path_dir_weights = Path(config._PATH_DIR_WEIGHTS)
    paths_weights = sorted(path_dir_weights.glob("*.pth"))
    names_weights_available = [str(path_weights.parent.relative_to(path_dir_weights) / path_weights.stem) for path_weights in paths_weights]

    return names_weights_available


def parse_args():
    names_weights_available = list_weights_available()

    parser = argparse.ArgumentParser(description="Export model.")
    parser.add_argument("--weights", dest="name_weights", help="Name of the saved weights", choices=names_weights_available, required=True)
    parser.add_argument("--width", dest="width_original", help="Width of original image", type=int, default=518)
    parser.add_argument("--height", dest="height_original", help="Height of original image", type=int, default=518)
    parser.add_argument("--min_size_resized", help="Size of the smaller dimension after resizing", type=int, default=518)
    parser.add_argument("--base", help="Base for input dimensions to be resized to a multiple of", type=int, default=14)
    parser.add_argument("--opset", dest="version_opset", help="Opset version", type=int)
    args = parser.parse_args()

    shape_original = (1, 3, args.height_original, args.width_original)

    return args.name_weights, shape_original, args.min_size_resized, args.base, args.version_opset


def export_model(name_weights, shape_original=(1, 3, 518, 518), min_size_resized=518, base=14, version_opset=None):
    print(f"Exporting model ...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path_weights = Path(config._PATH_DIR_WEIGHTS) / f"{name_weights}.pth"
    weights = torch.load(f"{path_weights}", map_location="cpu")

    model = DepthAnythingV2(**config._MODELS[name_weights]["kwargs"])
    model.load_state_dict(weights)
    model = model.to(device)
    model = model.eval()

    shape_input = transforms.resize_shape_to_multiple_of_base(shape_original, min_size_resized, base)
    input_dummy = torch.rand(shape_input, device=device, dtype=torch.float32)

    path_onnx = Path(config._PATH_DIR_ONNX) / f"{name_weights}.onnx"
    path_onnx.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        input_dummy,
        path_onnx,
        opset_version=version_opset,
        export_params=True,
        input_names=["input"],
        output_names=["output"],
    )
    # TODO: Add dynamic axes
    # dynamic_axes={
    #     "input": {2: "height", 3: "width"},
    #     "output": {1: "height", 2: "width"},
    # },
    # Alternative (currently not working):
    # torch.onnx.dynamo_export(
    #     model,
    #     input_dummy,
    #     # export_options=torch.onnx.ExportOptions(dynamic_shapes=True),
    # ).save(str(path_onnx))

    print(f"Model exported to {path_onnx}")


def main():
    name_weights, shape_original, min_size_resized, base, version_opset = parse_args()
    export_model(name_weights, shape_original, min_size_resized, base, version_opset)


if __name__ == "__main__":
    main()
