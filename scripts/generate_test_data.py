import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import os.path

ASYMMETRIC_REFLECT = "ASYMMETRIC_REFLECT"
AVOID = "AVOID"
CONSTANT_0 = "CONSTANT_0"
CONSTANT_2 = "CONSTANT_2"
REPEAT = "REPEAT"
SYMMETRIC_REFLECT = "SYMMETRIC_REFLECT"
WRAP = "WRAP"

methods = (
    ASYMMETRIC_REFLECT,
    AVOID,
    CONSTANT_0,
    CONSTANT_2,
    REPEAT,
    SYMMETRIC_REFLECT,
    WRAP
)


def prepare_input(arr, method, padding):
    padding_y, padding_x = padding
    if len(arr.shape) == 3:
        padding = ((padding_y, padding_y), (padding_x, padding_x), (0, 0))
    else:
        padding = ((padding_y, padding_y), (padding_x, padding_x))

    if method == "ASYMMETRIC_REFLECT":
        return np.pad(arr, padding, 'reflect')
    elif method == "AVOID":
        return np.array(arr)
    elif method == "CONSTANT_0":
        return np.pad(arr, padding, constant_values=0)
    elif method == "CONSTANT_2":
        return np.pad(arr, padding, constant_values=2)
    elif method == "REPEAT":
        return np.pad(arr, padding, 'edge')
    elif method == "SYMMETRIC_REFLECT":
        return np.pad(arr, padding, 'symmetric')
    elif method == "WRAP":
        return np.pad(arr, padding, 'wrap')
    else:
        raise ValueError(f"Unknown method '{method}'.")


def print_array(arr):
    def to_str(value):
        if type(value) in (float, np.float32, np.float64):
            return f"{value:>7.2f}"
        else:
            return f"{value:>3d}"

    if len(arr.shape) == 2:
        for y in range(arr.shape[0]):
            print(', '.join([to_str(_) for _ in arr[y]]))
    else:
        for y in range(arr.shape[0]):
            print(', '.join([f"{_}" for _ in arr[y]]))


def read_image(file_name, file_extension):
    image = Image.open(f'../resources/tests/{file_name}.{file_extension}')
    image = np.asarray(image)
    image = image.astype(np.float64) / np.max(image)
    if len(image.shape) == 2:
        image = image.reshape((image.shape[0], image.shape[1], 1))
    return image


def get_correct_filter(channels):
    zero = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]
    kernel = [
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]
    ]

    if channels == 1:
        return torch.tensor([
            [kernel]
        ], dtype=torch.double)
    elif channels == 3:
        return torch.tensor([
            [kernel, zero, zero],
            [zero, kernel, zero],
            [zero, zero, kernel]
        ], dtype=torch.double)
    else:
        raise RuntimeError("Nope not 4 channels!")


def convolve(image, stride, dilation):
    image = np.rollaxis(image, 2, 0)
    input_data = torch.from_numpy(image).unsqueeze(0)
    filter = get_correct_filter(input_data.shape[1])

    result = F.conv2d(input=input_data, weight=filter, padding=0, stride=stride, dilation=dilation)
    result = result.squeeze(0)
    result_data = result.numpy()

    result_data = np.rollaxis(result_data, 0, 3)
    return result_data


def normalize_after_convolution(arr):
    arr = arr - np.min(arr)
    arr = arr / np.max(arr)

    return np.round(arr * 255, 11)


def save_image(arr, file_name, method, file_extension, padding, stride, dilation, format):
    option_folder = f'{"x".join(map(str, padding))}-{"x".join(map(str, stride))}-{"x".join(map(str, dilation))}'
    file_path = f'../resources/tests/{option_folder}/{method}/{file_name}.{file_extension}'
    arr = arr.astype(np.uint8)

    if len(arr.shape) == 2:
        result_image = Image.fromarray(arr)
    else:
        result_image = Image.fromarray(arr, mode="RGB")

    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    if file_extension == "pbm":
        result_image = result_image.convert('1')
    result_image.save(file_path, format=format)


def convolve_with_method(options, method, file_extension, file_format):
    file_name = "Piercing-The-Ocean"
    image = read_image(file_name, file_extension)

    padding, stride, dilation = options

    prepared_image = prepare_input(image, method, padding)
    result_data = convolve(prepared_image, stride, dilation)

    result_data = normalize_after_convolution(result_data)
    if result_data.shape[2] == 1:
        result_data = result_data.squeeze(2)
    save_image(result_data, file_name, method, file_extension, padding, stride, dilation, file_format)


def main():
    list_of_options = (
        ((0, 0), (1, 1), (1, 1)),
        ((3, 2), (1, 1), (1, 1)),
        ((0, 0), (4, 5), (1, 1)),
        ((0, 0), (1, 1), (2, 3)),
        ((3, 2), (4, 5), (2, 3)),
    )

    extensions = (
        ("pgm", "PPM"),
        ("ppm", "PPM"),
    )

    for options in list_of_options:
        for method in methods:
            for file_extension, file_format in extensions:
                convolve_with_method(options, method, file_extension, file_format)


if __name__ == "__main__":
    main()
