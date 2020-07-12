import os.path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from itertools import product
from tqdm import tqdm

ASYMMETRIC_REFLECT = "ASYMMETRIC_REFLECT"
AVOID = "AVOID"
CONSTANT_0 = "CONSTANT_0"
CONSTANT_2 = "CONSTANT_2"
REPEAT = "REPEAT"
SYMMETRIC_REFLECT = "SYMMETRIC_REFLECT"
WRAP = "WRAP"

border_treatment_methods = (
    ASYMMETRIC_REFLECT,
    AVOID,
    CONSTANT_0,
    CONSTANT_2,
    REPEAT,
    SYMMETRIC_REFLECT,
    WRAP
)


def prepare_input(arr, border_treatment_method, padding):
    padding_y, padding_x = padding
    if len(arr.shape) == 3:
        padding = ((padding_y, padding_y), (padding_x, padding_x), (0, 0))
    else:
        padding = ((padding_y, padding_y), (padding_x, padding_x))

    if border_treatment_method == "ASYMMETRIC_REFLECT":
        return np.pad(arr, padding, 'reflect')
    elif border_treatment_method == "AVOID":
        return np.array(arr)
    elif border_treatment_method == "CONSTANT_0":
        return np.pad(arr, padding, constant_values=0)
    elif border_treatment_method == "CONSTANT_2":
        return np.pad(arr, padding, constant_values=2)
    elif border_treatment_method == "REPEAT":
        return np.pad(arr, padding, 'edge')
    elif border_treatment_method == "SYMMETRIC_REFLECT":
        return np.pad(arr, padding, 'symmetric')
    elif border_treatment_method == "WRAP":
        return np.pad(arr, padding, 'wrap')
    else:
        raise ValueError(f"Unknown method '{border_treatment_method}'.")


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
    image = Image.open(f'./resources/tests/{file_name}.{file_extension}')
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
        [-1, 8, -1],
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


def normalize_after_convolution(arr, mode):
    if mode == "rr":
        arr -= np.min(arr)
        arr /= np.max(arr)
    elif mode == "cr":
        arr_max = np.max(arr)
        arr = np.clip(arr, a_min=0, a_max=arr_max)
        arr /= arr_max
    elif mode == "cc":
        arr = np.clip(arr, a_min=0, a_max=1.0)
    else:
        raise ValueError(f"Unknown normalization mode '{mode}'")

    return np.round(arr * 255, 11)


def save_image(arr, file_name, border_treatment_method, normalization_method, file_extension, options, file_format):
    padding, stride, dilation = options

    option_folder = f'{"x".join(map(str, padding))}-{"x".join(map(str, stride))}-{"x".join(map(str, dilation))}'
    file_path = f'./resources/tests/{normalization_method}/{option_folder}/{border_treatment_method}/{file_name}.{file_extension}'
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
    result_image.save(file_path, format=file_format)


def convolve_with_method(options, method, normalization_method, file_extension, file_format):
    file_name = "Piercing-The-Ocean"
    image = read_image(file_name, file_extension)
    padding, stride, dilation = options

    prepared_image = prepare_input(image, method, padding)
    result_data = convolve(prepared_image, stride, dilation)

    result_data = normalize_after_convolution(result_data, normalization_method)
    if result_data.shape[2] == 1:
        result_data = result_data.squeeze(2)

    save_image(result_data, file_name, method, normalization_method, file_extension, options, file_format)


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

    normalization_methods = (
        "rr",
        "cr",
        "cc",
    )

    parameters = (list_of_options, border_treatment_methods, normalization_methods, extensions)
    total = np.prod([len(x) for x in parameters])

    for options, method, normalization_method, (file_extension, file_format) in tqdm(product(*parameters), ncols=100, total=total):
        convolve_with_method(options, method, normalization_method, file_extension, file_format)


if __name__ == "__main__":
    main()
