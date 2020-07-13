import sys
import traceback
from itertools import product
from os import makedirs
from time import sleep
from os.path import dirname, basename, join as p_join

import numpy as np
import torch.nn.functional as F
from torch import randint
from tqdm import tqdm

from convolution_util import ChannelPosition, KernelOptions
from explicit_convolution import convolve1D, convolve2D
from util import TimeMeasure


def compare_convolve1D(data, silent=True):
    (input_channels, input_width), (output_channels, kernel_width), (padding, dilation, stride), channel_position = data
    if not silent:
        print("=" * 120)
        print(f"input_channels: {input_channels}, input_width: {input_width}")
        print(f"output_channels: {output_channels}, kernel_width: {kernel_width}")
        print(f"padding: {padding}, dilation: {dilation}, stride: {stride}")
        print(f"channel_position: {channel_position}")
        print("_" * 120)

    def random_tensor(size, low=0, high=1000):
        return randint(low=low, high=high, size=size)

    kernels = random_tensor(size=(output_channels, input_channels, kernel_width), low=-10, high=10)

    if channel_position is ChannelPosition.FIRST:
        input_tensor = random_tensor(size=(input_channels, input_width))
    else:
        input_tensor = random_tensor(size=(input_width, input_channels))

    reference_failed = False
    try:
        if channel_position is ChannelPosition.FIRST:
            tmp = input_tensor.unsqueeze(0)
            expected_result = F.conv1d(input=tmp,
                                       weight=kernels,
                                       padding=padding,
                                       dilation=dilation,
                                       stride=stride
                                       )
            expected_result = expected_result.squeeze(0)
        else:
            tmp = input_tensor.transpose(0, 1).unsqueeze(0)
            expected_result = F.conv1d(input=tmp,
                                       weight=kernels,
                                       padding=padding,
                                       dilation=dilation,
                                       stride=stride
                                       )
            expected_result = expected_result.squeeze(0).transpose(0, 1)
    except RuntimeError:
        reference_failed = True

    actual_failed = False
    try:
        options = KernelOptions(padding=padding,
                                stride=stride,
                                dilation=dilation,
                                channel_position=channel_position
                                )
        actual_result = convolve1D(input_tensor.numpy(), kernels.numpy(), options)
    except ValueError:
        actual_failed = True

    if reference_failed and actual_failed:
        return True  # both not possible

    if reference_failed != actual_failed:
        raise RuntimeError(f"discrepancy reference: {reference_failed}, custom: {actual_failed}")

    assert actual_result.shape == expected_result.numpy().shape
    assert np.all(actual_result == expected_result.numpy())

    if not silent:
        print("Ok")
        print("_" * 120)


def check_convolution1D():
    logfile = f"test_result/convolve1D.txt"
    makedirs("test_result", exist_ok=True)

    input_channel_values = range(1, 5)
    input_widths = (3, 5, 7, 13, 17, 23, 50, 100, 200, 250, 500, 1000)
    kernel_widths = (2, 3, 4, 5, 7, 11, 13)
    paddings = range(0, 5)
    dilations = range(1, 6)
    strides = range(1, 6)
    channel_indices = (ChannelPosition.FIRST, ChannelPosition.LAST)

    parameters = (input_channel_values,
                  input_widths,
                  kernel_widths,
                  paddings,
                  dilations,
                  strides,
                  channel_indices)

    with open(logfile, "w") as fp:
        sleep(1)
        number_of_combinations = np.prod([len(_) for _ in parameters])

        for params in tqdm(product(*parameters), ncols=100, total=number_of_combinations, mininterval=5):
            input_channels, input_width = params[0:2]
            kernel_width = params[2]
            padding, dilation, stride = params[3:6]
            channel_index = params[6]

            for output_channels in range(1, input_channels):
                key = ((input_channels, input_width),
                       (output_channels, kernel_width),
                       (padding, dilation, stride),
                       channel_index)
                try:
                    if compare_convolve1D(key):
                        continue

                    print(f"{key}: Ok", file=fp)
                except (AssertionError, IndexError, RuntimeError):
                    print(f"{key}: Error", file=fp)
                except ValueError:
                    print(f"{key}: Not possible", file=fp)

    print_results(logfile)


def print_results(logfile):
    summary_path = p_join(dirname(logfile) + "_summary", basename(logfile))
    counters = {"Ok": 0, "Error": 0, "Not possible": 0}
    with open(logfile, "r") as fp:
        for line in fp:
            counters[line.split(":")[-1].strip()] += 1

    total = sum([v for v in counters.values()])
    makedirs(dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding='utf-8') as file:
        print({k: f"{100 * v / total:5.2f} %" for k, v in counters.items()})
        print(f"Total number of tests: {total}")
        print({k: f"{100 * v / total:5.2f} %" for k, v in counters.items()}, file=file)
        print(f"Total number of tests: {total}", file=file)


def compare_convolve2D(data, silent=True):
    input_channels, input_height, input_width = data[0]
    output_channels, kernel_height, kernel_width = data[1]
    (padding_y, padding_x), (dilation_y, dilation_x), (stride_y, stride_x) = data[2]
    channel_position = data[3]

    if not silent:
        print("=" * 120)
        print(f"input_channels: {input_channels}, "
              f"input_height: {input_height}, "
              f"input_width: {input_width}")
        print(f"output_channels: {output_channels},"
              f" kernel: {kernel_height}x{kernel_width}")
        print(f"padding: ({padding_y}, {padding_x}), "
              f"dilation: ({dilation_y}, {dilation_x}), "
              f"stride: ({stride_y}, {stride_x})")
        print(f"channel_position: {channel_position}")
        print("_" * 120)

    def random_tensor(size, low=0, high=1000):
        return randint(low=low, high=high, size=size)

    kernels = random_tensor(size=(output_channels, input_channels, kernel_height, kernel_width), low=-10, high=10)

    if channel_position is ChannelPosition.FIRST:
        input_tensor = random_tensor(size=(input_channels, input_height, input_width))
    else:
        input_tensor = random_tensor(size=(input_height, input_width, input_channels))

    reference_failed = False
    try:
        if channel_position is ChannelPosition.FIRST:
            tmp = input_tensor.unsqueeze(0)
            expected_result = F.conv2d(input=tmp,
                                       weight=kernels,
                                       padding=(padding_y, padding_x),
                                       dilation=(dilation_y, dilation_x),
                                       stride=(stride_y, stride_x)
                                       )
            expected_result = expected_result.squeeze(0)
        else:
            tmp = input_tensor.transpose(0, 2).transpose(1, 2).unsqueeze(0)
            expected_result = F.conv2d(input=tmp,
                                       weight=kernels,
                                       padding=(padding_y, padding_x),
                                       dilation=(dilation_y, dilation_x),
                                       stride=(stride_y, stride_x)
                                       )
            expected_result = expected_result.squeeze(0).transpose(1, 2).transpose(0, 2)
    except RuntimeError as e:
        reference_failed = True

    actual_failed = False
    try:
        options_y = KernelOptions(padding=padding_y,
                                  stride=stride_y,
                                  dilation=dilation_y,
                                  channel_position=channel_position
                                  )
        options_x = KernelOptions(padding=padding_x,
                                  stride=stride_x,
                                  dilation=dilation_x,
                                  channel_position=channel_position
                                  )
        actual_result = convolve2D(input_tensor.numpy(), kernels.numpy(), options_y, options_x)
    except ValueError as e:
        actual_failed = True

    if reference_failed and actual_failed:
        return True  # both not possible

    if reference_failed != actual_failed:
        raise RuntimeError(f"discrepancy reference: {reference_failed}, custom: {actual_failed}")

    expected_result = expected_result.numpy()
    assert actual_result.shape == expected_result.shape
    assert np.all(actual_result == expected_result)

    if not silent:
        print("Ok")
        print("_" * 120)


def check_convolution2D():
    logfile = f"test_result/convolve2D.txt"
    makedirs("test_result", exist_ok=True)

    input_channel_values = range(1, 4)
    input_sizes = (7, 8, 49, 50)
    kernel_sizes = (2, 3, 8, 11)
    paddings = range(0, 4)
    dilations = range(1, 5)
    strides = range(1, 5)
    channel_indices = (ChannelPosition.FIRST, ChannelPosition.LAST)

    parameters = (input_channel_values,
                  input_sizes,
                  input_sizes,
                  kernel_sizes,
                  kernel_sizes,
                  paddings,
                  paddings,
                  dilations,
                  dilations,
                  strides,
                  strides,
                  channel_indices)

    with open(logfile, "w") as fp:
        sleep(1)
        number_of_combinations = np.prod([len(_) for _ in parameters])

        for params in tqdm(product(*parameters), ncols=100, total=number_of_combinations, mininterval=5):
            input_channels, input_height, input_width = params[0:3]
            kernel_height, kernel_width = params[3:5]
            padding_y, padding_x = params[5:7]
            dilation_y, dilation_x = params[7:9]
            stride_y, stride_x = params[9:11]
            channel_index = params[11]

            for output_channels in range(1, input_channels):
                key = (
                    (input_channels, input_height, input_width),
                    (output_channels, kernel_height, kernel_width),
                    ((padding_y, padding_x), (dilation_y, dilation_x), (stride_y, stride_x)),
                    channel_index)
                try:
                    if compare_convolve2D(key):
                        continue

                    print(f"{key}: Ok", file=fp)
                except (AssertionError, IndexError, RuntimeError) as e:
                    print(f"{key}: Error", file=fp)
                except ValueError:
                    print(f"{key}: Not possible", file=fp)

    print_results(logfile)


def main():
    try:
        with TimeMeasure():
            check_convolution1D()

        with TimeMeasure():
            check_convolution2D()

    except Exception:
        sleep(1)
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)


if __name__ == "__main__":
    main()
