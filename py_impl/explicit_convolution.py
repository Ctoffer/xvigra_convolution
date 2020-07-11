import numpy as np
from convolution_util import KernelOptions, ChannelPosition, calculate_output_size


def convolve1D(input: np.ndarray,
               kernel: np.ndarray,
               options: KernelOptions
               ):
    if options.channel_position is ChannelPosition.IMPLICIT:
        raise ValueError("convolve1D(): Implicit channel option is not supported for explicit channels in input!")

    if input.ndim != 2:
        raise ValueError("convolve1D(): Need 2 dimensional (W x C or C x W) input!")

    if kernel.ndim != 3:
        raise ValueError("convolve1D(): Need a full 3-dimensional (C_out x C_in x W) kernel!")

    if options.channel_position is ChannelPosition.LAST:
        input_width = input.shape[0]
        input_channels = input.shape[1]
    else:
        input_width = input.shape[1]
        input_channels = input.shape[0]

    kernel_size = kernel.shape[2]
    radius = kernel_size // 2

    if kernel_size % 2 == 0:
        kernel_minimum = 0
        kernel_maximum = kernel_size
    else:
        kernel_minimum = -radius
        kernel_maximum = radius + 1

    if input_channels != kernel.shape[1]:
        raise ValueError("convolve1D(): Input channels of input and kernel do not align!")

    if input_width + options.padding_total < (kernel_size - 1) * options.dilation + 1:
        raise ValueError("convolve1D(): Kernel width is greater than padded input width!")

    if kernel_size % 2 == 0:
        input_width_minimum = -options.padding_begin
        input_width_maximum = input_width + options.padding_end - options.dilation * (kernel_size - 1)
    else:
        input_width_minimum = -options.padding_begin + options.dilation * abs(kernel_minimum)
        input_width_maximum = input_width + options.padding_end - options.dilation * radius

    output_width = calculate_output_size(input_width, kernel_size, options)
    input_width_indices = list(range(input_width_minimum, input_width_maximum, options.stride))

    if options.channel_position is ChannelPosition.FIRST:
        patch = np.zeros((input_channels, kernel_size, output_width))
        output_channels = kernel.shape[0]

        for input_channel in range(input_channels):
            for kernel_x in range(kernel_minimum, kernel_maximum):
                kernel_offset_x = options.dilation * kernel_x
                patch_kernel_x = kernel_x + abs(kernel_minimum)

                for out_index, input_width_index in enumerate(input_width_indices):
                    input_x = input_width_index + kernel_offset_x

                    if 0 <= input_x < input_width:
                        patch[input_channel, patch_kernel_x, out_index] = input[input_channel, input_x]

        reshaped_kernel = np.reshape(kernel, (output_channels, input_channels * kernel_size))
        reshaped_patch = np.reshape(patch, (input_channels * kernel_size, output_width))
        result = np.matmul(reshaped_kernel, reshaped_patch)
    else:
        patch = np.zeros((output_width, input_channels, kernel_size))
        output_channels = kernel.shape[0]

        for out_index, input_width_index in enumerate(input_width_indices):
            for kernel_x in range(kernel_minimum, kernel_maximum):
                kernel_offset_x = options.dilation * kernel_x
                patch_kernel_x = kernel_x + abs(kernel_minimum)
                input_x = input_width_index + kernel_offset_x

                if 0 <= input_x < input_width:
                    for input_channel in range(input_channels):
                        patch[out_index, input_channel, patch_kernel_x] = input[input_x, input_channel]

        reshaped_kernel = np.reshape(kernel, (output_channels, input_channels * kernel_size))
        reshaped_patch = np.reshape(patch, (output_width, input_channels * kernel_size))
        result = np.matmul(reshaped_patch, np.transpose(reshaped_kernel))

    return result


def convolve2D(input: np.ndarray,
               kernel: np.ndarray,
               options_y: KernelOptions,
               options_x: KernelOptions
               ):
    if options_y.channel_position != options_x.channel_position:
        raise ValueError("convolve2D(): Channel can't be on different positions for optionsY and optionsX!")

    if options_y.channel_position is ChannelPosition.IMPLICIT:
        raise ValueError("convolve2D(): Implicit channel option is not supported for explicit channels in input!")

    if input.ndim != 3:
        raise ValueError("convolve2D(): Need 3 dimensional (H x W x C or C x H x W) input!")

    if kernel.ndim != 4:
        raise ValueError("convolve2D(): Need a full 4-dimensional (C_out x C_in x H x W) kernel!")

    if options_y.channel_position == ChannelPosition.FIRST:
        input_channels = input.shape[0]
        input_height = input.shape[1]
        input_width = input.shape[2]
    else:
        input_height = input.shape[0]
        input_width = input.shape[1]
        input_channels = input.shape[2]

    output_channels = kernel.shape[0]
    kernel_height = kernel.shape[2]
    kernel_width = kernel.shape[3]

    if input_channels != kernel.shape[1]:
        raise ValueError("convolve2D(): Input channels of input and kernel do not align!")

    if input_height + options_y.padding_total < (kernel_height - 1) * options_y.dilation + 1:
        raise ValueError("convolve2D(): Kernel height is greater than padded input height!")

    if input_width + options_x.padding_total < (kernel_width - 1) * options_x.dilation + 1:
        raise ValueError("convolve2D(): Kernel width is greater than padded input width!")

    kernel_height_radius = kernel_height // 2
    kernel_height_minimum = 0 if kernel_height % 2 == 0 else -kernel_height_radius
    kernel_height_maximum = kernel_height if kernel_height % 2 == 0 else (kernel_height_radius + 1)

    kernel_width_radius = kernel_width // 2
    kernel_width_minimum = 0 if kernel_width % 2 == 0 else -kernel_width_radius
    kernel_width_maximum = kernel_width if kernel_width % 2 == 0 else (kernel_width_radius + 1)

    output_height = calculate_output_size(input_height, kernel_height, options_y)
    output_width = calculate_output_size(input_width, kernel_width, options_x)

    height_minimum = -options_y.padding_begin + options_y.dilation * abs(0 if kernel_height % 2 == 0 else kernel_height_minimum)
    height_maximum = input_height + options_y.padding_end - options_y.dilation * ((kernel_height - 1) if kernel_height % 2 == 0 else kernel_height_radius)

    width_minimum = -options_x.padding_begin + options_x.dilation * abs(0 if kernel_width % 2 == 0 else kernel_width_minimum)
    width_maximum = input_width + options_x.padding_end - options_x.dilation * ((kernel_width - 1) if kernel_width % 2 == 0 else kernel_width_radius)

    input_height_indices = list(range(height_minimum, height_maximum, options_y.stride))
    input_width_indices = list(range(width_minimum, width_maximum, options_x.stride))

    if options_y.channel_position is ChannelPosition.FIRST:
        patch = np.zeros((input_channels, kernel_height, kernel_width, output_height, output_width))

        for input_channel in range(input_channels):
            for kernel_y in range(kernel_height_minimum, kernel_height_maximum):
                out_kernel_y = kernel_y + abs(kernel_height_minimum)
                input_offset_y = kernel_y * options_y.dilation

                for kernel_x in range(kernel_width_minimum, kernel_width_maximum):
                    out_kernel_x = kernel_x + abs(kernel_width_minimum)
                    input_offset_x = kernel_x * options_x.dilation

                    for out_index_y, input_index_y in enumerate(input_height_indices):
                        input_y = input_index_y + input_offset_y

                        for out_index_x, input_index_x in enumerate(input_width_indices):
                            input_x = input_index_x + input_offset_x

                            if 0 <= input_y < input_height and 0 <= input_x < input_width:
                                patch[input_channel, out_kernel_y, out_kernel_x, out_index_y, out_index_x] = input[input_channel, input_y, input_x]

        reshaped_kernel = np.reshape(kernel, (output_channels, input_channels * kernel_height * kernel_width))
        reshaped_patch = np.reshape(patch, (input_channels * kernel_height * kernel_width, output_height * output_width))
        result = np.reshape(np.matmul(reshaped_kernel, reshaped_patch), (output_channels, output_height, output_width))
    else:
        patch = np.zeros((output_height, output_width, input_channels, kernel_height, kernel_width))

        for out_index_y, input_index_y in enumerate(input_height_indices):
            for out_index_x, input_index_x in enumerate(input_width_indices):
                for kernel_y in range(kernel_height_minimum, kernel_height_maximum):
                    out_kernel_y = kernel_y + abs(kernel_height_minimum)
                    input_offset_y = kernel_y * options_y.dilation
                    input_y = input_index_y + input_offset_y

                    for kernel_x in range(kernel_width_minimum, kernel_width_maximum):
                        out_kernel_x = kernel_x + abs(kernel_width_minimum)
                        input_offset_x = kernel_x * options_x.dilation
                        input_x = input_index_x + input_offset_x

                        if 0 <= input_y < input_height and 0 <= input_x < input_width:
                            for input_channel in range(input_channels):
                                patch[out_index_y, out_index_x, input_channel, out_kernel_y, out_kernel_x] = input[input_y, input_x, input_channel]

        reshaped_kernel = np.transpose(np.reshape(kernel, (output_channels, input_channels * kernel_height * kernel_width)))
        reshaped_patch = np.reshape(patch, (output_height * output_width, input_channels * kernel_height * kernel_width))
        result = np.reshape(np.matmul(reshaped_patch, reshaped_kernel), (output_height, output_width, output_channels))

    return result
