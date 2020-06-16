import numpy as np


def calculate_max_value(dimensions, axis, start_axis, end_axis):
    prod = 1

    for i in range(start_axis, end_axis):
        current_dim = dimensions[i]

        if axis != i:
            prod *= current_dim

    return prod


def decompose_index(compound_idx, dimensions, current_axis, start_axis, end_axis):
    result = ['all'] * len(dimensions)

    for i in range(start_axis, end_axis):
        max_value = calculate_max_value(dimensions, current_axis, i + 1, end_axis)
        if i != current_axis:
            rest = compound_idx % max_value
            result[i] = (compound_idx - rest) // max_value
            compound_idx = rest
    return result


def main():
    shape = (6, 5, 1)
    channel_posititon = len(shape) - 1
    start_axis, end_axis = 0, 2

    for axis in range(start_axis, end_axis):
        for channel in range(shape[channel_posititon]):
            print(axis)
            max_value = calculate_max_value(shape, axis, start_axis, end_axis)

            for compound_idx in range(max_value):
                decomposed_indices = decompose_index(compound_idx, shape, axis, start_axis, end_axis)

                decomposed_indices[channel_posititon] = channel
                print("  ", compound_idx, decomposed_indices)

    arr = np.array([
        [[1], [2], [3], [4], [5]],
        [[6], [7], [8], [9], [10]],
        [[11], [12], [13], [14], [15]],
        [[16], [17], [18], [19], [20]],
        [[21], [22], [23], [24], [25]],
        [[26], [27], [28], [29], [30]]
    ])
    print(arr.shape)
    print(arr[:, 0, 0])
    print(arr[:, 1, 0])


if __name__ == "__main__":
    main()
