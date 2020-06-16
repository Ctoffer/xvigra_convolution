from json import load as j_load
import os
from os.path import join as p_join, abspath as p_abs

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

from collections import defaultdict
import re
import platform


def init_plot(title, x_axis_name, y_axis_name):
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=16)
    plt.rc('legend', fontsize=12)

    plt.figure(figsize=(24, 18))
    plt.subplot(311)

    plt.title(title, fontsize=20)
    plt.xlabel(x_axis_name, fontsize=18)
    plt.ylabel(y_axis_name, fontsize=18)

    plt.grid(axis='y')


def group_data(path):
    def entry_classification(entry):
        pattern = re.compile(
            r".*_(?P<channel>channelOuterDimension|channelInnerDimension)<(?P<type>.*)>/(?P<param>\d+)")
        matcher = pattern.match(entry)
        channel, element_type, parameter = matcher.group("channel"), matcher.group("type"), matcher.group("param")
        return channel, element_type, parameter

    with open(path, 'r') as json_fp:
        json_object = j_load(json_fp)
    tmp = defaultdict(lambda: defaultdict(list))

    for entry in json_object["benchmarks"]:
        channel, element_type, parameter = entry_classification(entry["name"])
        cpu_time = entry["cpu_time"]

        tmp[channel][element_type].append((int(parameter), float(cpu_time)))

    result = dict()
    for channel in tmp:
        channel_result = dict()
        for k, v in tmp[channel].items():
            x, y = zip(*v)
            channel_result[k] = (np.array(x), np.array(y))

        result[channel] = channel_result

    return result


def plot_data(data):
    line_colors = {'std::uint16_t': '#55868C',
                   'int': '#EEC584',
                   'float': '#CE4257',
                   'double': '#59546C'}
    cell_predefined_colors = {'std::uint16_t': '#55868C',
                              'int': '#EEC584',
                              'float': '#CE4257',
                              'double': '#59546C'}

    columns = list()
    cell_colors = list()
    header_colors = list()
    rows = range(49)
    cell_text = list()

    for key, (x_data, y_data) in data.items():
        rows = x_data
        if x_data[2] > 50:
            plt.xticks([_ for _ in x_data if _ % 100 == 0])
        else:
            plt.xticks(x_data)

        plt.plot(x_data, y_data,
                 label=key,
                 linestyle="-",
                 color=line_colors[key]
                 )

    for key, (x_data, y_data) in data.items():
        columns.append(key)
        cell_text.append([f"{_:8.0f}" for _ in y_data])
        cell_colors.append([cell_predefined_colors[key] for _ in y_data])
        header_colors.append(line_colors[key])

    plt.legend(loc="upper left")

    plt.subplot(312)
    plt.axis("off")
    plt.title("Absolute Times In Nanoseconds", fontsize=20, y=0.94)
    table = plt.table(cellText=np.swapaxes(np.array(cell_text), 0, 1),
                      rowLabels=rows,
                      colLabels=columns,
                      cellLoc='center',
                      rowLoc='center',
                      colColours=header_colors,
                      cellColours=np.swapaxes(np.array(cell_colors), 0, 1),
                      bbox=[0, -1.5, 1.0, 2.45])

    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for (row, col), cell in table.get_celld().items():
        if (row == 0) or (col == -1):
            cell.set_text_props(fontproperties=FontProperties(weight='bold'))


def plot(name, x_axis_name, titles):
    path = os.path.join("benchmarks_result", platform.system(), "data", f"{name}.json")
    y_axis_name = "CPU time (ns)"

    data = group_data(path)

    for dim, title in titles.items():
        init_plot(title=title, x_axis_name=x_axis_name, y_axis_name=y_axis_name)
        plot_data(data[dim])
        out_path = os.path.join("benchmarks_result", platform.system(), "plots",f"plot_{name}_{dim}.svg")
        ensure_directories_exist(out_path)
        print(f"Save diagram to {out_path}")
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()


def ensure_directories_exist(path):
    parent = os.path.dirname(path)
    if not os.path.exists(parent):
        os.makedirs(parent)


def plot_conv1d():
    plot(name="conv1d_measure_access_time",
         x_axis_name="Input Width W",
         titles={"channelOuterDimension": "Conv1D(3, 2, 4)\nFilter: 3 x 3 x 7\nInput: 3 x W",
                 "channelInnerDimension": "Conv1D(3, 2, 4)\nFilter: 3 x 3 x 7\nInput: W x 3"}
         )
    plot(name="conv1d_measure_channels",
         x_axis_name="Input Channels I\nOutput Channels O := I - 1",
         titles={"channelOuterDimension": "Conv1D(3, 2, 4)\nFilter: O x I x 7\nInput: 3 x 500",
                 "channelInnerDimension": "Conv1D(3, 2, 4)\nFilter: O x I x 7\nInput: 500 x 3"}
         )
    plot(name="conv1d_measure_dilation",
         x_axis_name="Dilation D",
         titles={"channelOuterDimension": "Conv1D(3, D, 4)\nFilter: 3 x 3 x 7\nInput: 3 x 500",
                 "channelInnerDimension": "Conv1D(3, D, 4)\nFilter: 3 x 3 x 7\nInput: 500 x 3"}
         )
    plot(name="conv1d_measure_input_size",
         x_axis_name="Input Width W",
         titles={"channelOuterDimension": "Conv1D(3, 2, 4)\nFilter: 3 x 3 x 7\nInput: 3 x W",
                 "channelInnerDimension": "Conv1D(3, 2, 4)\nFilter: 3 x 3 x 7\nInput: W x 3"}
         )
    plot(name="conv1d_measure_padding",
         x_axis_name="Padding P",
         titles={"channelOuterDimension": "Conv1D(P, 2, 4)\nFilter: 3 x 3 x 7\nInput: 3 x 500",
                 "channelInnerDimension": "Conv1D(P, 2, 4)\nFilter: 3 x 3 x 7\nInput: 500 x 3"}
         )
    plot(name="conv1d_measure_stride",
         x_axis_name="Stride S",
         titles={"channelOuterDimension": "Conv1D(3, 2, S)\nFilter: 3 x 3 x 7\nInput: 3 x 500",
                 "channelInnerDimension": "Conv1D(3, 2, S)\nFilter: 3 x 3 x 7\nInput: 500 x 3"}
         )


def plot_conv2d():
    plot(name="conv2d_measure_access_time",
         x_axis_name="Input Width W",
         titles={"channelOuterDimension": "Conv2D({4, 3}, {3, 2}, {3, 4})\n"
                                          "Filter: 3 x 3 x 6 x 7\n"
                                          "Input: 3 x W+1 x W",
                 "channelInnerDimension": "Conv2D({4, 3}, {3, 2}, {3, 4})\n"
                                          "Filter: 3 x 3 x 6 x 7\n"
                                          "Input: W+1 x W x 3"}
         )
    plot(name="conv2d_measure_channels",
         x_axis_name="Input Channels I\nOutput Channels O := I - 1",
         titles={"channelOuterDimension": "Conv2D({4, 3}, {3, 2}, {3, 4})\n"
                                          "Filter: O x I x 6 x 7\n"
                                          "Input: 3 x 501 x 500",
                 "channelInnerDimension": "Conv2D({4, 3}, {3, 2}, {3, 4})\n"
                                          "Filter: O x I x 6 x 7\n"
                                          "Input: 501 x 500 x 3"}
         )
    plot(name="conv2d_measure_dilation",
         x_axis_name="Dilation D",
         titles={"channelOuterDimension": "Conv2D({4, 3}, {D+1, D}, {3, 4})\n"
                                          "Filter: 3 x 3 x 6 x 7\n"
                                          "Input: 3 x 501 x 500",
                 "channelInnerDimension": "Conv2D({4, 3}, {D+1, D}, {3, 4})\n"
                                          "Filter: 3 x 3 x 6 x 7\n"
                                          "Input: 501 x 500 x 3"}
         )
    plot(name="conv2d_measure_input_size",
         x_axis_name="Input Width W",
         titles={"channelOuterDimension": "Conv2D({4, 3}, {3, 2}, {3, 4})\n"
                                          "Filter: 3 x 3 x 6 x 7\n"
                                          "Input: 3 x W+1 x W",
                 "channelInnerDimension": "Conv2D({4, 3}, {3, 2}, {3, 4})\n"
                                          "Filter: 3 x 3 x 6 x 7\n"
                                          "Input: W+1 x W x 3"}
         )
    plot(name="conv2d_measure_padding",
         x_axis_name="Padding P",
         titles={"channelOuterDimension": "Conv2D({P+1, P}, {3, 2}, {3, 4})\n"
                                          "Filter: 3 x 3 x 6 x 7\n"
                                          "Input: 3 x 501 x 500",
                 "channelInnerDimension": "Conv2D({P+1, P}, {3, 2}, {3, 4})\n"
                                          "Filter: 3 x 3 x 6 x 7\n"
                                          "Input: 501 x 500 x 3"}
         )
    plot(name="conv2d_measure_stride",
         x_axis_name="Stride S",
         titles={"channelOuterDimension": "Conv2D({4, 3}, {3, 2}, {S, S+1})\n"
                                          "Filter: 3 x 3 x 6 x 7\n"
                                          "Input: 3 x 501 x 500",
                 "channelInnerDimension": "Conv2D({4, 3}, {3, 2}, {S, S+1})\n"
                                          "Filter: 3 x 3 x 6 x 7\n"
                                          "Input: 501 x 500 x 3"}
         )


def main():
    #plot_conv1d()
    plot_conv2d()


if __name__ == "__main__":
    main()
