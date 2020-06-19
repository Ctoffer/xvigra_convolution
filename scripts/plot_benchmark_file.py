from sys import argv
from argparse import ArgumentParser
from os.path import join as p_join, exists as p_exists, dirname as p_dirname
from os import makedirs
from types import SimpleNamespace

from yaml import load as y_load, FullLoader as YamlLoader
from json import load as j_load
import numpy as np

import matplotlib.pyplot as plt


def setup_argument_parser():
    parser = ArgumentParser(description="Plots data from data_file according to config_file into out_file.")
    parser.add_argument(
        '-d',
        '--data_dir',
        dest='data_dir',
        type=str,
        help="directory containing the data as json"
    )
    parser.add_argument(
        '-c',
        '--config_dir',
        dest='config_dir',
        type=str,
        help="directory containing plot configuration as yml"
    )
    parser.add_argument(
        '-o',
        '--out_dir',
        dest='out_dir',
        type=str,
        help="directory where the final graph should be saved as svg"
    )
    parser.add_argument(
        '-f',
        '--file_name',
        dest='file_name',
        type=str,
        help="name of the file, which will be used for data, config and out"
    )

    return parser


class FileContext:
    @staticmethod
    def from_args(args):
        return FileContext(
            data_dir=args.data_dir,
            config_dir=args.config_dir,
            out_dir=args.out_dir,
            file_name=args.file_name
        )

    def __init__(self, data_dir, config_dir, out_dir, file_name):
        self._data_file = p_join(data_dir, file_name + ".json")
        self._config_file = p_join(config_dir, file_name + ".yml")
        self._out_file = p_join(out_dir, file_name + ".svg")

        if not p_exists(self._data_file):
            raise ValueError(f"Data file '{self._data_file}' does not exist!")

        if not p_exists(self._config_file):
            raise ValueError(f"Config file '{self._config_file}' does not exist!")

        if not p_exists(p_dirname(self._out_file)):
            makedirs(p_dirname(self._out_file))

    @property
    def data_file(self):
        return self._data_file

    @property
    def config_file(self):
        return self._config_file

    @property
    def out_file(self):
        return self._out_file


def dict_to_simple_namespace(d) -> SimpleNamespace:
    if type(d) is not dict:
        return d
    return SimpleNamespace(**{key: dict_to_simple_namespace(value) for key, value in d.items()})


def load_yml_config(path):
    with open(path, 'r', encoding='utf-8') as fp:
        return dict_to_simple_namespace(y_load(fp, Loader=YamlLoader))


def parse_x(x_config):
    if x_config.type == "list":
        return x_config.values
    elif x_config.type == "range":
        return range(x_config.attrs.start, x_config.attrs.stop, x_config.attrs.step)
    elif x_config.type == "np.arange":
        return np.arange(x_config.attrs.start, x_config.attrs.stop, x_config.attrs.step)
    else:
        raise ValueError(f"Unknown x type '{x_config.type}'!")


def load_data(data_file):
    with open(data_file, 'r', encoding='utf-8') as fp:
        return j_load(fp)['benchmarks']


def group_data(benchmark_data, data_config):
    entries = {entry['name']: entry['cpu_time'] for entry in benchmark_data}

    x_values = parse_x(data_config.x)
    y_values = dict()

    base_key = data_config.key

    for specialization in data_config.specializations:
        for group_id in data_config.group_ids:
            for variant in data_config.variants:
                for aggregate in data_config.aggregates:
                    accumulated_data = np.array(x_values, dtype=np.float64)
                    for i, x in enumerate(x_values):
                        key = base_key.format(
                            group_name=getattr(data_config.group_names, group_id),
                            specialization=specialization,
                            x=x,
                            variant=variant,
                            aggregate=aggregate
                        )
                        accumulated_data[i] = entries[key]
                    y_values[(specialization, group_id, variant, aggregate)] = accumulated_data

    return x_values, y_values


def init_plot(graph_config):
    title = graph_config.title
    x_axis_name = graph_config.x_axis.label
    y_axis_name = graph_config.y_axis.label

    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=16)
    plt.rc('legend', fontsize=12)

    plt.figure(figsize=(24, 9))

    plt.title(title, fontsize=20)
    plt.xlabel(x_axis_name, fontsize=18)
    plt.ylabel(y_axis_name, fontsize=18)

    plt.grid(axis='y')


def plot_data(graph_config, x_data, y_values, specialization, group_id, variant, aggregate):
    y_data = y_values[(specialization, group_id, variant, aggregate)]
    key_kwargs = {
        'specialization': specialization,
        'group_id': group_id,
        'variant': variant,
        'aggregate': aggregate
    }
    legend_key = graph_config.legend.key.format(**key_kwargs)
    line_style_key = graph_config.line_style.key.format(**key_kwargs)
    line_color_key = graph_config.line_color.key.format(**key_kwargs)

    mode = graph_config.x_axis.mode, graph_config.y_axis.mode
    label = getattr(graph_config.legend.entries, legend_key)
    line_style = getattr(graph_config.line_style.entries, line_style_key)
    line_color = getattr(graph_config.line_color.entries, line_color_key)

    plot_kwargs = {'label': label, 'linestyle': line_style, 'color': line_color}

    if mode == ('normal', 'normal'):
        plot_function = plt.plot
    elif mode == ('normal', 'log'):
        plot_function = plt.semilogy
    elif mode == ('log', 'normal'):
        plot_function = plt.semilogx
    elif mode == ('log', 'log'):
        plot_function = plt.loglog
    else:
        raise ValueError(f"Unknown axis type combination '{mode}'!")

    plot_function(
        x_data,
        y_data,
        **plot_kwargs
    )


def plot_all(file_context, yml_config):
    data_config, graph_config = yml_config.data, yml_config.graph
    benchmark_data = load_data(file_context.data_file)
    x_values, y_values = group_data(benchmark_data, data_config)

    init_plot(graph_config)

    for specialization in data_config.specializations:
        for group_id in data_config.group_ids:
            for variant in data_config.variants:
                for aggregate in data_config.aggregates:
                    plot_data(graph_config, x_values, y_values, specialization, group_id, variant, aggregate)

    plt.legend(loc="upper left")
    print(f"Save diagram to {file_context.out_file}")
    plt.savefig(file_context.out_file, bbox_inches='tight')
    plt.close()


def main():
    parser = setup_argument_parser()
    file_context = FileContext.from_args(parser.parse_args())
    yml_config = load_yml_config(file_context.config_file)

    plot_all(file_context, yml_config)


if __name__ == "__main__":
    main()
