from itertools import product
from json import load as j_load
from os import makedirs
from os.path import join as p_join

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from plot_benchmark_file import FileContext, load_yml_config, parse_x, parse_arguments


def get_color_colors_of_block(values, mode):
    color_map = cm.get_cmap(mode, 100)
    vals = np.array(values)
    cell_colors = list()

    for i in range(vals.shape[0]):
        amin = np.amin(vals[i, :])
        vals[i, :] -= amin
        vals[i, :] /= np.amax(vals[i, :])
        cell_colors.append([color_map(val / 2 + 0.25) for val in vals[i, :]])

    return cell_colors


def draw_double_header_table(group_headers, subgroup_headers, subgroup_data: dict, color_modes):
    row_numbers = set(map(len, subgroup_data.values()))
    if len(row_numbers) != 1:
        raise ValueError("Subgroup-data has different number of rows for different keys!")
    else:
        row_numbers = tuple(row_numbers)
    number_of_rows = row_numbers[0] + 2

    longest_numbers = ([max(list(arr.flatten()), key=lambda x: len(str(x))) for arr in subgroup_data.values()])
    subgroup_header_length = len(max(subgroup_headers, key=len)) - max(subgroup_headers, key=len).count('-')
    factor = max(len(str(max(map(str, longest_numbers), key=len))), subgroup_header_length) / 7

    fig_width = int(factor * len(group_headers) * len(subgroup_headers))
    fig_height = (row_numbers[0] // 2) - 1
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    ax.table(
        cellText=[[''] * len(group_headers)],
        colLabels=group_headers,
        bbox=[0, (number_of_rows - 3) / number_of_rows, 1, 2 / number_of_rows],
        cellLoc='center',
    )

    for i, group in enumerate(group_headers):
        right_shift = len(str(max(np.floor(subgroup_data[group]), key=lambda x: len(str(int(x[0]))))))
        s = f' >{right_shift + 5}.4f'

        ax.table(
            cellText=[[("{value:" + s + "}").format(value=value) for value in row] for row in subgroup_data[group]],
            colLabels=subgroup_headers,
            cellColours=get_color_colors_of_block(subgroup_data[group], color_modes[group]),
            bbox=[i / len(group_headers), 0, 1 / len(group_headers), row_numbers[0] / number_of_rows],
            cellLoc='right',
        )

    fig.tight_layout()
    return fig


def extract_information(all_combinations, data):
    result = list()
    for top_level, specialization, group_id, variant, aggregate in all_combinations:
        key_word_args = {
            "top_level": top_level,
            "specialization": specialization,
            "group_id": group_id,
            "variant": variant,
            "aggregate": aggregate
        }
        key = data.key.format(**key_word_args)
        value = getattr(data.values, key)
        if value not in result:
            result.append(value)

    return tuple(result)


def load_subgroup_data(data_dir, group_headers, data_config, subgroup_data):
    variants = data_config.variants
    group_names = {group_id: getattr(data_config.group_names, group_id) for group_id in data_config.group_ids}
    x_range = parse_x(data_config.x)

    result = dict()
    for group_header in group_headers:
        group_data = getattr(subgroup_data.values, group_header)
        rows = list()

        with open(p_join(data_dir, group_data.file), 'r', encoding='utf-8') as fp:
            data = {e["name"]: e for e in j_load(fp)["benchmarks"]}
            for x in x_range:
                tmp = list()
                data_keys = set()
                for variant, group_id in product(variants, data_config.group_ids):
                    kwargs = {"x": x, "variant": variant, "group_name": group_names[group_id]}
                    data_key = group_data.key.format(**kwargs)

                    if data_key not in data_keys:
                        data_keys.add(data_key)
                        tmp.append(data[data_key]["cpu_time"])
                rows.append(tmp)

        result[group_header] = np.round(np.array(rows), 4)
    return result


def draw_table(data_dir, config_dir, out_dir, name):
    yml_config = load_yml_config(p_join(config_dir, f"{name}.yml"))
    data_config, table_config = yml_config.data, yml_config.table

    all_combinations = list(product(data_config.top_levels,
                                    data_config.specializations,
                                    data_config.group_ids,
                                    data_config.variants,
                                    data_config.aggregates))

    group_headers = extract_information(all_combinations, table_config.group_headers)
    subgroup_headers = extract_information(all_combinations, table_config.subgroup_headers)
    subgroup_data = load_subgroup_data(
        data_dir,
        group_headers,
        data_config,
        table_config.subgroup_data
    )
    color_modes = {key: getattr(table_config.color_modes.values, key) for key in group_headers}

    figure = draw_double_header_table(group_headers, subgroup_headers, subgroup_data, color_modes)

    makedirs(out_dir, exist_ok=True)
    figure.savefig(p_join(out_dir, f"{name}.png"), dpi=240)


def main(file_context: FileContext):
    draw_table(file_context.data_dir, file_context.config_dir, file_context.out_dir, file_context.file_name)


if __name__ == "__main__":
    main(parse_arguments())
