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


def draw_double_header_table(group_headers, subgroup_headers, subgroup_data: dict, color_modes, data_config):
    row_numbers = set(map(len, subgroup_data.values()))
    if len(row_numbers) != 1:
        raise ValueError("Subgroup-data has different number of rows for different keys!")
    else:
        row_numbers = tuple(row_numbers)
    number_of_rows = row_numbers[0]

    axis_label = data_config.axis_label
    longest_numbers = ([max(list(arr.flatten()), key=lambda x: len(str(x))) for arr in subgroup_data.values()])
    subgroup_header_length = len(max(subgroup_headers, key=len))
    max_num_len = len(str(max(map(str, longest_numbers), key=len)))
    factor = max(int(0.6 * max_num_len), int(0.5 * subgroup_header_length), int(0.5 * len(axis_label))) / 7
    # print(int(0.6 * max_num_len), int(0.6 * subgroup_header_length), int(0.5 * len(axis_label)))
    group_dependent_factors = {1: 1.7, 2: 1, 3: 1.7}
    subgroup_dependent_factors = {4: 1.1, 5: 1, 6: 1, 8: 0.5, 10: 0.6, 12: 0.6}

    fig_width = int(factor
                    * len(group_headers) * group_dependent_factors[len(group_headers)]
                    * len(subgroup_headers) * subgroup_dependent_factors[len(subgroup_headers)]
                    )
    fig_height = (row_numbers[0] // 3)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    col_len = subgroup_data[group_headers[0]].shape[1]
    if len(subgroup_headers) == col_len:
        cell_width = 1 / (len(subgroup_headers) * len(group_headers) + 1)
    else:
        cell_width = 1 / (len(subgroup_headers) + 1)

    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    ax.table(
        cellText=np.expand_dims(parse_x(data_config.x), axis=1),
        colLabels=[axis_label],
        bbox=[0, 0, cell_width, (row_numbers[0] - 1) / number_of_rows],
        cellLoc='center',
    )

    ax.table(
        cellText=[[''] * len(group_headers)],
        colLabels=group_headers,
        bbox=[cell_width, (number_of_rows - 2) / number_of_rows, 1 - cell_width, 2 / number_of_rows],
        cellLoc='center',
    )

    for i, group in enumerate(group_headers):
        longest_cell_number = max(list(np.floor(subgroup_data[group]).astype(int).flatten()), key=lambda x: len(str(x)))
        right_shift = len(str(longest_cell_number))
        s = f' >{right_shift + 0}.4f'
        col_len = subgroup_data[group].shape[1]
        if len(subgroup_headers) == col_len:
            column_labels = subgroup_headers

            ax.table(
                cellText=[[("{value:" + s + "}").format(value=value) for value in row] for row in subgroup_data[group]],
                colLabels=column_labels,
                cellColours=get_color_colors_of_block(subgroup_data[group], color_modes[group]),
                bbox=[cell_width + i * cell_width * len(subgroup_headers), 0, cell_width * len(subgroup_headers),
                      (row_numbers[0] - 1) / number_of_rows],
                cellLoc='right',
            )
        else:
            column_labels = subgroup_headers[i * col_len:col_len + i * col_len]

            ax.table(
                cellText=[[("{value:" + s + "}").format(value=value) for value in row] for row in subgroup_data[group]],
                colLabels=column_labels,
                cellColours=get_color_colors_of_block(subgroup_data[group], color_modes[group]),
                bbox=[cell_width + i * cell_width * len(subgroup_headers) / len(group_headers), 0, cell_width * len(subgroup_headers) / len(group_headers),
                      (row_numbers[0] - 1) / number_of_rows],
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

    figure = draw_double_header_table(group_headers, subgroup_headers, subgroup_data, color_modes, data_config)

    makedirs(out_dir, exist_ok=True)
    figure.savefig(p_join(out_dir, f"{name}.png"), dpi=240)
    plt.clf()
    plt.cla()
    plt.close(figure)


def main(file_context: FileContext):
    draw_table(file_context.data_dir, file_context.config_dir, file_context.out_dir, file_context.file_name)


if __name__ == "__main__":
    main(parse_arguments())
