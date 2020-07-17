from argparse import ArgumentParser

from plot_benchmark_file import FileContext
from plot_table_file import main as plot_with_context


def plot_both_from_args():
    file_name, directory_name = parse_file_name()
    main(file_name, os_name="Windows", folder=directory_name)
    main(file_name, os_name="Linux", folder=directory_name)


def parse_file_name():
    parser = ArgumentParser(description="Plots a table of data using the plot_table_file.py.")
    parser.add_argument(
        '-d',
        '--directory_name',
        dest='directory_name',
        type=str,
        help="name of the directory, which will be used for data and out"
    )
    parser.add_argument(
        '-f',
        '--file_name',
        dest='file_name',
        type=str,
        help="name of the file, which will be used for data, config and out"
    )
    arguments = parser.parse_args()
    return arguments.file_name, arguments.directory_name


def main(file_name, os_name="Windows", folder="xvigra"):
    context = FileContext(
        data_dir=f"benchmarks_result/{os_name}/data/{folder}",
        out_dir=f"benchmarks_result/{os_name}/tables/{folder}",
        config_dir=f"benchmarks_result/config-table/{folder}",
        file_name=file_name
    )
    plot_with_context(context)


if __name__ == "__main__":
    plot_both_from_args()
