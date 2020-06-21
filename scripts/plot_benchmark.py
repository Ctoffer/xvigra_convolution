from argparse import ArgumentParser

from plot_benchmark_file import FileContext, main as plot_with_context


def plot_both_from_args():
    file_name = parse_file_name()
    main(file_name, os_name="Windows")
    main(file_name, os_name="Linux")


def parse_file_name():
    parser = ArgumentParser(description="Plots data using the plot_benchmark_file.py.")
    parser.add_argument(
        '-f',
        '--file_name',
        dest='file_name',
        type=str,
        help="name of the file, which will be used for data, config and out"
    )
    arguments = parser.parse_args()
    return arguments.file_name


def main(file_name, os_name="Windows"):
    context = FileContext(
        data_dir=f"benchmarks_result/{os_name}/data",
        out_dir=f"benchmarks_result/{os_name}/plots",
        config_dir=f"benchmarks_result/config",
        file_name=file_name
    )
    plot_with_context(context)


if __name__ == "__main__":
    plot_both_from_args()
