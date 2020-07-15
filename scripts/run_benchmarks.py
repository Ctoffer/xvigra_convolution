import platform
import os.path
import subprocess
import sys

from util import TimeMeasure, string_framed_line
from plot_benchmark import main as plot


def build_all():
    os_name = platform.system()
    is_windows = "Windows" == os_name

    lines = string_framed_line(
        title=f"Running on {os_name}",
        length=100,
        style='='
    )
    for line in lines:
        print(line)
    print()

    execute_command = ["PowerShell", "-ExecutionPolicy", "Unrestricted", "-File",
                       ".\\build_all_windows.ps1"] if is_windows else ["bash", "build_all_linux.sh", "--no-clear"]
    subprocess.call(execute_command, cwd=os.getcwd(), shell=is_windows)


def call_benchmark(file_name, benchmark_parameters, folder="xvigra"):
    os_name = platform.system()
    is_windows = "Windows" == os_name

    if is_windows:
        executable_file = os.path.join("build-windows", "benchmarks", folder, "Release", f"{file_name}.exe")
    else:
        executable_file = os.path.join("build-linux", "benchmarks", folder, file_name)

    json_path = f"./benchmarks_result/{os_name}/data/{folder}/{file_name}.json"

    execute_command = f".\\{executable_file}" if is_windows else f"./{executable_file}"

    args = [f'--benchmark_{key}={value}' for key, value in benchmark_parameters.items()]
    args.append(f'--benchmark_out={json_path}')
    for argument in args:
        print(f"   {argument}")
    print()

    subprocess.call([execute_command] + args, cwd=os.getcwd(), shell=is_windows, stdout=open(os.devnull, 'w'))
    print()


def main():
    benchmark_parameters = {
        "format": "console",
        "min_time": 1,
        "repetitions": 10,
        "report_aggregates_only": True
    }

    benchmark_folders = {
        #"xtensor": (
            #"benchmark_normalizing",
            #"benchmark_strided-view_copy_complete_X",
            #"benchmark_strided-view_copy_complete_Y",
            #"benchmark_strided-view_copy_complete_Z",
            #"benchmark_strided-view_copy_paddingStride_X",
            #"benchmark_strided-view_copy_paddingStride_Y",
            #"benchmark_strided-view_copy_paddingStride_Z",
            #"benchmark_view_copy_complete_X",
            #"benchmark_view_copy_complete_Y",
            #"benchmark_view_copy_complete_Z",
            #"benchmark_view_copy_paddingStride_X",
            #"benchmark_view_copy_paddingStride_Y",
            #"benchmark_view_copy_paddingStride_Z",
        #),
         "xvigra": (
        #   "benchmark_convolve1D_inputSize_channelFirst",
        #   "benchmark_convolve1D_inputSize_channelLast",
        #   "benchmark_convolve2D_inputSize_channelFirst",
        #   "benchmark_convolve2D_inputSize_channelLast",
           "benchmark_separableConvolve1D_inputSize",
           "benchmark_separableConvolve2D_inputSize",
           "benchmark_separableConvolve1D_kernelSize",
           "benchmark_separableConvolve2D_kernelSize"
         )
    }

    build_all()

    for folder_name, benchmark_files in benchmark_folders.items():
        for file_name in benchmark_files:
            with TimeMeasure(f"{'─' * 100}\nRunning {file_name}:", f"Total time: {{}}\n{'─' * 100}\n"):
                call_benchmark(file_name, benchmark_parameters, folder=folder_name)
            print("\nPlotting...")
            plot(file_name, os_name=platform.system(), folder=folder_name)
            print("Finished plot\n")


if __name__ == "__main__":
    main()
