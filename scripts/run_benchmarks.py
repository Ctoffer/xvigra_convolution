import platform
import os.path
import subprocess
import sys

from util import TimeMeasure, string_framed_line


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

    execute_command = ["PowerShell", "-ExecutionPolicy", "Unrestricted", "-File", ".\\build_all.ps1"] if is_windows else ["bash",  "build_all.sh"]
    subprocess.call(execute_command, cwd=os.getcwd(), shell=is_windows)


def call_benchmark(file_name, benchmark_parameters, folder="xvigra"):
    os_name = platform.system()
    is_windows = "Windows" == os_name

    if is_windows:
        executable_file = os.path.join("build-windows", "benchmarks", folder, "Release", f"{file_name}.exe")
    else:
        executable_file = os.path.join("build-linux", "benchmarks", folder, file_name)

    json_path = f"./benchmarks_result/{os_name}/data/{file_name}.json"

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
        "min_time": 1.0,
        "repetitions": 3,
        "report_aggregates_only": True
    }

    benchmark_files = (
        # "benchmark_convolve1D_inputSize",
        "benchmark_separableConvolve1D_inputSize",
        "benchmark_separableConvolve2D_inputSize"
    )

    build_all()

    for file_name in benchmark_files:
        with TimeMeasure(f"{'─' * 100}\nRunning {file_name}:", f"Total time: {{}}\n{'─' * 100}\n"):
            call_benchmark(file_name, benchmark_parameters)
        print("\n")


if __name__ == "__main__":
    main()
