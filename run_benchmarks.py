import platform
import os.path
import subprocess


def call_benchmark(name, folder="conv1d"):
    try:
        is_windows = "Windows" == platform.system()
        out_format = "console"
        min_time = 1
        json_path = os.path.join("benchmarks_result", platform.system(), "data", f"{name}.json")
        ensure_directories_exist(json_path)

        if is_windows:
            executable_file = os.path.join("build-windows", "benchmarks", folder, "Release", f"{name}.exe")
        else:
            executable_file = os.path.join("build-linux", "benchmarks", folder, name)

        subprocess.call([
            f".\\{executable_file}" if is_windows else f"./{executable_file}"
            , f'--benchmark_format={out_format}'
            , f'--benchmark_min_time={min_time}'
            , f'--benchmark_out={json_path}'
        ], cwd=os.getcwd(), shell=is_windows)

        if not os.path.exists(json_path):
            raise ValueError(f"Data file '{json_path}' was not created.")

    except ValueError:
        print(name, "failed")


def ensure_directories_exist(path):
    parent = os.path.dirname(path)
    if not os.path.exists(parent):
        os.makedirs(parent)


def run_benchmarks(name="conv1d"):
    #call_benchmark(f"{name}_measure_access_time", folder=name)
    #call_benchmark(f"{name}_measure_channels", folder=name)
    #call_benchmark(f"{name}_measure_dilation", folder=name)
    call_benchmark(f"{name}_measure_input_size", folder=name)
    #call_benchmark(f"{name}_measure_padding", folder=name)
    #call_benchmark(f"{name}_measure_stride", folder=name)


def main():
    #run_benchmarks(name="conv1d")
    run_benchmarks(name="conv2d")


if __name__ == "__main__":
    main()
