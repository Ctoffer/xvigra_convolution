from plot_table import main as table


def main():
    benchmark_tables = {
        "xtensor": (
            "benchmark_normalizing",
            "benchmark_transpose-view",
            "benchmark_reshape-view",
            "benchmark_tensor_copy_completeX",
            "benchmark_tensor_copy_completeY",
            "benchmark_tensor_copy_completeZ",
            "benchmark_strided-view_copy_completeX",
            "benchmark_strided-view_copy_completeY",
            "benchmark_strided-view_copy_completeZ",
            "benchmark_strided-view_copy_paddingStrideX",
            "benchmark_strided-view_copy_paddingStrideY",
            "benchmark_strided-view_copy_paddingStrideZ",
            "benchmark_view_copy_completeX",
            "benchmark_view_copy_completeY",
            "benchmark_view_copy_completeZ",
            "benchmark_view_copy_paddingStrideX",
            "benchmark_view_copy_paddingStrideY",
            "benchmark_view_copy_paddingStrideZ",
        ),
        "xvigra": (
            "benchmark_convolve1D_inputSize",
            "benchmark_convolve2D_inputSize",
            "benchmark_separableConvolve1D_inputSize",
            "benchmark_separableConvolve2D_inputSize",
            "benchmark_separableConvolve1D_kernelSize",
            "benchmark_separableConvolve2D_kernelSize"
        )
    }

    print("Create tables")
    for os_name in ("Windows", "Linux"):
        print("Current OS:", os_name)
        for folder_name, benchmark_files in benchmark_tables.items():
            for file_name in benchmark_files:
                try:
                    print("  Create", file_name, "... ", end="")
                    table(file_name, os_name=os_name, folder=folder_name)
                    print("Finished table")
                except BaseException as e:
                    print("Failed table", type(e), e)
                    raise e
        print()

if __name__ == "__main__":
    main()
