from plot_benchmark import main as graph


def main():
    benchmark_graphs = {
        #"xtensor": (
        #    "benchmark_normalizing",
        #    "benchmark_transpose-view",
        #    "benchmark_reshape-view",
        #    "benchmark_tensor_copy_complete_X",
        #    "benchmark_tensor_copy_complete_Y",
        #    "benchmark_tensor_copy_complete_Z",
        #    "benchmark_strided-view_copy_complete_X",
        #    "benchmark_strided-view_copy_complete_Y",
        #    "benchmark_strided-view_copy_complete_Z",
        #    "benchmark_strided-view_copy_paddingStride_X",
        #    "benchmark_strided-view_copy_paddingStride_Y",
        #    "benchmark_strided-view_copy_paddingStride_Z",
        #    "benchmark_view_copy_complete_X",
        #    "benchmark_view_copy_complete_Y",
        #    "benchmark_view_copy_complete_Z",
        #    "benchmark_view_copy_paddingStride_X",
        #    "benchmark_view_copy_paddingStride_Y",
        #    "benchmark_view_copy_paddingStride_Z",
        #),
        "xvigra": (
            "benchmark_convolve1D_inputSize_channelFirst",
            "benchmark_convolve1D_inputSize_channelLast",
            "benchmark_convolve2D_inputSize_channelFirst",
            "benchmark_convolve2D_inputSize_channelLast",
            "benchmark_separableConvolve1D_inputSize",
            "benchmark_separableConvolve2D_inputSize",
            "benchmark_separableConvolve1D_kernelSize",
            "benchmark_separableConvolve2D_kernelSize"
        )
    }

    print("Create graphs")
    for os_name in ("Windows", "Linux"):
        print("Current OS:", os_name)
        for folder_name, benchmark_files in benchmark_graphs.items():
            for file_name in benchmark_files:
                try:
                    print("  Create", file_name, "... ", end="")
                    graph(file_name, os_name=os_name, folder=folder_name)
                    print("  Finished graph")
                except BaseException as e:
                    print("  Failed graph", type(e), e)
                    raise e
        print()

if __name__ == "__main__":
    main()
