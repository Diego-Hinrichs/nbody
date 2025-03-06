#ifndef ERROR_HANDLING_CUH
#define ERROR_HANDLING_CUH

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

/**
 * @brief Check CUDA error and output diagnostic information
 * @param err CUDA error code to check
 * @param func Name of the function that returned the error
 * @param file Source file name
 * @param line Line number in the source file
 */
inline void checkCudaError(cudaError_t err, const char *const func, const char *const file, int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

/**
 * @brief Check last CUDA error and output diagnostic information
 * @param file Source file name
 * @param line Line number in the source file
 */
inline void checkLastCudaError(const char *const file, int line)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// Macros for more convenient error checking
#define CHECK_CUDA_ERROR(val) checkCudaError((val), #val, __FILE__, __LINE__)
#define CHECK_LAST_CUDA_ERROR() checkLastCudaError(__FILE__, __LINE__)

// Macro for proper CUDA kernel call with error checking
#define CUDA_KERNEL_CALL(kernel, gridSize, blockSize, sharedMem, stream, ...) \
    do                                                                        \
    {                                                                         \
        kernel<<<gridSize, blockSize, sharedMem, stream>>>(__VA_ARGS__);      \
        CHECK_LAST_CUDA_ERROR();                                              \
    } while (0)

// Timer helper class for CUDA events
class CudaTimer
{
private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
    float &elapsed_time_;
    bool stopped_;

public:
    CudaTimer(float &elapsed_time) : elapsed_time_(elapsed_time), stopped_(false)
    {
        CHECK_CUDA_ERROR(cudaEventCreate(&start_));
        CHECK_CUDA_ERROR(cudaEventCreate(&stop_));
        CHECK_CUDA_ERROR(cudaEventRecord(start_));
    }

    ~CudaTimer()
    {
        if (!stopped_)
        {
            stop();
        }
        CHECK_CUDA_ERROR(cudaEventDestroy(start_));
        CHECK_CUDA_ERROR(cudaEventDestroy(stop_));
    }

    void stop()
    {
        CHECK_CUDA_ERROR(cudaEventRecord(stop_));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop_));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_time_, start_, stop_));
        stopped_ = true;
    }
};

#endif // ERROR_HANDLING_CUH