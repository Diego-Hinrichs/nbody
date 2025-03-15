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

inline bool checkCudaAvailability()
{
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess)
    {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
        return false;
    }

    if (deviceCount == 0)
    {
        std::cerr << "No CUDA-capable devices found" << std::endl;
        return false;
    }

    // Print GPU info
    cudaDeviceProp deviceProp;
    for (int i = 0; i < deviceCount; i++)
    {
        cudaGetDeviceProperties(&deviceProp, i);
        std::cout << "CUDA Device " << i << ": " << deviceProp.name << std::endl;
        std::cout << "  Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "  Total global memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    }

    // Set device to use
    cudaSetDevice(0);

    return true;
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
    bool initialized_;

public:
    CudaTimer(float &elapsed_time) : elapsed_time_(elapsed_time), stopped_(false), initialized_(false)
    {
        // Create start and stop events with error checking
        cudaError_t startErr = cudaEventCreate(&start_);
        cudaError_t stopErr = cudaEventCreate(&stop_);

        if (startErr != cudaSuccess || stopErr != cudaSuccess)
        {
            // Log the specific error
            std::cerr << "CUDA Timer initialization failed: "
                      << cudaGetErrorString(startErr) << " / "
                      << cudaGetErrorString(stopErr) << std::endl;

            // Set elapsed time to 0 as fallback
            elapsed_time_ = 0.0f;
            return;
        }

        initialized_ = true;
        cudaEventRecord(start_);
    }

    ~CudaTimer()
    {
        if (!stopped_ && initialized_)
        {
            stop();
        }

        if (initialized_)
        {
            cudaEventDestroy(start_);
            cudaEventDestroy(stop_);
        }
    }

    void stop()
    {
        if (!initialized_)
            return;

        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
        cudaEventElapsedTime(&elapsed_time_, start_, stop_);
        stopped_ = true;
    }
};

#endif // ERROR_HANDLING_CUH