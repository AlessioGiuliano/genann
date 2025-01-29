#include <cstddef>
#include <iostream>

#include "genann.h"
#include <cuda_runtime.h>


__global__ void genann_act_sigmoid_kernel(double* lookup, double f)
{
    const size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    double a = sigmoid_dom_min + f * i;
    if (i < LOOKUP_SIZE)
    {
        if (a < -45.0)
            lookup[i] = 0.;
        else if (a > 45.0)
            lookup[i] = 1.;
        else
            lookup[i] = 1.0 / (1 + std::exp(-a));
    }
}

#define cudaSafeCall(call)                                                                                             \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t err = call;                                                                                        \
        if (err != cudaSuccess)                                                                                        \
        {                                                                                                              \
            std::cout << "Line " << __LINE__ << ": cuda failure (" << cudaGetErrorString(err) << ')' << std::endl;                              \
        }                                                                                                              \
    } while (0)

double genann_init_sigmoid_lookup_cuda(double* lookup) {
        const double f = (sigmoid_dom_max - sigmoid_dom_min) / LOOKUP_SIZE;

        double* d_lookup;
        auto size = LOOKUP_SIZE * sizeof(double);

        cudaSafeCall(cudaMalloc((void**) &d_lookup, size));
        cudaSafeCall(cudaMemcpy((void*) d_lookup, lookup, size, cudaMemcpyHostToDevice));

        const size_t threadsPerBlock = 32;
        const size_t nbBlocks = std::ceil(LOOKUP_SIZE / threadsPerBlock);

        genann_act_sigmoid_kernel<<<nbBlocks, threadsPerBlock>>>(d_lookup, f);
        cudaSafeCall(cudaGetLastError());

        cudaSafeCall(cudaMemcpy((void*)lookup, d_lookup, size, cudaMemcpyDeviceToHost));
        cudaSafeCall(cudaFree((void*)d_lookup));

        return LOOKUP_SIZE / (sigmoid_dom_max - sigmoid_dom_min);
}
