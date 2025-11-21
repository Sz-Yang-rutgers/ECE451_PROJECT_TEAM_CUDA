// rkf45_cuda.cu
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>

#define CUDA_CHECK(err) \
    do { \
        cudaError_t e = (err); \
        if (e != cudaSuccess) { \
            std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__ \
                      << " : " << cudaGetErrorString(e) << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

// Example RHS: y'_i = -y_i (component-wise exponential decay)
// Replace this with your own system if needed.
__global__
void rhs_kernel(int n, double t, const double* y, double* dydt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        (void)t; // t is unused in this simple example
        dydt[i] = -y[i];
    }
}

__global__
void scale_kernel(int n, double h, double* v)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        v[i] *= h;
    }
}

__global__
void copy_kernel(int n, const double* src, double* dst)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = src[i];
    }
}

// y_out[i] = y[i] + a1*k1[i] + a2*k2[i] + ... + a6*k6[i]
__global__
void lincomb6_kernel(int n,
                     const double* y,
                     double a1, const double* k1,
                     double a2, const double* k2,
                     double a3, const double* k3,
                     double a4, const double* k4,
                     double a5, const double* k5,
                     double a6, const double* k6,
                     double* y_out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double val = y[i];
        if (k1) val += a1 * k1[i];
        if (k2) val += a2 * k2[i];
        if (k3) val += a3 * k3[i];
        if (k4) val += a4 * k4[i];
        if (k5) val += a5 * k5[i];
        if (k6) val += a6 * k6[i];
        y_out[i] = val;
    }
}

// Compute per-component squared normalized error:
// err2[i] = ((y5[i] - y4[i]) / (atol + rtol*max(|y[i]|,|y5[i]|)))^2
__global__
void error_kernel(int n,
                  const double* y,
                  const double* y4,
                  const double* y5,
                  double atol,
                  double rtol,
                  double* err2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double yi  = y[i];
        double y4i = y4[i];
        double y5i = y5[i];

        double sc = atol + rtol * fmax(fabs(yi), fabs(y5i));
        double ratio = (y5i - y4i) / sc;
        err2[i] = ratio * ratio;
    }
}

// One adaptive RKF45 step on the GPU.
// d_y, d_k1..d_k6, d_ytemp, d_y4, d_y5, d_err2 are device pointers of size n.
void rkf45_step_cuda(int n,
                     double& t,
                     double* d_y,
                     double* d_k1,
                     double* d_k2,
                     double* d_k3,
                     double* d_k4,
                     double* d_k5,
                     double* d_k6,
                     double* d_ytemp,
                     double* d_y4,
                     double* d_y5,
                     double* d_err2,
                     double& h,
                     double atol = 1e-6,
                     double rtol = 1e-6)
{
    const double safety     = 0.9;
    const double min_factor = 0.2;
    const double max_factor = 5.0;

    int threads = 256;
    int blocks  = (n + threads - 1) / threads;

    std::vector<double> h_err2(n);

    while (true) {
        // k1
        rhs_kernel<<<blocks, threads>>>(n, t, d_y, d_k1);
        CUDA_CHECK(cudaGetLastError());
        scale_kernel<<<blocks, threads>>>(n, h, d_k1);

        // k2: y_temp = y + 1/4 k1
        lincomb6_kernel<<<blocks, threads>>>(
            n, d_y,
            0.25, d_k1,
            0.0, nullptr,
            0.0, nullptr,
            0.0, nullptr,
            0.0, nullptr,
            0.0, nullptr,
            d_ytemp);
        CUDA_CHECK(cudaGetLastError());
        rhs_kernel<<<blocks, threads>>>(n, t + 0.25 * h, d_ytemp, d_k2);
        CUDA_CHECK(cudaGetLastError());
        scale_kernel<<<blocks, threads>>>(n, h, d_k2);

        // k3: y_temp = y + (3/32)k1 + (9/32)k2
        lincomb6_kernel<<<blocks, threads>>>(
            n, d_y,
            3.0/32.0, d_k1,
            9.0/32.0, d_k2,
            0.0, nullptr,
            0.0, nullptr,
            0.0, nullptr,
            0.0, nullptr,
            d_ytemp);
        CUDA_CHECK(cudaGetLastError());
        rhs_kernel<<<blocks, threads>>>(n, t + 3.0/8.0 * h, d_ytemp, d_k3);
        CUDA_CHECK(cudaGetLastError());
        scale_kernel<<<blocks, threads>>>(n, h, d_k3);

        // k4: y_temp = y + (1932/2197)k1 - (7200/2197)k2 + (7296/2197)k3
        lincomb6_kernel<<<blocks, threads>>>(
            n, d_y,
            1932.0/2197.0, d_k1,
           -7200.0/2197.0, d_k2,
            7296.0/2197.0, d_k3,
            0.0, nullptr,
            0.0, nullptr,
            0.0, nullptr,
            d_ytemp);
        CUDA_CHECK(cudaGetLastError());
        rhs_kernel<<<blocks, threads>>>(n, t + 12.0/13.0 * h, d_ytemp, d_k4);
        CUDA_CHECK(cudaGetLastError());
        scale_kernel<<<blocks, threads>>>(n, h, d_k4);

        // k5: y_temp = y + (439/216)k1 - 8k2 + (3680/513)k3 - (845/4104)k4
        lincomb6_kernel<<<blocks, threads>>>(
            n, d_y,
            439.0/216.0,  d_k1,
           -8.0,          d_k2,
            3680.0/513.0, d_k3,
           -845.0/4104.0, d_k4,
            0.0, nullptr,
            0.0, nullptr,
            d_ytemp);
        CUDA_CHECK(cudaGetLastError());
        rhs_kernel<<<blocks, threads>>>(n, t + h, d_ytemp, d_k5);
        CUDA_CHECK(cudaGetLastError());
        scale_kernel<<<blocks, threads>>>(n, h, d_k5);

        // k6: y_temp = y - (8/27)k1 + 2k2 - (3544/2565)k3 + (1859/4104)k4 - (11/40)k5
        lincomb6_kernel<<<blocks, threads>>>(
            n, d_y,
           -8.0/27.0,       d_k1,
            2.0,            d_k2,
           -3544.0/2565.0,  d_k3,
            1859.0/4104.0,  d_k4,
           -11.0/40.0,      d_k5,
            0.0, nullptr,
            d_ytemp);
        CUDA_CHECK(cudaGetLastError());
        rhs_kernel<<<blocks, threads>>>(n, t + 0.5 * h, d_ytemp, d_k6);
        CUDA_CHECK(cudaGetLastError());
        scale_kernel<<<blocks, threads>>>(n, h, d_k6);

        // 4th-order solution y4:
        // y4 = y + (25/216)k1 + (1408/2565)k3 + (2197/4104)k4 - (1/5)k5
        lincomb6_kernel<<<blocks, threads>>>(
            n, d_y,
            25.0/216.0,       d_k1,
            0.0,              nullptr,
            1408.0/2565.0,    d_k3,
            2197.0/4104.0,    d_k4,
           -1.0/5.0,          d_k5,
            0.0,              nullptr,
            d_y4);
        CUDA_CHECK(cudaGetLastError());

        // 5th-order solution y5:
        // y5 = y + (16/135)k1 + (6656/12825)k3 + (28561/56430)k4 - (9/50)k5 + (2/55)k6
        lincomb6_kernel<<<blocks, threads>>>(
            n, d_y,
            16.0/135.0,        d_k1,
            0.0,               nullptr,
            6656.0/12825.0,    d_k3,
            28561.0/56430.0,   d_k4,
           -9.0/50.0,          d_k5,
            2.0/55.0,          d_k6,
            d_y5);
        CUDA_CHECK(cudaGetLastError());

        // Error vector on device
        error_kernel<<<blocks, threads>>>(
            n, d_y, d_y4, d_y5, atol, rtol, d_err2);
        CUDA_CHECK(cudaGetLastError());

        // Copy error^2 back to host and compute norm
        CUDA_CHECK(cudaMemcpy(h_err2.data(), d_err2,
                              n * sizeof(double),
                              cudaMemcpyDeviceToHost));

        double sum = 0.0;
        for (int i = 0; i < n; ++i) {
            sum += h_err2[i];
        }
        double err_norm = std::sqrt(sum / static_cast<double>(n));

        if (err_norm <= 1.0) {
            // Accept step
            t += h;
            // y = y5
            copy_kernel<<<blocks, threads>>>(n, d_y5, d_y);
            CUDA_CHECK(cudaGetLastError());

            // New step size (order = 5 -> exponent 1/(p+1) = 1/6 ~ 0.166..,
            // but often 0.2 is used; keep consistent with earlier code)
            double factor = safety * std::pow(std::max(err_norm, 1e-10), -0.2);
            factor = std::min(max_factor, std::max(min_factor, factor));
            h *= factor;
            break;
        } else {
            // Reject step, shrink h
            double factor = safety * std::pow(std::max(err_norm, 1e-10), -0.25);
            factor = std::min(1.0, std::max(min_factor, factor));
            h *= factor;

            if (h < 1e-16) {
                std::cerr << "Step size underflow in rkf45_step_cuda\n";
                std::exit(EXIT_FAILURE);
            }
        }
    }
}

int main()
{
    // Problem size: vector dimension
    const int    N     = 1024;   // number of equations
    double       t     = 0.0;
    const double t_end = 5.0;
    double       h     = 0.1;
    double       atol  = 1e-6;
    double       rtol  = 1e-6;

    // Host state (vector)
    std::vector<double> h_y(N, 1.0);  // y_i(0) = 1 for all i

    // Device pointers
    double *d_y   = nullptr;
    double *d_k1  = nullptr;
    double *d_k2  = nullptr;
    double *d_k3  = nullptr;
    double *d_k4  = nullptr;
    double *d_k5  = nullptr;
    double *d_k6  = nullptr;
    double *d_ytemp = nullptr;
    double *d_y4    = nullptr;
    double *d_y5    = nullptr;
    double *d_err2  = nullptr;

    size_t bytes = N * sizeof(double);

    CUDA_CHECK(cudaMalloc(&d_y,     bytes));
    CUDA_CHECK(cudaMalloc(&d_k1,    bytes));
    CUDA_CHECK(cudaMalloc(&d_k2,    bytes));
    CUDA_CHECK(cudaMalloc(&d_k3,    bytes));
    CUDA_CHECK(cudaMalloc(&d_k4,    bytes));
    CUDA_CHECK(cudaMalloc(&d_k5,    bytes));
    CUDA_CHECK(cudaMalloc(&d_k6,    bytes));
    CUDA_CHECK(cudaMalloc(&d_ytemp, bytes));
    CUDA_CHECK(cudaMalloc(&d_y4,    bytes));
    CUDA_CHECK(cudaMalloc(&d_y5,    bytes));
    CUDA_CHECK(cudaMalloc(&d_err2,  bytes));

    // Copy initial condition to GPU
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), bytes, cudaMemcpyHostToDevice));

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "# t      y_num[0]   y_exact\n";

    while (t < t_end) {
        if (t + h > t_end)
            h = t_end - t;

        rkf45_step_cuda(
            N, t,
            d_y,
            d_k1, d_k2, d_k3, d_k4, d_k5, d_k6,
            d_ytemp, d_y4, d_y5, d_err2,
            h, atol, rtol
        );

        // Fetch just the first component to print
        CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost));
        double y_exact = std::exp(-t);
        std::cout << t << "  " << h_y[0] << "  " << y_exact << "\n";
    }

    // Clean up
    cudaFree(d_y);
    cudaFree(d_k1);
    cudaFree(d_k2);
    cudaFree(d_k3);
    cudaFree(d_k4);
    cudaFree(d_k5);
    cudaFree(d_k6);
    cudaFree(d_ytemp);
    cudaFree(d_y4);
    cudaFree(d_y5);
    cudaFree(d_err2);

    return 0;
}
