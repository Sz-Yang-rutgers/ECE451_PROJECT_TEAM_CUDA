#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "Body.h"

constexpr float G = 5.67e-11f;       // universal gravitational constant
constexpr float SOFTENING = 1e-9f;   // avoids division by zero for identical positions

#define CUDA_CHECK(expr)                                                             \
    do {                                                                             \
        cudaError_t __err = (expr);                                                  \
        if (__err != cudaSuccess) {                                                  \
            std::cerr << "CUDA error (" << #expr << "): "                           \
                      << cudaGetErrorString(__err) << '\n';                         \
            std::exit(EXIT_FAILURE);                                                 \
        }                                                                            \
    } while (0)

__global__ void update_velocities(int n,
                                  const float* m,
                                  const float* x,
                                  const float* y,
                                  const float* z,
                                  float* vx,
                                  float* vy,
                                  float* vz,
                                  float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float xi = x[idx];
    float yi = y[idx];
    float zi = z[idx];

    float dvx = 0.0f;
    float dvy = 0.0f;
    float dvz = 0.0f;

    for (int j = 0; j < n; j++) {
        if (j == idx)
            continue;
        float dx = xi - x[j];
        float dy = yi - y[j];
        float dz = zi - z[j];
        float dist2 = dx * dx + dy * dy + dz * dz + SOFTENING;
        float inv_dist = 1.0f / sqrtf(dist2); //r
        float inv_dist3 = inv_dist * inv_dist * inv_dist;
        float scale = -G * m[j] * inv_dist3 * dt; // acceleration * dt
        dvx += scale * dx;
        dvy += scale * dy;
        dvz += scale * dz;
    }

    vx[idx] += dvx;
    vy[idx] += dvy;
    vz[idx] += dvz;
}

__global__ void update_positions(int n,
                                 float* x,
                                 float* y,
                                 float* z,
                                 const float* vx,
                                 const float* vy,
                                 const float* vz,
                                 float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    x[idx] += vx[idx] * dt;
    y[idx] += vy[idx] * dt;
    z[idx] += vz[idx] * dt;
}

void copy_device_to_host(std::vector<Body>& bodies,
                         std::vector<float>& h_x,
                         std::vector<float>& h_y,
                         std::vector<float>& h_z,
                         std::vector<float>& h_vx,
                         std::vector<float>& h_vy,
                         std::vector<float>& h_vz,
                         const float* d_x,
                         const float* d_y,
                         const float* d_z,
                         const float* d_vx,
                         const float* d_vy,
                         const float* d_vz) {
    const int n = static_cast<int>(bodies.size());
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    CUDA_CHECK(cudaMemcpy(h_x.data(), d_x, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_z.data(), d_z, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vx.data(), d_vx, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vy.data(), d_vy, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vz.data(), d_vz, bytes, cudaMemcpyDeviceToHost));
    
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        bodies[i].set_x(h_x[i]);
        bodies[i].set_y(h_y[i]);
        bodies[i].set_z(h_z[i]);
        bodies[i].set_vx(h_vx[i]);
        bodies[i].set_vy(h_vy[i]);
        bodies[i].set_vz(h_vz[i]);
    }
}

void write_state(std::ostream& out, const std::vector<Body>& bodies, float t) {
    out << std::setw(12) << "t=:" << t << '\n';
    for (const auto& b : bodies) {
        out << b << '\n';
    }
    out << '\n';
}

int main(int argc, char* argv[]) {
    constexpr float YEAR = 365.25f * 24.0f * 60.0f * 60.0f ; // matches gravsim0.cpp
    constexpr float MONTH = 30.0f * 24.0f * 60.0f * 60.0f;

    const float dt = argc > 1 ? static_cast<float>(atof(argv[1])) : 100.0f; // timestep in seconds
    const float END = argc > 2 ? static_cast<float>(atof(argv[2])) * YEAR : YEAR;
    float print_interval = argc > 3 ? static_cast<float>(atof(argv[3])) : 1000.0f; // in steps

    std::vector<Body> bodies;
    bodies.emplace_back("Sun", 1.989e30, 0.0, -200.0, 0.0, 0.0, 0.0, 0.0);
    bodies.emplace_back("Earth", 5.97219e24, 149.59787e9, 0.0, 0.0, 0.0, 29784.8, 0.0);
    bodies.emplace_back("Mars", -6.39e23, 228e9, 0.0, 0.0, 0.0, -24130.8, 0.0);
    bodies.emplace_back("Ceres", -9.3839e20, 0.0, 449e9, 0.0, -16.9e3, 0.0, 0.0);

    std::ifstream in("planet_bodies_1M.dat");
    if (!in.is_open()) {
        std::cerr << "Error opening planet_bodies_100.dat\n";
        return 1;
    }

    std::string header;
    std::getline(in, header); // skip header line

    std::string name;
    double m, x, y, z, vx, vy, vz;
    while (in >> name >> m >> x >> y >> z >> vx >> vy >> vz) {
        bodies.push_back(Body(name, m, x, y, z, vx, vy, vz));
    }
    in.close();

    std::cerr << "Total bodies loaded: " << bodies.size() << '\n';

    const int n = static_cast<int>(bodies.size());
    if (n == 0) {
        std::cerr << "No bodies to simulate.\n";
        return 1;
    }

    std::vector<float> h_m(n), h_x(n), h_y(n), h_z(n), h_vx(n), h_vy(n), h_vz(n);
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        h_m[i] = bodies[i].get_m();
        h_x[i] = bodies[i].get_x();
        h_y[i] = bodies[i].get_y();
        h_z[i] = bodies[i].get_z();
        h_vx[i] = bodies[i].get_vx();
        h_vy[i] = bodies[i].get_vy();
        h_vz[i] = bodies[i].get_vz();
    }

    float *d_m = nullptr, *d_x = nullptr, *d_y = nullptr, *d_z = nullptr;
    float *d_vx = nullptr, *d_vy = nullptr, *d_vz = nullptr;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_m, bytes));
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMalloc(&d_z, bytes));
    CUDA_CHECK(cudaMalloc(&d_vx, bytes));
    CUDA_CHECK(cudaMalloc(&d_vy, bytes));
    CUDA_CHECK(cudaMalloc(&d_vz, bytes));

    CUDA_CHECK(cudaMemcpy(d_m, h_m.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z, h_z.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vx, h_vx.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vy, h_vy.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vz, h_vz.data(), bytes, cudaMemcpyHostToDevice));

    std::string output_filename = "output_cuda_1M.txt";
    std::ofstream outfile(output_filename);
    std::cerr << "Writing output to " << output_filename << '\n';
    if (!outfile.is_open()) {
        std::cerr << "Error opening output.txt for writing\n";
        return 1;
    }

    const double print_steps = print_interval;
    std::cerr << "dt=" << dt << "\tEND=" << (END / YEAR) << " years\tprint="
              << print_steps << '\n';
    print_interval *= dt; // convert to seconds to match timestep

    int threads = 256;
    int blocks = 24;
    std::cerr<< "Using " << blocks << " blocks of " << threads << " threads\n";
    auto start = std::chrono::high_resolution_clock::now();

    for (float t = 0.0f; t < END;) {
        copy_device_to_host(bodies, h_x, h_y, h_z, h_vx, h_vy, h_vz, d_x, d_y, d_z, d_vx, d_vy, d_vz);
        write_state(outfile, bodies, t);

        float next_print = t + print_interval;
        while (t < next_print && t < END) {
            float step_dt = std::min(dt, END - t);
            update_velocities<<<blocks, threads>>>(n, d_m, d_x, d_y, d_z, d_vx, d_vy, d_vz, step_dt);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            update_positions<<<blocks, threads>>>(n, d_x, d_y, d_z, d_vx, d_vy, d_vz, step_dt);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            t += step_dt;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    outfile << "\nExecution time: " << duration.count() << " ms\n";
    std::cerr << "Execution time: " << duration.count() << " ms\n";
    outfile.close();

    CUDA_CHECK(cudaFree(d_m));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));
    CUDA_CHECK(cudaFree(d_vx));
    CUDA_CHECK(cudaFree(d_vy));
    CUDA_CHECK(cudaFree(d_vz));

    return 0;
}

//100 [planets] in 1 year: 18430 ms ~= 18.43 seconds ~= 0.3 minutes
//1000 [planets] in 1 year: 93.366 seconds ~= 1.56 minutes
//10000 [planets] in 1 year: 891685 ms ~= 891.685 seconds ~= 14.86 minutes
//100000 [planets] in 1 year: 4483706 ms ~= 4483.706 seconds ~= 74.73 minutes
//1000000  [planets] in 1 year: 101517190 ms ~= 101517.19 seconds ~= 1691.95 minutes ~= 28.2 hours