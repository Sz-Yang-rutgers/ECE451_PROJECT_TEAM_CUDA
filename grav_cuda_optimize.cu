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
/**
 * 
 * This file is to generate a output file for the solar system positions after calcualtion and iteration for certain time
 * 
 */

constexpr float G = 5.67e-11f;       
constexpr float POSITION_CORRECTION = 1e-9f;   

#define CUDA_CHECK(expr)                                                             \
    do {                                                                             \
        cudaError_t __err = (expr);                                                  \
        if (__err != cudaSuccess) {                                                  \
            std::cerr << "CUDA error (" << #expr << "): "                           \
                      << cudaGetErrorString(__err) << '\n';                         \
            std::exit(EXIT_FAILURE);                                                 \
        }                                                                            \
    } while (0)

// kernels
__global__ void update_velocities(int n, const float* m, const float* x, const float* y, const float* z,
                                  float* vx, float* vy, float* vz, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float fx = 0.0f, fy = 0.0f, fz = 0.0f;
    float my_x = x[i];
    float my_y = y[i];
    float my_z = z[i];

    for (int j = 0; j < n; ++j) {
        if (i == j) continue; // Skip the body itself

        float dx = x[j] - my_x;
        float dy = y[j] - my_y;
        float dz = z[j] - my_z;
        
        // r^2 + POSITION_CORRECTION
        float dist_sq = dx * dx + dy * dy + dz * dz + POSITION_CORRECTION;
        float dist_inv = rsqrtf(dist_sq);       // 1/r
        float dist_inv3 = dist_inv * dist_inv * dist_inv; // 1/r^3

        float f = G * m[j] * dist_inv3;
        fx += f * dx;
        fy += f * dy;
        fz += f * dz;
    }
    
    vx[i] += fx * dt;
    vy[i] += fy * dt;
    vz[i] += fz * dt;
}

__global__ void update_positions(int n, float* x, float* y, float* z,
                                 const float* vx, const float* vy, const float* vz, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    x[i] += vx[i] * dt;
    y[i] += vy[i] * dt;
    z[i] += vz[i] * dt;
}

//main function
int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " [dt] [END] [print_interval]\n";
        return 1;
    }
    constexpr float YEAR = 365.25f * 24.0f * 60.0f * 60.0f ; // matches gravsim0.cpp
    //constexpr float MONTH = 30.0f * 24.0f * 60.0f * 60.0f;

    const float dt = argc > 1 ? static_cast<float>(atof(argv[1])) : 10.0f; // timestep in seconds
    const float END = argc > 2 ? static_cast<float>(atof(argv[2])) * YEAR : YEAR;
    const float print_interval = argc > 3 ? static_cast<float>(atof(argv[3])) : 100.0f; // in steps


 std::string input_filename = "planet_bodies_100.dat";

    // read from input
    std::ifstream infile(input_filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open input file '" << input_filename << "'\n";
        return 1;
    }

    int n = 0;
    std::string line;
    
    // Host Vectors (Structure of Arrays)
    std::vector<std::string> h_names;
    std::vector<float> h_m, h_x, h_y, h_z, h_vx, h_vy, h_vz;
    
    std::cout << "Reading from " << input_filename << "...\n";

    while (infile.peek() != EOF) {
        // Skip 't=...' headers if present (e.g., from a restart file)
        if (infile.peek() == 't') {
            std::getline(infile, line);
            continue;
        }

        std::string name;
        float mass, x, y, z, vx, vy, vz;
        // name, mass, x, y, z, vx, vy, vz
        if (infile >> name >> mass >> x >> y >> z >> vx >> vy >> vz) {
            h_names.push_back(name);
            h_m.push_back(mass);
            h_x.push_back(x);
            h_y.push_back(y);
            h_z.push_back(z);
            h_vx.push_back(vx);
            h_vy.push_back(vy);
            h_vz.push_back(vz);
        } else {
            // Skip malformed lines or whitespace
            infile.clear(); 
            std::getline(infile, line);
        }
    }
    infile.close();

    n = h_names.size();
    if (n == 0) {
        std::cerr << "Error: No bodies found in file.\n";
        return 1;
    }
    std::cout << "Loaded " << n << " bodies.\n";

    // allocate memories on device
    float *d_m, *d_x, *d_y, *d_z, *d_vx, *d_vy, *d_vz;
    size_t bytes = n * sizeof(float);
    
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

    //  allocate memories for t=0 to t=END calculation on GPU
    // Calculate max frames to allocate exact memory
    size_t total_frames = (size_t)ceil(END / print_interval) + 2; 
    size_t history_bytes = total_frames * n * sizeof(float);

    std::cout << "Allocating GPU memories: " 
              << (history_bytes * 3) / (1024.0 * 1024.0) << " MB... \n";
    std::cerr << "dt=" << dt << "\tEND=" << (END / YEAR) << " years\tprint="
              << print_interval << '\n';
          
    float *d_hist_x, *d_hist_y, *d_hist_z;
    CUDA_CHECK(cudaMalloc(&d_hist_x, history_bytes));
    CUDA_CHECK(cudaMalloc(&d_hist_y, history_bytes));
    CUDA_CHECK(cudaMalloc(&d_hist_z, history_bytes));
    std::cout << "Done.\n";

    // Setting up kernal parameters
    int threads = 256;
    int blocks = 256;
    int frame_count = 0;
    float t = 0.0f;
    std::cerr<< "Using " << blocks << " blocks of " << threads << " threads\n";    
    // Output file
    std::string output_filename = "output_cuda_optimized_100.txt";
    std::ofstream outfile(output_filename);
    outfile << std::scientific << std::setprecision(5);

    auto start = std::chrono::high_resolution_clock::now();

    while (t < END) {
        // Copy current d_x to d_hist_x
        size_t offset = (size_t)frame_count * n;
        
        
        CUDA_CHECK(cudaMemcpy(d_hist_x + offset, d_x, bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_hist_y + offset, d_y, bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_hist_z + offset, d_z, bytes, cudaMemcpyDeviceToDevice));
        
        frame_count++; //calculate how many frames needed based on time interval

        //update velocity and position, step foward based on time interval
        float next_print = t + print_interval;
        while (t < next_print && t < END) {
            float step_dt = std::min(dt, END - t);
            
            update_velocities<<<blocks, threads>>>(n, d_m, d_x, d_y, d_z, d_vx, d_vy, d_vz, step_dt);
            cudaDeviceSynchronize(); 
            update_positions<<<blocks, threads>>>(n, d_x, d_y, d_z, d_vx, d_vy, d_vz, step_dt);
            
            t += step_dt;
        }
    }

    // last frame needed
    size_t offset = (size_t)frame_count * n;
    CUDA_CHECK(cudaMemcpy(d_hist_x + offset, d_x, bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_hist_y + offset, d_y, bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_hist_z + offset, d_z, bytes, cudaMemcpyDeviceToDevice));
    frame_count++;

    CUDA_CHECK(cudaDeviceSynchronize()); // Wait for all GPU work to finish
    auto end = std::chrono::high_resolution_clock::now();

    // transfer all data back to CPU
    std::cout << "Simulation complete. Transferring " << frame_count << " frames to CPU...\n";
    
    std::vector<float> h_hist_x(total_frames * n);
    std::vector<float> h_hist_y(total_frames * n);
    std::vector<float> h_hist_z(total_frames * n);

    CUDA_CHECK(cudaMemcpy(h_hist_x.data(), d_hist_x, frame_count * n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_hist_y.data(), d_hist_y, frame_count * n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_hist_z.data(), d_hist_z, frame_count * n * sizeof(float), cudaMemcpyDeviceToHost));

    // write to a file
    std::cout << "Writing to disk...\n";
    std::cerr << "Writing output to " << output_filename << '\n';
    std::cerr << "Total bodies framed: " << h_names.size() << '\n';
    float write_t = 0.0f;
    
    for (int f = 0; f < frame_count; f++) {
        outfile << "t=:" << write_t << "\n";
        //don't need vx,vy,vz for output,only the relative positions and names
        for (int i = 0; i < n; i++) {
            size_t idx = f * n + i;
            outfile << h_names[i] << " "
                    << h_hist_x[idx] << " " 
                    << h_hist_y[idx] << " " 
                    << h_hist_z[idx] << "\n";
            
        }
        write_t += print_interval;
    }

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Execution time (Compute): " << duration.count() << " ms\n";
    outfile.close();

    // free memories
    CUDA_CHECK(cudaFree(d_m));
    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_y)); CUDA_CHECK(cudaFree(d_z));
    CUDA_CHECK(cudaFree(d_vx)); CUDA_CHECK(cudaFree(d_vy)); CUDA_CHECK(cudaFree(d_vz));
    CUDA_CHECK(cudaFree(d_hist_x)); CUDA_CHECK(cudaFree(d_hist_y)); CUDA_CHECK(cudaFree(d_hist_z));

    return 0;
}
//optimized version
//100 planet in 1 year 10 1 100: Execution time : 36136 ms ~= 36.136 seconds ~= 0.6 minutes
