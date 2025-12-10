#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>

#include <cuda_runtime.h>
/**
 * This cuda file is to implement a simple ray tracing renderer
 * Output images of a solar system
 * 
 */

//use following command to generate a video from ppm frames
//ffmpeg -framerate 30 -i frame_%d.ppm -c:v libx264 -pix_fmt yuv420p simulation_solar_system.mp4

// resolution
#define WIDTH 1980
#define HEIGHT 1080
#define OUTPUT_FILENAME_BASE "frame_"

// CUDA Error Helper
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s line %d\n", cudaGetErrorString(err), __LINE__); \
            exit(1); \
        } \
    } while (0)

// float3_ structure

struct float3_ { float x, y, z; };

//sphere structure
struct Sphere {
    float x, y, z;
    float radius;
    float r, g, b;
    float sun_radiation;   // 1.0 for normal shading, 0.0 for sun
    float emission;        // >0 for sun emissive, 0 for planets
};

// helper device functions

__device__ float3_ sub(float3_ a, float3_ b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
__device__ float dot(float3_ a, float3_ b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

__device__ float3_ normalize(float3_ v) {
    float len = sqrtf(dot(v, v));
    if (len < 1e-6f) return {0,0,0};
    return {v.x / len, v.y / len, v.z / len};
}

__device__ float intersect_sphere(float3_ ray_o, float3_ ray_d, Sphere s) {
    float3_ oc = sub(ray_o, {s.x, s.y, s.z});
    float b = dot(oc, ray_d);
    float c = dot(oc, oc) - s.radius * s.radius;
    float h = b * b - c;
    if (h < 0.0f) return -1.0f;
    return -b - sqrtf(h);
}

// kernel to render the scene

__global__ void render_kernel(uchar4* output, int width, int height,
                              const Sphere* spheres, int num_spheres,
                              float3_ cam_o, float3_ cam_target, float fov_scale)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    // camera ray
    float u = (2.0f * x / width) - 1.0f;
    float v = (2.0f * y / height) - 1.0f;
    u *= (float)width / height; // aspect ratio

    float3_ forward = normalize(sub(cam_target, cam_o));
    
    float3_ right = normalize({forward.y, -forward.x, 0.0f});
    float3_ true_up = { -right.y*forward.z, right.x*forward.z, right.x*forward.y - right.y*forward.x };

    float3_ ray_dir;
    ray_dir.x = forward.x + u * right.x * fov_scale + v * true_up.x * fov_scale;
    ray_dir.y = forward.y + u * right.y * fov_scale + v * true_up.y * fov_scale;
    ray_dir.z = forward.z + u * right.z * fov_scale + v * true_up.z * fov_scale;
    ray_dir = normalize(ray_dir);

    //find closest intersection
    float closest_t = 1e20f;
    int hit_idx = -1;

    for (int i = 0; i < num_spheres; i++) {
        float t = intersect_sphere(cam_o, ray_dir, spheres[i]);
        if (t > 0.0f && t < closest_t) {
            closest_t = t;
            hit_idx = i;
        }
    }

   //shading (sun is the origin, at (0,0,0))
float r = 0.0f, g = 0.0f, b = 0.0f;

if (hit_idx != -1) {
    Sphere s = spheres[hit_idx];

    float3_ hit_pos = {cam_o.x + closest_t * ray_dir.x,
                       cam_o.y + closest_t * ray_dir.y,
                       cam_o.z + closest_t * ray_dir.z};

    float3_ normal   = normalize(sub(hit_pos, {s.x, s.y, s.z}));
    float3_ lightDir = normalize(sub({0.0f, 0.0f, 0.0f}, hit_pos)); // sun at origin

    float diff    = fmaxf(dot(normal, lightDir), 0.0f);
    float ambient = 0.1f;

    float lightTerm = s.sun_radiation * diff + ambient;
    float brightness = lightTerm + s.emission;

    r = s.r * brightness;
    g = s.g * brightness;
    b = s.b * brightness;

    //  make sure r,g,b are between 0 and 1
    r = fminf(fmaxf(r, 0.0f), 1.0f);
    g = fminf(fmaxf(g, 0.0f), 1.0f);
    b = fminf(fmaxf(b, 0.0f), 1.0f);
}

output[idx] = make_uchar4(
    (unsigned char)(r * 255),
    (unsigned char)(g * 255),
    (unsigned char)(b * 255),
    255);
}

std::vector<std::vector<Sphere>> all_frames;
float max_r = 0.0f;
float max_z = 0.0f;
float z_stretch = 1.0f;

/*
 * Load Data from Text File
 * New behavior:
 *   - At each 't=:' (new frame), pre-create Sun as the FIRST sphere in current_frame.
 *   - Any line with name == "Sun" in the file is ignored (we already have a sun).
 */
void load_data(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: " << filename << " not found." << std::endl;
        exit(1);
    }

    std::string line;
    std::vector<Sphere> current_frame;
    bool inside_frame = false;

    std::cout << "Loading Data From: " << filename << std::endl;

    while (std::getline(infile, line)) {
        if (line.find("t=:") != std::string::npos) {
            // start a new frame
            if (!current_frame.empty()) {
                all_frames.push_back(current_frame);
            }
            current_frame.clear();

            // Pre-create SUN at origin (no shading logic here, just position)
            Sphere sun;
            sun.x = 0.0f; sun.y = 0.0f; sun.z = 0.0f;
            sun.radius   = 0.0f;    
            sun.r = sun.g = sun.b = 0.0f;
            sun.sun_radiation  = 0.0f;    
            sun.emission = 0.0f;
            current_frame.emplace_back(sun);

            inside_frame = true;
            continue;
        }

        if (!inside_frame) {
            // Skip lines before the first t=:
            continue;
        }

        std::stringstream ss(line);
        std::string name;
        float x, y, z;
        if (ss >> name >> x >> y >> z) {
            // don't read sun from line – sun already created for index 0 in the array
            if (name == "Sun")
                continue;

            Sphere s;
            s.x = x; s.y = y; s.z = z;
            s.radius   = 0.0f;  
            s.r = s.g = s.b = 0.0f;
            s.sun_radiation  = 0.0f;
            s.emission = 0.0f;

            max_r = std::max(max_r, std::abs(x));
            max_r = std::max(max_r, std::abs(y));
            max_z = std::max(max_z, std::abs(z));

            current_frame.emplace_back(s);
        }
    }

    if (!current_frame.empty()) {
        all_frames.push_back(current_frame);
    }

    if (max_z == 0.0f) max_z = 1.0f;
    float ratio = max_z / max_r;
    if (ratio < 0.0001f) ratio = 0.0001f;
    z_stretch = 0.3f / ratio;
    if (z_stretch < 1.0f) z_stretch = 1.0f;

    std::cout << "Loaded " << all_frames.size()
              << " frames. Z-axis scaling: " << z_stretch << "x" << std::endl;
}

// Save image as PPM
void save_ppm(const std::string& filename, const std::vector<uchar4>& buffer, int w, int h) {
    std::ofstream ofs(filename, std::ios::binary);
    ofs << "P6\n" << w << " " << h << "\n255\n";
    for (int i = 0; i < w * h; ++i) {
        uchar4 p = buffer[i];
        ofs.put(p.x); // R
        ofs.put(p.y); // G
        ofs.put(p.z); // B
    }
    ofs.close();
}

int main(int argc, char** argv) {
    std::string filename = "output_cuda_optimized_100.txt";
    if (argc > 1) filename = argv[1];

    load_data(filename);

    // allocate Memory
    uchar4* d_output;
    Sphere* d_spheres;

    size_t img_bytes = WIDTH * HEIGHT * sizeof(uchar4);

    size_t max_spheres = 0;
    for (const auto& f : all_frames) max_spheres = std::max(max_spheres, f.size());

    CUDA_CHECK(cudaMalloc(&d_output, img_bytes));
    CUDA_CHECK(cudaMalloc(&d_spheres, max_spheres * sizeof(Sphere)));

    std::vector<uchar4> h_output(WIDTH * HEIGHT);

    dim3 blockSize(16, 16);//optimized for 3070Ti
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x,
                  (HEIGHT + blockSize.y - 1) / blockSize.y);
    std::cerr<< "Using Threads:" << blockSize.x* blockSize.y << " blocks of " << gridSize.x * gridSize.y << " threads\n"; 
    float cam_dist  = max_r * 2.5f;
    float cam_theta = 0.0f;
    
    for (int i = 0; i < (int)all_frames.size(); ++i) {
        std::vector<Sphere> frame = all_frames[i];

        // Prepare frame data (Apply Z stretch & per-sphere params)
        #pragma omp parallel for
        for (int j = 0; j < (int)frame.size(); ++j) {
            Sphere& s = frame[j];

            // Scale Z-axis
            s.z *= z_stretch;

            if (j == 0) {
                // j == 0 is Sun (created in load_data)
                s.radius = max_r * 0.05f;
                s.r = 1.0f; s.g = 0.8f; s.b = 0.0f;

                s.sun_radiation  = 0.0f;   // no shading for sun
                s.emission = 5.0f;   // emission light for sun
            } else {
                // planet rendering
                s.radius = max_r * 0.008f;
                s.r = 0.0f; s.g = 1.0f; s.b = 1.0f;

                s.sun_radiation  = 1.0f; // normal sun->object shading
                s.emission = 0.0f;
            }
        }

        // copy struct to device
        CUDA_CHECK(cudaMemcpy(d_spheres, frame.data(),
                              frame.size() * sizeof(Sphere),
                              cudaMemcpyHostToDevice));

        // update camera position for orbits
        cam_theta += 0.005f;
        float3_ cam_pos = { cosf(cam_theta)*cam_dist,
                            sinf(cam_theta)*cam_dist,
                            max_r * 0.5f };
        float3_ target = {0,0,0};

        // ray tracing kernel calling
        render_kernel<<<gridSize, blockSize>>>(
            d_output, WIDTH, HEIGHT,
            d_spheres, (int)frame.size(),
            cam_pos, target, 0.6f
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        // copy image data back to host
        CUDA_CHECK(cudaMemcpy(h_output.data(), d_output,
                              img_bytes, cudaMemcpyDeviceToHost));

        // write out to file and save as ppm file
        std::stringstream ss;
        ss << OUTPUT_FILENAME_BASE << i << ".ppm";
        save_ppm(ss.str(), h_output, WIDTH, HEIGHT);

        if (i % 10 == 0)
            std::cout << "Rendered frame " << i << " / " << all_frames.size() << "\r" << std::flush;
    }

    std::cout << "\nDone! Images saved as frame_X.ppm" << std::endl;

    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_spheres));
    return 0;
}
