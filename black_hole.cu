// blackhole_curved.cu
// CUDA ray marcher: black hole with approximate gravitational lensing
// (curved light paths) + 8 orbiting planets + glowing accretion ring.
// Outputs: blackhole_curved.ppm

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>
#include <fstream>
#include <iostream>

// ---------------- basic math ----------------

struct Vec3 {
    double x, y, z;

    __host__ __device__
    Vec3(double xx=0, double yy=0, double zz=0) : x(xx), y(yy), z(zz) {}

    __host__ __device__ Vec3 operator+(const Vec3& o) const { return Vec3(x+o.x, y+o.y, z+o.z); }
    __host__ __device__ Vec3 operator-(const Vec3& o) const { return Vec3(x-o.x, y-o.y, z-o.z); }
    __host__ __device__ Vec3 operator*(double s) const      { return Vec3(x*s, y*s, z*s); }
    __host__ __device__ Vec3 operator/(double s) const      { return Vec3(x/s, y/s, z/s); }
};

__host__ __device__
inline Vec3 operator*(double s, const Vec3& v) {
    return Vec3(v.x*s, v.y*s, v.z*s);
}

__host__ __device__
inline double dot(const Vec3& a, const Vec3& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__host__ __device__
inline Vec3 normalize(const Vec3& v) {
    double len2 = dot(v,v);
    if (len2 == 0.0) return v;
    double inv = 1.0 / sqrt(len2);
    return v * inv;
}

__host__ __device__
inline Vec3 clamp01(const Vec3& c) {
    auto f = [] __host__ __device__ (double v){ return v<0?0:(v>1?1:v); };
    return Vec3(f(c.x), f(c.y), f(c.z));
}

// --------------- scene types -----------------

struct Ray {
    Vec3 orig;
    Vec3 dir;
};

struct Sphere {
    Vec3 center;
    double radius;
    Vec3 color;
    double emissive;
    int isBlackHole;
};

// -------------- device helpers ----------------

// signed distance to sphere surface (negative inside)
__device__
double sphereDistance(const Vec3& p, const Sphere& s) {
    double dx = p.x - s.center.x;
    double dy = p.y - s.center.y;
    double dz = p.z - s.center.z;
    double d2 = dx*dx + dy*dy + dz*dz;
    return sqrt(d2) - s.radius;
}

__device__
Vec3 shadeHit(const Vec3& hitPos, const Sphere& obj,
              const Sphere* spheres, int nSpheres,
              const Vec3& lightPos)
{
    // black hole is handled outside
    Vec3 N = normalize(hitPos - obj.center);
    Vec3 L = normalize(lightPos - hitPos);

    double diff = dot(N,L);
    if (diff < 0.0) diff = 0.0;

    // simple shadow
    bool inShadow = false;
    Ray shadowRay;
    shadowRay.orig = hitPos + N*1e-3;
    shadowRay.dir  = L;

    const double SHADOW_MAX_DIST = 50.0;
    Vec3 pos = shadowRay.orig;
    Vec3 dir = shadowRay.dir;

    const double step = 0.05;
    for (int sIdx=0; sIdx<nSpheres && !inShadow; ++sIdx) {
        if (&spheres[sIdx] == &obj) continue;
    }

    // super simple: just check straight-line march for any hit
    for (int i=0; i<300 && !inShadow; ++i) {
        double minDist = 1e30;
        const Sphere* blocker = nullptr;
        for (int j=0; j<nSpheres; ++j) {
            if (&spheres[j] == &obj) continue;
            double d = sphereDistance(pos, spheres[j]);
            if (d < minDist) {
                minDist = d;
                blocker = &spheres[j];
            }
        }
        if (blocker && minDist < 0.0) {
            inShadow = true;
            break;
        }
        pos = pos + dir*step;
        if (dot(pos - shadowRay.orig, pos - shadowRay.orig) > SHADOW_MAX_DIST*SHADOW_MAX_DIST)
            break;
    }

    double ambient = 0.1;
    double li = inShadow ? ambient : ambient + diff;

    Vec3 col = obj.color * li;
    col = col + obj.color * obj.emissive;
    return clamp01(col);
}

// curved-ray integrator with approximate gravity around BH at origin
__device__
Vec3 traceRayCurved(const Ray& ray,
                    const Sphere* spheres, int nSpheres,
                    const Vec3& lightPos,
                    const Vec3& bhCenter, double bhRadius)
{
    const int   MAX_STEPS   = 600;
    const double STEP       = 0.03;   // move per integration step
    const double HIT_EPS    = 0.02;   // when we are "close enough" to surface
    const double MAX_RADIUS = 40.0;   // beyond this: space background
    const double GRAVITY_K  = 2.5;    // strength of bending (tweak for look)

    Vec3 pos = ray.orig;
    Vec3 dir = normalize(ray.dir);

    for (int i=0; i<MAX_STEPS; ++i) {
        // capture by event horizon
        Vec3 toBH = pos - bhCenter;
        double r2 = dot(toBH,toBH);
        double r  = sqrt(r2);
        if (r < bhRadius) {
            // fell into black hole
            return Vec3(0.0,0.0,0.0);
        }
        if (r > MAX_RADIUS) {
            // escaped far: background gradient
            double t = 0.5*(dir.y + 1.0);
            Vec3 top(0.02,0.02,0.05);
            Vec3 bottom(0.0,0.0,0.0);
            return (1.0-t)*top + t*bottom;
        }

        // find nearest sphere distance at current position
        double minDist = 1e30;
        const Sphere* hitObj = nullptr;
        for (int s=0; s<nSpheres; ++s) {
            double d = sphereDistance(pos, spheres[s]);
            if (d < minDist) {
                minDist = d;
                hitObj  = &spheres[s];
            }
        }

        if (hitObj && minDist < HIT_EPS) {
            // hit something
            if (hitObj->isBlackHole) {
                return Vec3(0.0,0.0,0.0);
            } else {
                return shadeHit(pos, *hitObj, spheres, nSpheres, lightPos);
            }
        }

        // approximate gravitational acceleration toward BH
        // a ~ -k * (r_hat / r^2)
        if (r > bhRadius + 1e-3) {
            double inv_r3 = 1.0 / (r2*r);
            Vec3 acc = (-GRAVITY_K) * inv_r3 * toBH;  // points inward
            dir = normalize(dir + acc * STEP);
        }

        // advance ray
        pos = pos + dir * STEP;
    }

    // if we exit loop without hit: background
    double t = 0.5*(dir.y + 1.0);
    Vec3 top(0.02,0.02,0.05);
    Vec3 bottom(0.0,0.0,0.0);
    return (1.0-t)*top + t*bottom;
}

__global__
void renderKernel(Vec3* fb,
                  int W, int H,
                  Vec3 camPos,
                  double scale, double aspect,
                  const Sphere* spheres, int nSpheres,
                  Vec3 lightPos,
                  Vec3 bhCenter, double bhRadius)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    double u = (2.0*(x+0.5)/double(W) - 1.0) * aspect * scale;
    double v = (1.0 - 2.0*(y+0.5)/double(H)) * scale;

    Ray ray;
    ray.orig = camPos;
    ray.dir  = Vec3(u, v, -1.0);

    Vec3 col = traceRayCurved(ray, spheres, nSpheres, lightPos, bhCenter, bhRadius);
    fb[y*W + x] = col;
}

// ---------------- host side ----------------

void savePPM(const std::string& fname,
             const std::vector<Vec3>& fb,
             int W, int H)
{
    std::ofstream ofs(fname, std::ios::out | std::ios::binary);
    ofs << "P3\n" << W << " " << H << "\n255\n";
    for (auto &c : fb) {
        Vec3 cc = clamp01(c);
        int r = int(255.99 * cc.x);
        int g = int(255.99 * cc.y);
        int b = int(255.99 * cc.z);
        ofs << r << " " << g << " " << b << "\n";
    }
}

int main() {
    const int W = 800;
    const int H = 800;

    Vec3 camPos(0.0, 0.0, 6.0);
    double fov = 35.0; // slightly narrower to emphasize lensing
    double aspect = double(W)/H;
    double scale = tan(fov * 0.5 * M_PI / 180.0);

    Vec3 lightPos(5.0, 5.0, 5.0);
    Vec3 bhCenter(0.0, 0.0, 0.0);
    double bhRadius = 1.1; // event horizon radius

    // ----- build scene on host -----

    std::vector<Sphere> h_spheres;

    // Black hole sphere (just for geometry; shading returns black)
    {
        Sphere bh;
        bh.center = bhCenter;
        bh.radius = bhRadius;
        bh.color  = Vec3(0.0,0.0,0.0);
        bh.emissive = 0.0;
        bh.isBlackHole = 1;
        h_spheres.push_back(bh);
    }

    // Accretion disk made of glowing mini-spheres
    int ringSegments = 96;
    double diskRadius = 1.6;
    for (int i=0; i<ringSegments; ++i) {
        double a = 2.0 * M_PI * i / ringSegments;
        double x = diskRadius * cos(a);
        double y = 0.25 * sin(2.0*a);   // slight vertical wobble
        double z = diskRadius * sin(a);
        Sphere s;
        s.center = Vec3(x,y,z);
        s.radius = 0.10;
        s.color  = Vec3(1.0, 0.8, 0.3);
        s.emissive = 0.9;   // bright glowing disk
        s.isBlackHole = 0;
        h_spheres.push_back(s);
    }

    // Planets arranged in a ring further out
    int NPLANETS = 8;
    double orbitRadius = 3.5;
    double planetRadius = 0.35;

    for (int i=0; i<NPLANETS; ++i) {
        double a = 2.0 * M_PI * i / NPLANETS;
        double x = orbitRadius * cos(a);
        double y = orbitRadius * 0.1 * sin(2.0*a);
        double z = orbitRadius * sin(a);

        double r = 0.4 + 0.6 * 0.5*(cos(a)+1.0);
        double g = 0.3 + 0.7 * 0.5*(cos(a+2.0)+1.0);
        double b = 0.3 + 0.7 * 0.5*(cos(a+4.0)+1.0);

        Sphere p;
        p.center = Vec3(x,y,z);
        p.radius = planetRadius;
        p.color  = Vec3(r,g,b);
        p.emissive = 0.05; // slight glow so they pop
        p.isBlackHole = 0;
        h_spheres.push_back(p);
    }

    int nSpheres = (int)h_spheres.size();

    // ----- copy to device -----
    Sphere* d_spheres = nullptr;
    Vec3* d_fb = nullptr;
    size_t sphBytes = nSpheres * sizeof(Sphere);
    size_t fbBytes  = W*H*sizeof(Vec3);

    cudaMalloc(&d_spheres, sphBytes);
    cudaMalloc(&d_fb, fbBytes);
    cudaMemcpy(d_spheres, h_spheres.data(), sphBytes, cudaMemcpyHostToDevice);

    dim3 block(16,16);
    dim3 grid((W+block.x-1)/block.x, (H+block.y-1)/block.y);

    // ----- render -----
    renderKernel<<<grid,block>>>(d_fb, W,H,
                                 camPos, scale, aspect,
                                 d_spheres, nSpheres,
                                 lightPos,
                                 bhCenter, bhRadius);
    cudaDeviceSynchronize();

    // ----- copy back and save -----
    std::vector<Vec3> h_fb(W*H);
    cudaMemcpy(h_fb.data(), d_fb, fbBytes, cudaMemcpyDeviceToHost);

    savePPM("blackhole_curved.ppm", h_fb, W,H);
    std::cout << "Rendered blackhole_curved.ppm\n";

    cudaFree(d_fb);
    cudaFree(d_spheres);
    return 0;
}
