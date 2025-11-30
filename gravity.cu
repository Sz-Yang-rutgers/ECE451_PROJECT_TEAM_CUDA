#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <thrust/device_vector.h>
#include <fstream>
#inlucde <omp.h>


#define G 6.67430e-11f // Gravitational constant