#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <chrono>
#include "Body.h"
#include <omp.h>
constexpr double G = 5.67E-11; // universal gravitational constant

/*
	Gravitational force F = G m1 m2 / r^2
	Every body exerts a force on every other body.

	find the problems in this code
	compiler doesn't do well with floating point

	c1*x*c2 = (c1*c2)*x
	x*c1*c2 = x*(c1*c2)
	
	
	*/

/*
bodies [0] [1] [2] [3] ...
      /     |
    /       \
body0       body1

*/
std::vector<Body> bodies;

// step all bodies forward in time
void step_forward_all(double dt) { 
	#pragma omp parallel for
	for (int i = 0; i < bodies.size(); i++) {
		Body b = bodies[i];
		#pragma omp simd
		for (int j = 0; j < bodies.size(); j++) {
			if (i == j)
				continue;
			Body other = bodies[j];
			double r = b.dist(other);
			double F = G * b.get_m() * other.get_m() / (r*r);
			double a = F / b.get_m();
			b.set_vx(b.get_vx()-b.d2x(other,a)); 
			b.set_vy(b.get_vy()-b.d2y(other,a));
			b.set_vz(b.get_vz()-b.d2z(other,a));
			
		}
	}
	// step forward after we calculate everyone's velocity
	for (int i = 0; i < bodies.size(); i++) {
		bodies[i].step_forward(dt);
	}
}



void print(double t) {
	std::cout << std::setw(12) <<"t=:" <<t << '\n';
	#pragma omp parallel for
	for (int i = 0; i < bodies.size(); i++) {
		#pragma omp critical
		std::cout << bodies[i] << '\n';
	}
	std::cout << '\n';
}

int main(int argc, char* argv[]) {
	constexpr double YEAR = 365.25 * 24 * 60 * 60*2;
	constexpr double MONTH = 30 * 24 * 60 * 60;
	const double dt = argc > 1 ? atof(argv[1]) : 100; // 100 second default timestep
	const double END = argc > 2 ? atof(argv[2]) * YEAR : YEAR;
	double print_interval = argc > 3 ? atof(argv[3]) : 1000;

	bodies.push_back(Body("Sun", 1.989e30, 0,-200,0, 0,0,0));
	bodies.push_back(Body("Earth", 5.97219e24, 149.59787e9,0,0, 0,29784.8,0));
	bodies.push_back(Body("Mars", -6.39e23, 228e9,0,0, 0,-24130.8,0));
	bodies.push_back(Body("Ceres", -9.3839e20, 0, 449e9,0, -16.9e3,0,0));
    
	std::ifstream in("bodies_1M.dat");
	if (!in.is_open()) {
		std::cerr << "Error opening bodies_1M.dat\n";
		return 1;
	}

std::string header;
std::getline(in, header);  // Skip header line
std::string name;
double m, x, y, z, vx, vy, vz;

while (in >> name >> m >> x >> y >> z >> vx >> vy >> vz) {
    bodies.push_back(Body(name, m, x, y, z, vx, vy, vz));
}
in.close();

	std::cerr << "Total bodies loaded: " << bodies.size() << '\n';
	std::ofstream outfile("output.txt");
	std::cerr << "dt=" << dt << "\tEND=" << (END/YEAR) << " years \tprint=" << print_interval << '\n';
	print_interval *= dt;
	double next_print;
	auto start = std::chrono::high_resolution_clock::now();
	for (double t = 0; t < END;) {
		outfile << std::setw(12) <<"t=:" << t << '\n';
		for (int i = 0; i < bodies.size(); i++) {
			outfile << bodies[i] << '\n';
		}
		outfile << '\n';
		next_print = t + print_interval;
		for ( ; t < next_print; t += dt) {
			step_forward_all(dt);
		}
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	outfile << "\nExecution time: " << duration.count() << " ms\n";
	std::cerr << "Execution time: " << duration.count() << " ms\n";
	outfile.close();
}
