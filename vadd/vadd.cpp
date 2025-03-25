/*******************************************************************************
Vendor: Xilinx
Associated Filename: vadd.cpp
Purpose: VITIS vector addition

*******************************************************************************
Copyright (C) 2019 XILINX, Inc.

This file contains confidential and proprietary information of Xilinx, Inc. and
is protected under U.S. and international copyright and other intellectual
property laws.

DISCLAIMER
This disclaimer is not a license and does not grant any rights to the materials
distributed herewith. Except as otherwise provided in a valid license issued to
you by Xilinx, and to the maximum extent permitted by applicable law:
(1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND WITH ALL FAULTS, AND XILINX
HEREBY DISCLAIMS ALL WARRANTIES AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY,
INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT, OR
FITNESS FOR ANY PARTICULAR PURPOSE; and (2) Xilinx shall not be liable (whether
in contract or tort, including negligence, or under any other theory of
liability) for any loss or damage of any kind or nature related to, arising under
or in connection with these materials, including for any direct, or any indirect,
special, incidental, or consequential loss or damage (including loss of data,
profits, goodwill, or any type of loss or damage suffered as a result of any
action brought by a third party) even if such damage or loss was reasonably
foreseeable or Xilinx had been advised of the possibility of the same.

CRITICAL APPLICATIONS
Xilinx products are not designed or intended to be fail-safe, or for use in any
application requiring fail-safe performance, such as life-support or safety
devices or systems, Class III medical devices, nuclear facilities, applications
related to the deployment of airbags, or any other applications that could lead
to death, personal injury, or severe property or environmental damage
(individually and collectively, "Critical Applications"). Customer assumes the
sole risk and liability of any use of Xilinx products in Critical Applications,
subject only to applicable laws and regulations governing limitations on product
liability.

THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS PART OF THIS FILE AT
ALL TIMES.

*******************************************************************************/

#define OCL_CHECK(error, call)                                                                   \
    call;                                                                                        \
    if (error != CL_SUCCESS) {                                                                   \
        printf("%s:%d Error calling " #call ", error code is: %d\n", __FILE__, __LINE__, error); \
        exit(EXIT_FAILURE);                                                                      \
    }

#include "vadd.h"
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <cmath>
#include <random> 

// Default simulation parameters
static const int PARTICLES = 64;
static const float DEFAULT_EPSILON = 1.0f;  // Depth of potential well
static const float DEFAULT_SIGMA = 1.0f;    // Distance at which potential is zero
static const float DEFAULT_CUTOFF = 2.5f * DEFAULT_SIGMA;  // Typical cutoff distance

// CPU implementation of Lennard-Jones force calculation for verification
void calculate_lj_forces_cpu(
    std::vector<particle_position_t, aligned_allocator<particle_position_t>>& positions,
    std::vector<force_vector_t, aligned_allocator<force_vector_t>>& forces
) {
    int num_particles = positions.size();
    
    // Initialize forces to zero
    for (int i = 0; i < num_particles; i++) {
        forces[i] = {0.0f, 0.0f, 0.0f};
    }
    
    // Calculate forces between all pairs
    for (int i = 0; i < num_particles; i++) {
        for (int j = i+1; j < num_particles; j++) {
            
            // Calculate distance between particles
            float dx = positions[j].x - positions[i].x;
            float dy = positions[j].y - positions[i].y;
            float dz = positions[j].z - positions[i].z;
            float r_squared = dx*dx + dy*dy + dz*dz;
            
            // Skip if particles are beyond cutoff
            if (r_squared > DEFAULT_CUTOFF * DEFAULT_CUTOFF) continue;
            // Skip if particles are very close
            if (r_squared <= 0.0001f) continue;
            
            float r = std::sqrt(r_squared);
            float inv_r = 1.0f / r;
            
            // Calculate LJ force magnitude
            float inv_r6 = std::pow(inv_r, 6);
            float inv_r12 = inv_r6 * inv_r6;
            float sigma6 = std::pow(DEFAULT_SIGMA, 6);
            float sigma12 = sigma6 * sigma6;
            float force_mag = 24.0f * DEFAULT_EPSILON * (2.0f * sigma12 * inv_r12 - sigma6 * inv_r6) * inv_r * inv_r;
            
            // Accumulate force
            forces[i].x += force_mag * dx;
            forces[i].y += force_mag * dy;
            forces[i].z += force_mag * dz;

            forces[j].x -= force_mag * dx; 
            forces[j].y -= force_mag * dy;
            forces[j].z -= force_mag * dz;
        }
    }
}

// Initialize particles in a simple cubic lattice
void initialize_particles(
    std::vector<particle_position_t, aligned_allocator<particle_position_t>>& positions
) {
    int num_particles = positions.size();

    // Calculate cubic dimensions to fit all particles
    int particles_per_side = std::ceil(std::cbrt(num_particles));
    float spacing = 1.5f * DEFAULT_SIGMA;  // Reasonable spacing to avoid huge initial forces
    
    // Place particles in a cubic lattice
    int count = 0;
    for (int x = 0; x < particles_per_side && count < num_particles; x++) {
        for (int y = 0; y < particles_per_side && count < num_particles; y++) {
            for (int z = 0; z < particles_per_side && count < num_particles; z++) {
                positions[count].x = x * spacing;
                positions[count].y = y * spacing;
                positions[count].z = z * spacing;
                count++;
            }
        }
    }
    
    // Add small random displacements to avoid perfect symmetry
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-0.01f, 0.01f);
    
    for (int i = 0; i < num_particles; i++) {
        positions[i].x += dist(gen);
        positions[i].y += dist(gen);
        positions[i].z += dist(gen);
    }
}

static const std::string error_message =
    "Error: Result mismatch:\n"
    "i = %d CPU result = %d Device result = %d\n";

int main(int argc, char* argv[]) {
    // TARGET_DEVICE macro needs to be passed from gcc command line
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <xclbin>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string xclbinFilename = argv[1];

    // Compute the size of array in bytes
    size_t positions_size_bytes = PARTICLES * sizeof(particle_position_t);
    size_t forces_size_bytes = PARTICLES * sizeof(force_vector_t);

    // Allocate aligned memory for particle data
    std::vector<particle_position_t, aligned_allocator<particle_position_t>> 
        particle_positions(PARTICLES);
    std::vector<force_vector_t, aligned_allocator<force_vector_t>> 
        particle_forces_fpga(PARTICLES);
    std::vector<force_vector_t, aligned_allocator<force_vector_t>> 
        particle_forces_cpu(PARTICLES);
    
    // Initialize particles
    initialize_particles(particle_positions);

    std::vector<cl::Device> devices;
    cl_int err;
    cl::Context context;
    cl::CommandQueue q;
    cl::Kernel krnl_vector_add;
    cl::Program program;
    std::vector<cl::Platform> platforms;
    bool found_device = false;

    // traversing all Platforms To find Xilinx Platform and targeted
    // Device in Xilinx Platform
    cl::Platform::get(&platforms);
    for (size_t i = 0; (i < platforms.size()) & (found_device == false); i++) {
        cl::Platform platform = platforms[i];
        std::string platformName = platform.getInfo<CL_PLATFORM_NAME>();
        if (platformName == "Xilinx") {
            devices.clear();
            platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
            if (devices.size()) {
                found_device = true;
                break;
            }
        }
    }
    if (found_device == false) {
        std::cout << "Error: Unable to find Target Device " << std::endl;
        return EXIT_FAILURE;
    }

    // Attempts to open the XCLBIN file to verify it exists
    std::cout << "INFO: Reading " << xclbinFilename << std::endl;
    FILE* fp;
    if ((fp = fopen(xclbinFilename.c_str(), "r")) == nullptr) {
        printf("ERROR: %s xclbin not available please build\n", xclbinFilename.c_str());
        exit(EXIT_FAILURE);
    }
    // Load xclbin
    std::cout << "Loading: '" << xclbinFilename << "'\n";
    std::ifstream bin_file(xclbinFilename, std::ifstream::binary);
    bin_file.seekg(0, bin_file.end);
    unsigned nb = bin_file.tellg();
    bin_file.seekg(0, bin_file.beg);
    char* buf = new char[nb];
    bin_file.read(buf, nb);

    // Creating Program from Binary File
    cl::Program::Binaries bins;
    bins.push_back({buf, nb});
    bool valid_device = false;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        // Creating Context and Command Queue for selected Device
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            OCL_CHECK(err, krnl_vector_add = cl::Kernel(program, "krnl_vadd", &err));
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

    // These commands will allocate memory on the Device. The cl::Buffer objects can
    // be used to reference the memory locations on the device.
    OCL_CHECK(err, cl::Buffer buffer_input(context, CL_MEM_READ_ONLY, positions_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_result(context, CL_MEM_WRITE_ONLY, forces_size_bytes, NULL, &err));

    // set the kernel Arguments
    int narg = 0;
    OCL_CHECK(err, err = krnl_vector_add.setArg(narg++, buffer_input));
    OCL_CHECK(err, err = krnl_vector_add.setArg(narg++, buffer_result));
    OCL_CHECK(err, err = krnl_vector_add.setArg(narg++, PARTICLES));

    // We then need to map our OpenCL buffers to get the pointers
    particle_position_t* ptr_input;
    force_vector_t* ptr_result;
    OCL_CHECK(err, ptr_input = (particle_position_t*)q.enqueueMapBuffer(buffer_input, CL_TRUE, CL_MAP_WRITE, 0, positions_size_bytes, NULL, NULL, &err));
    OCL_CHECK(err, ptr_result = (force_vector_t*)q.enqueueMapBuffer(buffer_result, CL_TRUE, CL_MAP_READ, 0, forces_size_bytes, NULL, NULL, &err));

    // Copy input data to mapped memory
    memcpy(ptr_input, particle_positions.data(), positions_size_bytes);

    // Data will be migrated to kernel space
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_input}, 0 /* 0 means from host to device*/));

    // Launch the Kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl_vector_add));

    // The result of the previous kernel execution will need to be retrieved in
    // order to view the results. This call will transfer the data from FPGA to
    // source_results vector
    OCL_CHECK(err, q.enqueueMigrateMemObjects({buffer_result}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, q.finish());

    // Copy results back to host vectors
    memcpy(particle_forces_fpga.data(), ptr_result, forces_size_bytes);

    // Verify the result
    calculate_lj_forces_cpu(particle_positions, particle_forces_cpu);
    float tolerance = 1e-3f;  // Tolerance for floating-point comparison
    float max_diff = 0.0f;
    int max_diff_idx = -1;

    bool match = true;
    for (int i = 0; i < PARTICLES; i++) {
        float fx_diff = std::abs(particle_forces_fpga[i].x - particle_forces_cpu[i].x);
        float fy_diff = std::abs(particle_forces_fpga[i].y - particle_forces_cpu[i].y);
        float fz_diff = std::abs(particle_forces_fpga[i].z - particle_forces_cpu[i].z);
        
        float diff = std::max(std::max(fx_diff, fy_diff), fz_diff);
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = i;
        }
        
        float cpu_magnitude = std::sqrt(
            particle_forces_cpu[i].x * particle_forces_cpu[i].x +
            particle_forces_cpu[i].y * particle_forces_cpu[i].y +
            particle_forces_cpu[i].z * particle_forces_cpu[i].z
        );
        
        // Use relative error for larger values, absolute for small values
        if (cpu_magnitude > 1e-5f) {
            float rel_error = diff / cpu_magnitude;
            if (rel_error > tolerance) {
                match = false;
            }
        } else if (diff > tolerance) {
            match = false;
        }
    }
    
    if (!match && max_diff_idx >= 0) {
        std::cout << "Maximum difference at particle " << max_diff_idx << ":" << std::endl;
        std::cout << "  FPGA: (" << particle_forces_fpga[max_diff_idx].x << ", " 
                  << particle_forces_fpga[max_diff_idx].y << ", " 
                  << particle_forces_fpga[max_diff_idx].z << ")" << std::endl;
        std::cout << "  CPU:  (" << particle_forces_cpu[max_diff_idx].x << ", " 
                  << particle_forces_cpu[max_diff_idx].y << ", " 
                  << particle_forces_cpu[max_diff_idx].z << ")" << std::endl;
        std::cout << "  Max difference: " << max_diff << std::endl;
    }

    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_input, ptr_input));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_result, ptr_result));
    OCL_CHECK(err, err = q.finish());

    std::cout << "TEST " << (match ? "PASSED" : "FAILED") << std::endl;
    return (match ? EXIT_SUCCESS : EXIT_FAILURE);
}
