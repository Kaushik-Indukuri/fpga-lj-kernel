/*******************************************************************************
Vendor: Xilinx
Associated Filename: vadd.h
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

/*******************************************************************************
Description:

    This example uses the load/compute/store coding style which is generally
    the most efficient for implementing kernels using HLS. The load and store
    functions are responsible for moving data in and out of the kernel as
    efficiently as possible. The core functionality is decomposed across one
    of more compute functions. Whenever possible, the compute function should
    pass data through HLS streams and should contain a single set of nested loops.

    HLS stream objects are used to pass data between producer and consumer
    functions. Stream read and write operations have a blocking behavior which
    allows consumers and producers to synchronize with each other automatically.

    The dataflow pragma instructs the compiler to enable task-level pipelining.
    This is required for to load/compute/store functions to execute in a parallel
    and pipelined manner. Here the kernel loads, computes and stores NUM_WORDS integer values per
    clock cycle and is implemented as below:
                                       _____________
                                      |             |<----- Input Vector 1 from Global Memory
                                      |  load_input |       __
                                      |_____________|----->|  |
                                       _____________       |  | in1_stream
Input Vector 2 from Global Memory --->|             |      |__|
                               __     |  load_input |        |
                              |  |<---|_____________|        |
                   in2_stream |  |     _____________         |
                              |__|--->|             |<--------
                                      | compute_add |      __
                                      |_____________|---->|  |
                                       ______________     |  | out_stream
                                      |              |<---|__|
                                      | store_result |
                                      |______________|-----> Output result to Global Memory

*******************************************************************************/

// Includes
#include <stdint.h>
#include <hls_stream.h>
#include <cmath>

// Define data types
typedef struct {
    float x, y, z;
} particle_position_t;

typedef struct {
    float x, y, z;
} force_vector_t;

// Default simulation parameters
#define PARTICLES 10000
static const float DEFAULT_EPSILON = 1.0f;  // Depth of potential well
static const float DEFAULT_SIGMA = 1.0f;    // Distance at which potential is zero
static const float DEFAULT_CUTOFF = 2.5f;  // Typical cutoff distance

static void load_input(particle_position_t* in, hls::stream<particle_position_t>& inStream, int num_particles) {
mem_rd:
    for (int i = 0; i < num_particles; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 10000 max = 10000
        inStream << in[i];
    }
}

// Compute Lennard-Jones force magnitude between two particles
static float compute_lj_force_magnitude(float r_squared, float epsilon, float sigma) {
    float r = std::sqrt(r_squared);
    float inv_r = 1.0f / r;
    float inv_r6 = inv_r * inv_r * inv_r * inv_r * inv_r * inv_r;
    float inv_r12 = inv_r6 * inv_r6;
    float sigma6 = sigma * sigma * sigma * sigma * sigma * sigma;
    float sigma12 = sigma6 * sigma6;
    
    // F(r) = 24ε[(2σ^12/r^13) - (σ^6/r^7)]
    return 24.0f * epsilon * (2.0f * sigma12 * inv_r12 - sigma6 * inv_r6) * inv_r * inv_r;
}

// Compute Lennard-Jones forces between all pairs of particles
static void compute_lj_forces(hls::stream<particle_position_t>& in_stream,
                              hls::stream<force_vector_t>& out_stream,
                              int num_particles) {

execute:
    particle_position_t particle_positions[PARTICLES];

    // First, read all positions into local memory
    for (int i = 0; i < num_particles; i++) {
    #pragma HLS LOOP_TRIPCOUNT min = 10000 max = 10000
        particle_positions[i] = in_stream.read();
    }

    // Calculate forces for each particle
    for (int i = 0; i < num_particles; i++) {
        #pragma HLS LOOP_TRIPCOUNT min = 10000 max = 10000
        float net_fx = 0.0f;
        float net_fy = 0.0f;
        float net_fz = 0.0f;
        // Get current particle position once to reduce redundant loads
        particle_position_t p1 = particle_positions[i];
                
        // Calculate interaction with all other particles
        for (int j = 0; j < num_particles; j++) {
            #pragma HLS LOOP_TRIPCOUNT min = 10000 max = 10000
            if (i != j) {
                particle_position_t p2 = particle_positions[j];

                float dx = p1.x - p2.x;
                float dy = p1.y - p2.y;
                float dz = p1.z - p2.z;
                float r_squared = dx*dx + dy*dy + dz*dz;
                
                if (r_squared <= DEFAULT_CUTOFF * DEFAULT_CUTOFF && r_squared > 0.0001f) { 
                    float force_mag = compute_lj_force_magnitude(r_squared, DEFAULT_EPSILON, DEFAULT_SIGMA);
                    
                    // Accumulate force components
                    net_fx += force_mag * dx;
                    net_fy += force_mag * dy;
                    net_fz += force_mag * dz;
                }
            }
        }

        force_vector_t net_force;
        net_force.x = net_fx;
        net_force.y = net_fy;
        net_force.z = net_fz;
        out_stream << net_force;
    }
}

static void store_result(force_vector_t* out, hls::stream<force_vector_t>& out_stream, int num_particles) {
mem_wr:
    for (int i = 0; i < num_particles; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 10000 max = 10000
        out[i] = out_stream.read();
    }
}

extern "C" {

/*
    Lennard-Jones Force Calculation Kernel

    Arguments:
        in  (input)  --> Input particle positions vector
        out  (output) --> Output forces vector
        num_particles (input)  --> Number of particles in vector
*/

void krnl_vadd(particle_position_t* in, force_vector_t* out, int num_particles) {
#pragma HLS INTERFACE m_axi port = in bundle = gmem0
#pragma HLS INTERFACE m_axi port = out bundle = gmem1

    static hls::stream<particle_position_t> in_stream("input_stream"); 
    static hls::stream<force_vector_t> out_stream("output_stream");

#pragma HLS dataflow
    // dataflow pragma instruct compiler to run following three APIs in parallel
    load_input(in, in_stream, num_particles);
    compute_lj_forces(in_stream, out_stream, num_particles);
    store_result(out, out_stream, num_particles);
}
}
