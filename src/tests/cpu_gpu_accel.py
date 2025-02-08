import math

import numpy as np
import pyopencl as cl
from numba import njit, prange

# Constants (example)
M_SUN = 1.989e30  # Mass of the Sun (kg)
M_EARTH = 5.972e24  # Mass of the Earth (kg)


# Setup OpenCL (Intel Iris X GPU) - We will use this for GPU acceleration
def setup_opencl():
    platforms = cl.get_platforms()
    platform = platforms[0]  # Choose the first platform
    devices = platform.get_devices()
    device = devices[0]  # Choose the first device (Intel Iris X)

    context = cl.Context([device])
    queue = cl.CommandQueue(context, device)

    return context, queue, device


# OpenCL Kernel code for Lagrange point calculation
opencl_kernel_code = """
__kernel void compute_lagrange_points(__global const float *sun_pos,
                                      __global const float *earth_pos,
                                      const float M_SUN,
                                      const float M_EARTH,
                                      __global float *output) {
    int id = get_global_id(0);
    
    float MU = M_EARTH / (M_SUN + M_EARTH);
    float MUminus = 1.0f - MU;
    
    // Earth-Sun distance
    float dx = earth_pos[0] - sun_pos[0];
    float dy = earth_pos[1] - sun_pos[1];
    float dz = earth_pos[2] - sun_pos[2];
    float r = sqrt(dx * dx + dy * dy + dz * dz);
    
    // Angle of the Earth-Sun vector
    float angle = atan2(dy, dx);
    
    // Barycenter computation
    float bary_pos_x = MUminus * sun_pos[0] + MU * earth_pos[0];
    float bary_pos_y = MUminus * sun_pos[1] + MU * earth_pos[1];
    float bary_pos_z = MUminus * sun_pos[2] + MU * earth_pos[2];
    
    // L1, L2, L3, L4, L5 calculations based on barycenter
    if (id == 0) {  // L1
        output[0] = bary_pos_x + (r + 0.01f) * cos(angle);
        output[1] = bary_pos_y + (r + 0.01f) * sin(angle);
        output[2] = bary_pos_z;
    } else if (id == 1) {  // L2
        output[3] = bary_pos_x + (r - 0.01f) * cos(angle);
        output[4] = bary_pos_y + (r - 0.01f) * sin(angle);
        output[5] = bary_pos_z;
    } else if (id == 2) {  // L3
        output[6] = bary_pos_x - r * cos(angle);
        output[7] = bary_pos_y - r * sin(angle);
        output[8] = bary_pos_z;
    } else if (id == 3) {  // L4
        float angle_offset = 2.094f;  // 60 degrees in radians
        output[9] = bary_pos_x + r * cos(angle + angle_offset);
        output[10] = bary_pos_y + r * sin(angle + angle_offset);
        output[11] = bary_pos_z;
    } else if (id == 4) {  // L5
        float angle_offset = -2.094f;  // -60 degrees in radians
        output[12] = bary_pos_x + r * cos(angle + angle_offset);
        output[13] = bary_pos_y + r * sin(angle + angle_offset);
        output[14] = bary_pos_z;
    }
}
"""


# Numba CPU Parallelized function for Lagrange Points (for CPU)
@njit(parallel=True)
def compute_lagrange_points_cpu(sun_pos, earth_pos):
    MU = M_EARTH / (M_SUN + M_EARTH)
    MUminus = 1 - MU

    # Compute distance and angle
    delta = earth_pos - sun_pos
    r = math.sqrt(delta[0] ** 2 + delta[1] ** 2 + delta[2] ** 2)  # Earth-Sun distance
    angle = math.atan2(delta[1], delta[0])  # Angle of the Earth-Sun vector

    # Barycenter computation
    bary_pos = MUminus * sun_pos + MU * earth_pos

    # Precompute sin and cos values for the Lagrange points
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)

    # Calculate the Lagrange points (L1, L2, L3, L4, L5)
    L1 = bary_pos + (r + 0.01) * np.array([cos_angle, sin_angle, 0])
    L2 = bary_pos + (r - 0.01) * np.array([cos_angle, sin_angle, 0])
    L3 = bary_pos + r * np.array([-cos_angle, -sin_angle, 0])

    angle_offset = math.pi / 3  # 60 degrees in radians
    cos_L4, sin_L4 = math.cos(angle + angle_offset), math.sin(angle + angle_offset)
    cos_L5, sin_L5 = math.cos(angle - angle_offset), math.sin(angle - angle_offset)

    L4 = bary_pos + r * np.array([cos_L4, sin_L4, 0])
    L5 = bary_pos + r * np.array([cos_L5, sin_L5, 0])

    return [list(L1), list(L2), list(L3), list(L4), list(L5)]


# Main function to orchestrate CPU and GPU work
def main(sun_pos, earth_pos):
    # Setup OpenCL for GPU
    context, queue, device = setup_opencl()

    # Create OpenCL program from kernel code
    program = cl.Program(context, opencl_kernel_code).build()

    # Buffer for inputs and outputs (GPU)
    sun_pos_buf = cl.Buffer(
        context,
        cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
        hostbuf=np.array(sun_pos, dtype=np.float32),
    )
    earth_pos_buf = cl.Buffer(
        context,
        cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
        hostbuf=np.array(earth_pos, dtype=np.float32),
    )
    output_buf = cl.Buffer(
        context, cl.mem_flags.WRITE_ONLY, size=5 * 3 * 4
    )  # 5 Lagrange points, each 3 floats

    # Launch the OpenCL kernel
    program.compute_lagrange_points(
        queue,
        (5,),
        None,
        sun_pos_buf,
        earth_pos_buf,
        np.float32(M_SUN),
        np.float32(M_EARTH),
        output_buf,
    )

    # Read results from GPU
    output = np.empty(15, dtype=np.float32)
    cl.enqueue_copy(queue, output, output_buf).wait()

    # Process the results (reshape them)
    lagrange_points_gpu = [tuple(output[i : i + 3]) for i in range(0, 15, 3)]

    # Use CPU parallel computation to compute the Lagrange points as well
    lagrange_points_cpu = compute_lagrange_points_cpu(
        np.array(sun_pos), np.array(earth_pos)
    )

    print("Lagrange points (from GPU):", lagrange_points_gpu)
    print("Lagrange points (from CPU):", lagrange_points_cpu)


if __name__ == "__main__":
    # Example coordinates (Sun and Earth)
    sun_pos = [0.0, 0.0, 0.0]  # Sun at the origin
    earth_pos = [1.4960e8, 0.0, 0.0]  # Earth at 1 AU along the x-axis

    main(sun_pos, earth_pos)
