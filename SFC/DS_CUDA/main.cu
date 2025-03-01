#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cstdint>
#include <opencv2/opencv.hpp>
#include "err.h" // Debe contener la macro CHECK_CUDA_ERROR

//--------------------------------------------------
// Constantes para simulación y visualización
//--------------------------------------------------
#define BLOCK_SIZE 256
#define NUM_BODIES_DEFAULT 300
#define WINDOW_WIDTH 1600
#define WINDOW_HEIGHT 1600
#define NBODY_WIDTH 10.0e11
#define NBODY_HEIGHT 10.0e11
#define GRAVITY 6.67E-11
#define E 0.5
#define DT 25000
#define CENTERX 0
#define CENTERY 0
#define COLLISION_TH 1.0e10
#define MIN_DIST 2.0e10
#define MAX_DIST 5.0e11
#define SUN_MASS 1.9890e30
#define SUN_DIA 1.3927e6
#define EARTH_MASS 5.974e24
#define EARTH_DIA 12756

//--------------------------------------------------
// Video para almacenar la simulación
//--------------------------------------------------
cv::VideoWriter video("nbody3D.avi",
                      cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                      30,
                      cv::Size(WINDOW_WIDTH, WINDOW_HEIGHT));

//--------------------------------------------------
// Definiciones de estructuras
//--------------------------------------------------
typedef struct
{
    double x;
    double y;
    double z;
} Vector3;

typedef struct
{
    bool isDynamic;
    double mass;
    double radius;
    Vector3 position;
    Vector3 velocity;
    Vector3 acceleration;
} Body;

//--------------------------------------------------
// Funciones __device__ para la simulación
//--------------------------------------------------
__device__ double getDistance3D(const Vector3 &pos1, const Vector3 &pos2)
{
    double dx = pos1.x - pos2.x;
    double dy = pos1.y - pos2.y;
    double dz = pos1.z - pos2.z;
    return sqrt(dx * dx + dy * dy + dz * dz);
}

__device__ bool isCollide(Body &b1, Body &b2)
{
    double dist = getDistance3D(b1.position, b2.position);
    return (b1.radius + b2.radius + COLLISION_TH) > dist;
}

__global__ void DirectSumTiledKernel3D(Body *bodies, int n)
{
    __shared__ Body Bds[BLOCK_SIZE];

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int i = bx * blockDim.x + tx;

    if (i < n)
    {
        Body &bi = bodies[i];
        double fx = 0.0, fy = 0.0, fz = 0.0;
        bi.acceleration = {0.0, 0.0, 0.0};

        for (int tile = 0; tile < gridDim.x; ++tile)
        {
            int idx = tile * blockDim.x + tx;
            if (idx < n)
                Bds[tx] = bodies[idx];
            __syncthreads();

            for (int b = 0; b < BLOCK_SIZE; ++b)
            {
                int j = tile * blockDim.x + b;
                if (j < n)
                {
                    Body &bj = Bds[b];
                    if (!isCollide(bi, bj) && bi.isDynamic)
                    {
                        double rx = bj.position.x - bi.position.x;
                        double ry = bj.position.y - bi.position.y;
                        double rz = bj.position.z - bi.position.z;
                        double r = sqrt(rx * rx + ry * ry + rz * rz + (E * E));
                        double f = (GRAVITY * bi.mass * bj.mass) / (r * r * r + (E * E));
                        fx += (f * rx) / bi.mass;
                        fy += (f * ry) / bi.mass;
                        fz += (f * rz) / bi.mass;
                    }
                }
            }
            __syncthreads();
        }
        bi.acceleration.x += fx;
        bi.acceleration.y += fy;
        bi.acceleration.z += fz;

        bi.velocity.x += bi.acceleration.x * DT;
        bi.velocity.y += bi.acceleration.y * DT;
        bi.velocity.z += bi.acceleration.z * DT;

        bi.position.x += bi.velocity.x * DT;
        bi.position.y += bi.velocity.y * DT;
        bi.position.z += bi.velocity.z * DT;
    }
}

//--------------------------------------------------
// Funciones para la visualización (host)
//--------------------------------------------------
cv::Point scaleToWindow(const Vector3 &pos3D)
{
    double scaleX = static_cast<double>(WINDOW_WIDTH) / (NBODY_WIDTH * 2.0);
    double scaleY = static_cast<double>(WINDOW_HEIGHT) / (NBODY_HEIGHT * 2.0);
    double screenX = (pos3D.x + NBODY_WIDTH) * scaleX;
    double screenY = (pos3D.y + NBODY_HEIGHT) * scaleY;
    return cv::Point(static_cast<int>(screenX), static_cast<int>(WINDOW_HEIGHT - screenY));
}

void storeFrame(Body *bodies, int n, int id)
{
    cv::Mat image = cv::Mat::zeros(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC3);
    for (int i = 0; i < n; i++)
    {
        cv::Point center = scaleToWindow(bodies[i].position);
        cv::circle(image, center, 1, cv::Scalar(255, 255, 255), -1);
    }
    video.write(image);
    // Opcional: guardar frames individuales
    // cv::imwrite("frame3D_" + std::to_string(id) + ".jpg", image);
}

//--------------------------------------------------
// Funciones de ordenamiento SFC en el host
//--------------------------------------------------

// Esta función expande 21 bits de un entero a 63 bits intercalando ceros.
uint64_t expandBits(uint32_t v)
{
    uint64_t x = v & 0x1fffff; // 21 bits
    x = (x | x << 32) & 0x1f00000000ffffULL;
    x = (x | x << 16) & 0x1f0000ff0000ffULL;
    x = (x | x << 8) & 0x100f00f00f00f00fULL;
    x = (x | x << 4) & 0x10c30c30c30c30c3ULL;
    x = (x | x << 2) & 0x1249249249249249ULL;
    return x;
}

// Calcula la clave Morton (Z-order) para un cuerpo.
uint64_t computeMortonKey(const Body &b)
{
    double nx = (b.position.x + NBODY_WIDTH) / (2 * NBODY_WIDTH);
    double ny = (b.position.y + NBODY_HEIGHT) / (2 * NBODY_HEIGHT);
    double nz = (b.position.z + NBODY_WIDTH) / (2 * NBODY_WIDTH);
    uint32_t ix = (uint32_t)(nx * ((1 << 21) - 1));
    uint32_t iy = (uint32_t)(ny * ((1 << 21) - 1));
    uint32_t iz = (uint32_t)(nz * ((1 << 21) - 1));
    return (expandBits(ix) << 2) | (expandBits(iy) << 1) | (expandBits(iz));
}

// Hilbert curve auxiliaries
static inline int rotateRight(int value, int shift)
{
    return ((value >> shift) | (value << (3 - shift))) & 7;
}

static inline int gray(int x)
{
    return x ^ (x >> 1);
}

/**
 * Calcula el índice Hilbert en 3D a partir de coordenadas enteras.
 * Implementación basada en el algoritmo de John Skilling.
 */
uint64_t hilbert3D(uint32_t x, uint32_t y, uint32_t z, int bits)
{
    uint64_t H = 0;
    int rotation = 0;
    int mask = 1 << (bits - 1);
    for (int i = 0; i < bits; i++)
    {
        int rx = (x & mask) ? 1 : 0;
        int ry = (y & mask) ? 1 : 0;
        int rz = (z & mask) ? 1 : 0;
        int current = (rx << 2) | (ry << 1) | rz;
        current = rotateRight(current, rotation);
        H = (H << 3) | (current & 7);
        rotation ^= gray(current);
        mask >>= 1;
    }
    return H;
}

/**
 * Calcula la clave Hilbert para un cuerpo.
 */
uint64_t computeHilbertKey(const Body &b)
{
    const int bits = 21; // número de bits por coordenada
    double nx = (b.position.x + NBODY_WIDTH) / (2 * NBODY_WIDTH);
    double ny = (b.position.y + NBODY_HEIGHT) / (2 * NBODY_HEIGHT);
    double nz = (b.position.z + NBODY_WIDTH) / (2 * NBODY_WIDTH);
    uint32_t ix = static_cast<uint32_t>(nx * ((1u << bits) - 1));
    uint32_t iy = static_cast<uint32_t>(ny * ((1u << bits) - 1));
    uint32_t iz = static_cast<uint32_t>(nz * ((1u << bits) - 1));
    return hilbert3D(ix, iy, iz, bits);
}

//--------------------------------------------------
// Inicializa los cuerpos de forma aleatoria en 3D
//--------------------------------------------------
Body *initRandomBodies3D(int n)
{
    Body *bodies = new Body[n];
    srand(time(NULL));

    double maxDistance = MAX_DIST;
    double minDistance = MIN_DIST;
    Vector3 centerPos = {CENTERX, CENTERY, 0.0};

    for (int i = 0; i < n; ++i)
    {
        double r = (maxDistance - minDistance) * (rand() / (double)RAND_MAX) + minDistance;
        double theta = 2.0 * M_PI * (rand() / (double)RAND_MAX);
        double phi = M_PI * (rand() / (double)RAND_MAX);
        double x = centerPos.x + r * sin(phi) * cos(theta);
        double y = centerPos.y + r * sin(phi) * sin(theta);
        double z = r * cos(phi);

        bodies[i].isDynamic = true;
        bodies[i].mass = SUN_MASS;
        bodies[i].radius = SUN_DIA;
        bodies[i].position = {x, y, z};
        bodies[i].velocity = {0.0, 0.0, 0.0};
        bodies[i].acceleration = {0.0, 0.0, 0.0};
    }
    return bodies;
}

//--------------------------------------------------
// MAIN
//--------------------------------------------------
int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::cerr << "Uso: " << argv[0] << " <nBodies> <iters> <orderType>\n";
        std::cerr << "   orderType: 0 = sin orden, 1 = Z-order (Morton), 2 = Hilbert\n";
        return -1;
    }

    int nBodies = atoi(argv[1]);
    int iters = atoi(argv[2]);
    int orderType = atoi(argv[3]); // 0: sin reordenar, 1: Morton, 2: Hilbert

    int bytes = nBodies * sizeof(Body);

    // Inicializar cuerpos en el host
    Body *h_bodies = initRandomBodies3D(nBodies);

    // Reservar memoria en la GPU
    Body *d_bodies;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_bodies, bytes));

    // Copiar datos iniciales al device
    CHECK_CUDA_ERROR(cudaMemcpy(d_bodies, h_bodies, bytes, cudaMemcpyHostToDevice));

    int blockSize = BLOCK_SIZE;
    int gridSize = (nBodies + blockSize - 1) / blockSize;

    std::cout << "iteration,totalIterationMs" << std::endl;
    using Clock = std::chrono::steady_clock;
    for (int it = 0; it < iters; it++)
    {
        auto start = Clock::now();

        // Lanza el kernel de Direct Sum con tiling
        DirectSumTiledKernel3D<<<gridSize, blockSize>>>(d_bodies, nBodies);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        auto end = Clock::now();
        double iterationTime = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << it << "," << iterationTime << std::endl;

        // Copiar resultados de la GPU al host para dibujar
        CHECK_CUDA_ERROR(cudaMemcpy(h_bodies, d_bodies, bytes, cudaMemcpyDeviceToHost));
        storeFrame(h_bodies, nBodies, it);

        // Si se seleccionó un ordenamiento SFC, reordena el arreglo de cuerpos en el host
        if (orderType != 0)
        {
            std::vector<std::pair<uint64_t, Body>> keyed(nBodies);
            for (int i = 0; i < nBodies; i++)
            {
                uint64_t key = (orderType == 1) ? computeMortonKey(h_bodies[i])
                                                : computeHilbertKey(h_bodies[i]);
                keyed[i] = std::make_pair(key, h_bodies[i]);
            }
            std::sort(keyed.begin(), keyed.end(),
                      [](const std::pair<uint64_t, Body> &a, const std::pair<uint64_t, Body> &b)
                      {
                          return a.first < b.first;
                      });
            for (int i = 0; i < nBodies; i++)
            {
                h_bodies[i] = keyed[i].second;
            }
            // Copiar la versión reordenada de vuelta a la GPU
            CHECK_CUDA_ERROR(cudaMemcpy(d_bodies, h_bodies, bytes, cudaMemcpyHostToDevice));
        }
    }

    video.release();
    CHECK_CUDA_ERROR(cudaFree(d_bodies));
    free(h_bodies);

    return 0;
}
