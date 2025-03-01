#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "err.h" // Asegúrate de tener este header con las macros de chequeo de errores CUDA

#define BLOCK_SIZE 256
#define NUM_BODIES 300
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

// Video para almacenar las simulaciones
// Nota: Se define globalmente para usarlo en storeFrame()
cv::VideoWriter video("nbody3D.avi",
                      cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                      30,
                      cv::Size(WINDOW_WIDTH, WINDOW_HEIGHT));

/**
 * Estructura de 3 componentes (x, y, z).
 */
typedef struct
{
    double x;
    double y;
    double z;
} Vector3;

/**
 * Estructura para un cuerpo (Body) en 3D.
 */
typedef struct
{
    bool isDynamic;
    double mass;
    double radius;
    Vector3 position;
    Vector3 velocity;
    Vector3 acceleration;
} Body;

/**
 * Calcula la distancia 3D entre dos posiciones.
 */
__device__ double getDistance3D(const Vector3 &pos1, const Vector3 &pos2)
{
    double dx = pos1.x - pos2.x;
    double dy = pos1.y - pos2.y;
    double dz = pos1.z - pos2.z;
    return sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * Determina si dos cuerpos colisionan en 3D (según la suma de sus radios y un umbral).
 */
__device__ bool isCollide(Body &b1, Body &b2)
{
    double dist = getDistance3D(b1.position, b2.position);
    return (b1.radius + b2.radius + COLLISION_TH) > dist;
}

/**
 * Kernel que aplica el método Direct Sum en 3D usando tiling.
 */
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
        // Inicializamos la aceleración
        bi.acceleration = {0.0, 0.0, 0.0};

        for (int tile = 0; tile < gridDim.x; ++tile)
        {
            // Copiamos un tile de cuerpos a memoria compartida
            int idx = tile * blockDim.x + tx;
            if (idx < n)
                Bds[tx] = bodies[idx];
            __syncthreads();

            // Procesamos cada elemento del tile
            for (int b = 0; b < BLOCK_SIZE; ++b)
            {
                int j = tile * blockDim.x + b;
                if (j < n)
                {
                    Body &bj = Bds[b]; // Acceso a la memoria compartida
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
        // Actualizamos la aceleración
        bi.acceleration.x += fx;
        bi.acceleration.y += fy;
        bi.acceleration.z += fz;

        // Integramos velocidad
        bi.velocity.x += bi.acceleration.x * DT;
        bi.velocity.y += bi.acceleration.y * DT;
        bi.velocity.z += bi.acceleration.z * DT;

        // Integramos posición
        bi.position.x += bi.velocity.x * DT;
        bi.position.y += bi.velocity.y * DT;
        bi.position.z += bi.velocity.z * DT;
    }
}

/**
 * Proyecta una posición 3D a 2D para dibujar en la ventana.
 * En este ejemplo se ignora el eje Z (proyección ortográfica).
 */
cv::Point scaleToWindow(const Vector3 &pos3D)
{
    double scaleX = WINDOW_WIDTH / (NBODY_WIDTH * 2.0);
    double scaleY = WINDOW_HEIGHT / (NBODY_HEIGHT * 2.0);
    double screenX = (pos3D.x + NBODY_WIDTH) * scaleX;
    double screenY = (pos3D.y + NBODY_HEIGHT) * scaleY;
    return cv::Point((int)screenX, (int)(WINDOW_HEIGHT - screenY));
}

/**
 * Dibuja los cuerpos en una imagen y la escribe al video.
 */
void storeFrame(Body *bodies, int n, int id)
{
    cv::Mat image = cv::Mat::zeros(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC3);
    for (int i = 0; i < n; i++)
    {
        cv::Point center = scaleToWindow(bodies[i].position);
        cv::circle(image, center, 1, cv::Scalar(255, 255, 255), -1);
    }
    video.write(image);
    // Para guardar frames individuales como imagen:
    // cv::imwrite("frame3D_" + std::to_string(id) + ".jpg", image);
}

/**
 * Inicializa los cuerpos en 3D aleatoriamente alrededor de un centro.
 */
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

/**
 * Función auxiliar para setear un cuerpo concreto.
 */
void setBody(Body *bodies, int i,
             bool isDynamic, double mass, double radius,
             Vector3 position, Vector3 velocity, Vector3 acceleration)
{
    bodies[i].isDynamic = isDynamic;
    bodies[i].mass = mass;
    bodies[i].radius = radius;
    bodies[i].position = position;
    bodies[i].velocity = velocity;
    bodies[i].acceleration = acceleration;
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "Uso: " << argv[0] << " <nBodies> <iters>\n";
        return -1;
    }

    int nBodies = atoi(argv[1]);
    int iters = atoi(argv[2]); // Número de iteraciones

    // Inicializamos cuerpos en el host
    Body *h_bodies = initRandomBodies3D(nBodies);
    int bytes = nBodies * sizeof(Body);

    // Reservar memoria en la GPU
    Body *d_bodies;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_bodies, bytes));

    // Copiar datos del host a la GPU
    CHECK_CUDA_ERROR(cudaMemcpy(d_bodies, h_bodies, bytes, cudaMemcpyHostToDevice));

    // Configuración del kernel
    int blockSize = BLOCK_SIZE;
    int gridSize = (nBodies + blockSize - 1) / blockSize;

    // Encabezado CSV para mediciones
    std::cout << "iteration,totalIterationMs" << std::endl;
    using Clock = std::chrono::steady_clock;
    for (int it = 0; it < iters; it++)
    {
        auto start = Clock::now();

        // Lanza el kernel
        DirectSumTiledKernel3D<<<gridSize, blockSize>>>(d_bodies, nBodies);
        cudaDeviceSynchronize();

        auto end = Clock::now();
        double iterationTime = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << it << "," << iterationTime << std::endl;

        // Copiar resultados de la GPU al host para dibujar el frame
        CHECK_CUDA_ERROR(cudaMemcpy(h_bodies, d_bodies, bytes, cudaMemcpyDeviceToHost));
        storeFrame(h_bodies, nBodies, it);
    }

    // Liberar recursos
    video.release();
    CHECK_CUDA_ERROR(cudaFree(d_bodies));
    free(h_bodies);

    return 0;
}
