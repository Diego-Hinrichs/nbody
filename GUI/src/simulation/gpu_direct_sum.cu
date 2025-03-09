#include "../../include/simulation/gpu_direct_sum.cuh"
#include <iostream>

/**
 * @brief CUDA kernel for direct force calculation between all body pairs
 *
 * This kernel computes the gravitational forces between all pairs of bodies
 * using the Direct Sum approach (O(n²) complexity).
 *
 * @param bodies Array of body structures
 * @param nBodies Number of bodies in the simulation
 */
__global__ void DirectSumForceKernel(Body *bodies, int nBodies)
{
    // Reducir el tamaño del array de memoria compartida
    // Usar un valor más pequeño que BLOCK_SIZE
    __shared__ Vector sharedPos[256];  // Reducido de BLOCK_SIZE
    __shared__ double sharedMass[256]; // Reducido de BLOCK_SIZE
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    
    // Cargar datos solo si el índice es válido
    Vector myPos = Vector(0, 0, 0);
    Vector myVel = Vector(0, 0, 0);
    Vector myAcc = Vector(0, 0, 0);
    double myMass = 0.0;
    bool isDynamic = false;
    
    if (i < nBodies) {
        myPos = bodies[i].position;
        myVel = bodies[i].velocity;
        myMass = bodies[i].mass;
        isDynamic = bodies[i].isDynamic;
    }
    
    // Reducir la cantidad de cálculos
    const int tileSize = 256; // Usar un tamaño de tile más pequeño
    
    // Procesar todos los tiles
    for (int tile = 0; tile < (nBodies + tileSize - 1) / tileSize; ++tile) {
        // Cargar este tile a memoria compartida
        int idx = tile * tileSize + tx;
        
        // Solo cargar datos válidos a memoria compartida
        if (tx < tileSize) { // Asegurarse de que no excedemos el tamaño del array
            if (idx < nBodies) {
                sharedPos[tx] = bodies[idx].position;
                sharedMass[tx] = bodies[idx].mass;
            } else {
                sharedPos[tx] = Vector(0, 0, 0);
                sharedMass[tx] = 0.0;
            }
        }
        
        __syncthreads();
        
        // Calcular fuerza solo para cuerpos válidos y dinámicos
        if (i < nBodies && isDynamic) {
            // Limitar el bucle al tamaño real del tile
            int tileLimit = min(tileSize, nBodies - tile * tileSize);
            
            for (int j = 0; j < tileLimit; ++j) {
                int jBody = tile * tileSize + j;
                
                // Evitar auto-interacción
                if (jBody != i) {
                    // Vector de distancia
                    double rx = sharedPos[j].x - myPos.x;
                    double ry = sharedPos[j].y - myPos.y;
                    double rz = sharedPos[j].z - myPos.z;
                    
                    // Distancia al cuadrado con suavizado
                    double distSqr = rx*rx + ry*ry + rz*rz + E*E;
                    double dist = sqrt(distSqr);
                    
                    // Aplicar fuerza solo si estamos por encima del umbral de colisión
                    if (dist >= COLLISION_TH) {
                        double forceMag = (GRAVITY * myMass * sharedMass[j]) / (dist * distSqr);
                        
                        // Acumular aceleración
                        myAcc.x += rx * forceMag / myMass;
                        myAcc.y += ry * forceMag / myMass;
                        myAcc.z += rz * forceMag / myMass;
                    }
                }
            }
        }
        
        __syncthreads();
    }
    
    // Actualizar el cuerpo solo si es válido y dinámico
    if (i < nBodies && isDynamic) {
        // Guardar aceleración
        bodies[i].acceleration = myAcc;
        
        // Actualizar velocidad
        myVel.x += myAcc.x * DT;
        myVel.y += myAcc.y * DT;
        myVel.z += myAcc.z * DT;
        bodies[i].velocity = myVel;
        
        // Actualizar posición
        myPos.x += myVel.x * DT;
        myPos.y += myVel.y * DT;
        myPos.z += myVel.z * DT;
        bodies[i].position = myPos;
    }
}

GPUDirectSum::GPUDirectSum(int numBodies, BodyDistribution dist, unsigned int seed)
    : SimulationBase(numBodies, dist, seed)
{
    std::cout << "GPU Direct Sum Simulation created with " << numBodies << " bodies." << std::endl;
}

GPUDirectSum::~GPUDirectSum()
{
    // Base class destructor handles most memory cleanup
}

void GPUDirectSum::computeForces()
{
    // Medir tiempo de ejecución
    CudaTimer timer(metrics.forceTimeMs);

    // Lanzar kernel con un tamaño de bloque más pequeño
    int blockSize = 256; // Reducido de BLOCK_SIZE (1024)
    int gridSize = (nBodies + blockSize - 1) / blockSize;

    // Lanzar kernel con comprobación de errores
    DirectSumForceKernel<<<gridSize, blockSize, 0, 0>>>(d_bodies, nBodies);
    CHECK_LAST_CUDA_ERROR();
}

void GPUDirectSum::update()
{
    // Ensure initialization
    checkInitialization();

    // Measure total execution time
    CudaTimer timer(metrics.totalTimeMs);

    // Reset unused metrics
    metrics.resetTimeMs = 0.0f;  // Not used
    metrics.bboxTimeMs = 0.0f;   // Not used
    metrics.octreeTimeMs = 0.0f; // Not used

    // Compute forces and update positions in one kernel
    computeForces();
}