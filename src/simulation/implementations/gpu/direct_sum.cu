#include "../../../../include/simulation/implementations/gpu/direct_sum.cuh"
#include <iostream>

__global__ void DirectSumForceKernel(Body *bodies, int nBodies)
{
  extern __shared__ char sharedMemory[];
  Vector *sharedPos = (Vector *)sharedMemory;
  double *sharedMass = (double *)(sharedPos + blockDim.x);

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int tx = threadIdx.x;

  // Cargar datos solo si es un índice válido para reducir divergencia
  Vector myPos = Vector(0, 0, 0);
  Vector myVel = Vector(0, 0, 0);
  Vector myAcc = Vector(0, 0, 0);
  double myMass = 0.0;
  bool isDynamic = false;
  bool isValid = false;

  if (i < nBodies)
  {
    myPos = bodies[i].position;
    myVel = bodies[i].velocity;
    myMass = bodies[i].mass;
    isDynamic = bodies[i].isDynamic;
    isValid = true;
  }

  const int tileSize = blockDim.x;

  // Procesar todos los tiles en orden para mejor localidad de memoria
  for (int tile = 0; tile < (nBodies + tileSize - 1) / tileSize; ++tile)
  {
    // Cargar este tile a memoria compartida
    int idx = tile * tileSize + tx;

    // Inicializar memoria compartida a valores por defecto
    sharedPos[tx] = Vector(0, 0, 0);
    sharedMass[tx] = 0.0;

    // Solo cargar datos válidos a memoria compartida
    if (idx < nBodies)
    {
      sharedPos[tx] = bodies[idx].position;
      sharedMass[tx] = bodies[idx].mass;
    }

    __syncthreads();

    // Pre-comprobar si necesitamos calcular fuerzas para reducir divergencia
    if (isValid && isDynamic)
    {
      // Limitar el bucle al tamaño real del tile
      int tileLimit = min(tileSize, nBodies - tile * tileSize);

      for (int j = 0; j < tileLimit; ++j)
      {
        int jBody = tile * tileSize + j;

        // Evitar auto-interacción y solo considerar cuerpos con masa
        if (jBody != i && sharedMass[j] > 0.0)
        {
          // Vector de distancia
          double rx = sharedPos[j].x - myPos.x;
          double ry = sharedPos[j].y - myPos.y;
          double rz = sharedPos[j].z - myPos.z;

          // Distancia al cuadrado con suavizado
          double distSqr = rx * rx + ry * ry + rz * rz + E * E;

          // Optimización: solo calcular sqrt si es necesario
          if (distSqr >= COLLISION_TH * COLLISION_TH)
          {
            double dist = sqrt(distSqr);
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

  // Actualizar el cuerpo solo si es válido y dinámico para reducir operaciones de memoria
  if (isValid && isDynamic)
  {
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

GPUDirectSum::GPUDirectSum(int numBodies, BodyDistribution dist, unsigned int seed, MassDistribution massDist)
    : SimulationBase(numBodies, dist, seed, massDist)
{
  std::cout << "GPU Direct Sum Simulation created with " << numBodies << " bodies." << std::endl;
}

GPUDirectSum::~GPUDirectSum() {}

void GPUDirectSum::computeForces()
{
  // Medir tiempo de ejecución
  CudaTimer timer(metrics.forceTimeMs);

  // Use the global block size variable
  int blockSize = g_blockSize;
  int gridSize = (nBodies + blockSize - 1) / blockSize;

  // Lanzar kernel con comprobación de errores
  DirectSumForceKernel<<<gridSize, blockSize, blockSize * sizeof(Vector) + blockSize * sizeof(double), 0>>>(d_bodies, nBodies);
  CHECK_LAST_CUDA_ERROR();
}

void GPUDirectSum::update()
{
  // Ensure initialization
  checkInitialization();

  // Measure total execution time
  CudaTimer timer(metrics.totalTimeMs);

  // Reset unused metrics
  metrics.resetTimeMs = 0.0f;
  metrics.bboxTimeMs = 0.0f;
  metrics.octreeTimeMs = 0.0f;

  // Compute forces and update positions in one kernel
  computeForces();
}