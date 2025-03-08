#include "../../include/common/types.cuh"
#include "../../include/common/constants.cuh"

#define NODE_CACHE_SIZE 64 // Número de nodos a cachear en memoria compartida

/**
 * @brief Calculate the distance between two positions
 *
 * @param pos1 First position
 * @param pos2 Second position
 * @return Distance between positions
 */
__device__ double getDistance(Vector pos1, Vector pos2)
{
    return sqrt((pos1.x - pos2.x) * (pos1.x - pos2.x) +
                (pos1.y - pos2.y) * (pos1.y - pos2.y) +
                (pos1.z - pos2.z) * (pos1.z - pos2.z));
}

/**
 * @brief Check if a body collides with a center of mass
 *
 * @param b1 Body to check
 * @param cm Center of mass position
 * @return True if collision occurs, false otherwise
 */
__device__ bool isCollide(Body &b1, Vector cm)
{
    double d = getDistance(b1.position, cm);
    double threshold = b1.radius * 2 + COLLISION_TH;
    return threshold > d;
}

/**
 * @brief CUDA kernel to compute the forces between bodies in an N-Body simulation.
 *
 * This kernel calculates the gravitational forces exerted on each body by all other bodies
 * in the simulation. The forces are then used to update the velocities and positions of the bodies.
 *
 * @param positions Array of body positions in the simulation.
 * @param velocities Array of body velocities in the simulation.
 * @param forces Array to store the computed forces for each body.
 * @param numBodies The total number of bodies in the simulation.
 * @param deltaTime The time step for the simulation.
 * @param gravitationalConstant The gravitational constant used in the force calculation.
 */
__global__ void ComputeForceKernel(
    Node *nodes, Body *bodies, int *orderedIndices, bool useSFC,
    int nNodes, int nBodies, int leafLimit)
{
    // Memoria compartida para nodos frecuentemente accedidos
    __shared__ Node sharedNodes[NODE_CACHE_SIZE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Cargar nodos cercanos a la raíz (accedidos por todos los hilos)
    if (threadIdx.x < NODE_CACHE_SIZE)
    {
        // Solo cargar nodos válidos
        if (threadIdx.x < nNodes)
        {
            sharedNodes[threadIdx.x] = nodes[threadIdx.x];
        }
    }

    __syncthreads(); // Sincronizar antes de usar la memoria compartida

    if (i >= nBodies)
    {
        return;
    }

    // Obtener el índice real del cuerpo si estamos usando SFC
    int realBodyIndex = useSFC && orderedIndices != nullptr ? orderedIndices[i] : i;

    // Calcular ancho del dominio para criterio multipolo
    double width = max(
        fabs(nodes[0].botRightBack.x - nodes[0].topLeftFront.x),
        max(
            fabs(nodes[0].botRightBack.y - nodes[0].topLeftFront.y),
            fabs(nodes[0].botRightBack.z - nodes[0].topLeftFront.z)));

    Body &bi = bodies[realBodyIndex];

    // Solo procesar cuerpos dinámicos
    if (bi.isDynamic)
    {
        // Reiniciar aceleración
        bi.acceleration = Vector(0.0, 0.0, 0.0);

        // Recorrer el árbol para calcular fuerzas - Optimización iterativa
        int stack[128]; // Tamaño máximo de pila para recorrido
        int stackSize = 0;

        stack[stackSize++] = 0; // Iniciar con el nodo raíz

        // Recorrido iterativo en lugar de recursivo
        while (stackSize > 0)
        {
            int nodeIndex = stack[--stackSize];

            // Verificar si el nodo está en caché
            Node curNode;
            if (nodeIndex < NODE_CACHE_SIZE)
            {
                curNode = sharedNodes[nodeIndex];
            }
            else
            {
                curNode = nodes[nodeIndex];
            }

            // Saltar nodos vacíos
            if (curNode.start == -1 && curNode.end == -1)
            {
                continue;
            }

            // Evitar auto-interacción
            if (curNode.isLeaf && curNode.start == i && curNode.end == i)
            {
                continue;
            }

            // Calcular ancho del nodo actual
            double curWidth = width;
            for (int level = 0; level < 30; level++)
            {
                if ((nodeIndex >> (3 * level)) == 0)
                {
                    curWidth = width / (1 << level);
                    break;
                }
            }

            // Verificar criterio de aproximación multipolar
            bool useMultipole = false;
            if (!curNode.isLeaf && curNode.totalMass > 0.0)
            {
                double dist = getDistance(bi.position, curNode.centerMass);
                if (dist > 0.0 && curWidth / dist < THETA)
                {
                    useMultipole = true;
                }
            }

            // Aplicar fuerza si el nodo es hoja o podemos usar aproximación multipolar
            if (curNode.isLeaf || useMultipole)
            {
                // Omitir si no se ha calculado el centro de masa o hay colisión
                if (curNode.totalMass <= 0.0 || isCollide(bi, curNode.centerMass))
                {
                    continue;
                }

                // Calcular vector del cuerpo al centro de masa
                Vector rij = Vector(
                    curNode.centerMass.x - bi.position.x,
                    curNode.centerMass.y - bi.position.y,
                    curNode.centerMass.z - bi.position.z);

                // Distancia al cuadrado con suavizado
                double r2 = rij.lengthSquared();
                double r = sqrt(r2 + (E * E));

                // Calcular fuerza gravitacional: G * m1 * m2 / r^3
                double f = (GRAVITY * bi.mass * curNode.totalMass) / (r * r * r);

                // Aplicar fuerza a la aceleración del cuerpo
                bi.acceleration.x += (rij.x * f / bi.mass);
                bi.acceleration.y += (rij.y * f / bi.mass);
                bi.acceleration.z += (rij.z * f / bi.mass);
            }
            else
            {
                // Añadir hijos a la pila para procesamiento (desde el 8 al 1)
                for (int c = 8; c >= 1; c--)
                {
                    int childIndex = (nodeIndex * 8) + c;
                    if (childIndex < nNodes)
                    {
                        stack[stackSize++] = childIndex;
                    }
                }
            }
        }

        // Actualizar velocidad (integración Euler)
        bi.velocity.x += bi.acceleration.x * DT;
        bi.velocity.y += bi.acceleration.y * DT;
        bi.velocity.z += bi.acceleration.z * DT;

        // Actualizar posición
        bi.position.x += bi.velocity.x * DT;
        bi.position.y += bi.velocity.y * DT;
        bi.position.z += bi.velocity.z * DT;
    }
}

/**
 * @brief CUDA kernel to compute the bounding box for a set of bodies.
 *
 * This kernel calculates the bounding box for a given set of bodies, which
 * are represented by nodes and bodies arrays. The bounding box is used to
 * determine the spatial extent of the bodies in the simulation.
 *
 * @param nodes Pointer to the array of nodes representing the bounding box.
 * @param bodies Pointer to the array of bodies in the simulation.
 * @param orderedIndices Pointer to the array of ordered indices for the bodies.
 * @param useSFC Boolean flag indicating whether to use Space-Filling Curve (SFC) for ordering.
 * @param mutex Pointer to the mutex used for synchronization.
 * @param nBodies The number of bodies in the simulation.
 */
__global__ void ComputeBoundingBoxKernel(
    Node *nodes, Body *bodies, int *orderedIndices, bool useSFC,
    int *mutex, int nBodies)
{
    // Shared memory para la reducción paralela
    __shared__ double topLeftFrontX[BLOCK_SIZE];
    __shared__ double topLeftFrontY[BLOCK_SIZE];
    __shared__ double topLeftFrontZ[BLOCK_SIZE];
    __shared__ double botRightBackX[BLOCK_SIZE];
    __shared__ double botRightBackY[BLOCK_SIZE];
    __shared__ double botRightBackZ[BLOCK_SIZE];

    int tx = threadIdx.x;
    int b = blockIdx.x * blockDim.x + tx;

    // Inicializar con valores extremos
    topLeftFrontX[tx] = INFINITY;
    topLeftFrontY[tx] = INFINITY;
    topLeftFrontZ[tx] = INFINITY;
    botRightBackX[tx] = -INFINITY;
    botRightBackY[tx] = -INFINITY;
    botRightBackZ[tx] = -INFINITY;

    __syncthreads();

    // Cargar datos de cuerpos con o sin SFC
    if (b < nBodies)
    {
        // Obtener el índice correcto dependiendo si usamos SFC o no
        int bodyIndex = (useSFC && orderedIndices != nullptr) ? orderedIndices[b] : b;

        Body body = bodies[bodyIndex];
        topLeftFrontX[tx] = body.position.x;
        topLeftFrontY[tx] = body.position.y;
        topLeftFrontZ[tx] = body.position.z;

        botRightBackX[tx] = body.position.x;
        botRightBackY[tx] = body.position.y;
        botRightBackZ[tx] = body.position.z;
    }

    // Reducción paralela para encontrar min/max valores
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        __syncthreads();
        if (tx < s)
        {
            // Min reduction for top-left-front
            topLeftFrontX[tx] = fmin(topLeftFrontX[tx], topLeftFrontX[tx + s]);
            topLeftFrontY[tx] = fmin(topLeftFrontY[tx], topLeftFrontY[tx + s]);
            topLeftFrontZ[tx] = fmin(topLeftFrontZ[tx], topLeftFrontZ[tx + s]);

            // Max reduction for bottom-right-back
            botRightBackX[tx] = fmax(botRightBackX[tx], botRightBackX[tx + s]);
            botRightBackY[tx] = fmax(botRightBackY[tx], botRightBackY[tx + s]);
            botRightBackZ[tx] = fmax(botRightBackZ[tx], botRightBackZ[tx + s]);
        }
    }

    // Actualizar root node con mutex para evitar race conditions
    if (tx == 0)
    {
        // Wait until mutex is available
        while (atomicCAS(mutex, 0, 1) != 0)
        {
        }

        // Update bounds with a margin for numerical stability
        // Update minimum bounds (top-left-front corner)
        nodes[0].topLeftFront.x = fmin(nodes[0].topLeftFront.x, topLeftFrontX[0] - 1.0e10);
        nodes[0].topLeftFront.y = fmin(nodes[0].topLeftFront.y, topLeftFrontY[0] - 1.0e10);
        nodes[0].topLeftFront.z = fmin(nodes[0].topLeftFront.z, topLeftFrontZ[0] - 1.0e10);

        // Update maximum bounds (bottom-right-back corner)
        nodes[0].botRightBack.x = fmax(nodes[0].botRightBack.x, botRightBackX[0] + 1.0e10);
        nodes[0].botRightBack.y = fmax(nodes[0].botRightBack.y, botRightBackY[0] + 1.0e10);
        nodes[0].botRightBack.z = fmax(nodes[0].botRightBack.z, botRightBackZ[0] + 1.0e10);

        // Release mutex
        atomicExch(mutex, 0);
    }
}