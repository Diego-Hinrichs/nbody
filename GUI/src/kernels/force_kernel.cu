#include "../../include/common/types.cuh"
#include "../../include/common/constants.cuh"

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
 * @brief Compute force on a body using the Barnes-Hut algorithm
 *
 * Recursively traverses the octree to compute gravitational forces.
 * Uses multipole approximation when the distance is large enough
 * compared to the node size.
 *
 * @param node Array of octree nodes
 * @param bodies Array of bodies
 * @param nodeIndex Index of current node in traversal
 * @param bodyIndex Index of body to compute forces for
 * @param nNodes Total number of nodes
 * @param nBodies Total number of bodies
 * @param leafLimit Leaf node limit
 * @param width Width of current node
 */
// __device__ void ComputeForce(
//     Node *node, Body *bodies, int nodeIndex, int bodyIndex,
//     int nNodes, int nBodies, int leafLimit, double width)
// {

//     // Check if node index is valid
//     if (nodeIndex >= nNodes)
//     {
//         return;
//     }

//     Node curNode = node[nodeIndex];
//     Body bi = bodies[bodyIndex];

//     // Skip empty nodes
//     if (curNode.start == -1 && curNode.end == -1)
//     {
//         return;
//     }

//     // Skip self-interaction
//     if (curNode.isLeaf && curNode.start == bodyIndex && curNode.end == bodyIndex)
//     {
//         return;
//     }

//     // Determine if we can use multipole approximation
//     bool useMultipole = false;
//     if (!curNode.isLeaf && curNode.totalMass > 0.0)
//     {
//         double dist = getDistance(bi.position, curNode.centerMass);
//         if (dist > 0.0 && width / dist < THETA)
//         {
//             useMultipole = true;
//         }
//     }

//     // Apply force if:
//     // 1. Node is a leaf, or
//     // 2. We can use multipole approximation
//     if (curNode.isLeaf || useMultipole)
//     {
//         // Skip if center of mass not computed or collision detected
//         if (curNode.totalMass <= 0.0 || isCollide(bi, curNode.centerMass))
//         {
//             return;
//         }

//         // Calculate vector from body to center of mass
//         Vector rij = Vector(
//             curNode.centerMass.x - bi.position.x,
//             curNode.centerMass.y - bi.position.y,
//             curNode.centerMass.z - bi.position.z);

//         // Square distance with softening
//         double r2 = rij.lengthSquared();
//         double r = sqrt(r2 + (E * E));

//         // Calculate gravitational force: G * m1 * m2 / r^3
//         double f = (GRAVITY * bi.mass * curNode.totalMass) / (r * r * r);

//         // Apply force to body acceleration
//         bodies[bodyIndex].acceleration.x += (rij.x * f / bi.mass);
//         bodies[bodyIndex].acceleration.y += (rij.y * f / bi.mass);
//         bodies[bodyIndex].acceleration.z += (rij.z * f / bi.mass);

//         return;
//     }

//     // Recursively traverse child nodes
//     for (int i = 1; i <= 8; i++)
//     {
//         int childIndex = (nodeIndex * 8) + i;
//         if (childIndex < nNodes)
//         {
//             ComputeForce(node, bodies, childIndex, bodyIndex, nNodes, nBodies, leafLimit, width / 2);
//         }
//     }
// }

// /**
//  * @brief Compute forces on all bodies using the Barnes-Hut algorithm
//  *
//  * Each thread processes one body, calculating gravitational forces
//  * and updating position and velocity.
//  *
//  * @param node Array of octree nodes
//  * @param bodies Array of bodies
//  * @param nNodes Total number of nodes
//  * @param nBodies Total number of bodies
//  * @param leafLimit Leaf node limit
//  */
// __global__ void ComputeForceKernel(
//     Node *node, Body *bodies, int nNodes, int nBodies, int leafLimit)
// {

//     int i = blockIdx.x * blockDim.x + threadIdx.x;

//     if (i >= nBodies)
//     {
//         return;
//     }

//     // Calculate domain width for multipole criterion
//     double width = max(
//         fabs(node[0].botRightBack.x - node[0].topLeftFront.x),
//         max(
//             fabs(node[0].botRightBack.y - node[0].topLeftFront.y),
//             fabs(node[0].botRightBack.z - node[0].topLeftFront.z)));

//     Body &bi = bodies[i];

//     // Skip non-dynamic bodies
//     if (bi.isDynamic)
//     {
//         // Reset acceleration
//         bi.acceleration = Vector(0.0, 0.0, 0.0);

//         // Compute forces from octree
//         ComputeForce(node, bodies, 0, i, nNodes, nBodies, leafLimit, width);

//         // Update velocity (Euler integration)
//         bi.velocity.x += bi.acceleration.x * DT;
//         bi.velocity.y += bi.acceleration.y * DT;
//         bi.velocity.z += bi.acceleration.z * DT;

//         // Update position
//         bi.position.x += bi.velocity.x * DT;
//         bi.position.y += bi.velocity.y * DT;
//         bi.position.z += bi.velocity.z * DT;
//     }
// }

// Añadir al inicio del archivo force_kernel.cu
#define NODE_CACHE_SIZE 64 // Número de nodos a cachear en memoria compartida

// Versión modificada de ComputeForce que utiliza la caché de nodos
__device__ void ComputeForce(
    Node *globalNodes, Node *sharedNodes, Body *bodies, int nodeIndex, int bodyIndex,
    int nNodes, int nBodies, int leafLimit, double width)
{
    // Verificar si el índice del nodo es válido
    if (nodeIndex >= nNodes)
    {
        return;
    }

    // Usar memoria compartida para nodos cercanos a la raíz
    Node curNode;
    if (nodeIndex < NODE_CACHE_SIZE)
    {
        curNode = sharedNodes[nodeIndex];
    }
    else
    {
        curNode = globalNodes[nodeIndex];
    }

    Body bi = bodies[bodyIndex];

    // Skip empty nodes
    if (curNode.start == -1 && curNode.end == -1)
    {
        return;
    }

    // Skip self-interaction
    if (curNode.isLeaf && curNode.start == bodyIndex && curNode.end == bodyIndex)
    {
        return;
    }

    // Determine if we can use multipole approximation
    bool useMultipole = false;
    if (!curNode.isLeaf && curNode.totalMass > 0.0)
    {
        double dist = getDistance(bi.position, curNode.centerMass);
        if (dist > 0.0 && width / dist < THETA)
        {
            useMultipole = true;
        }
    }

    // Apply force if:
    // 1. Node is a leaf, or
    // 2. We can use multipole approximation
    if (curNode.isLeaf || useMultipole)
    {
        // Skip if center of mass not computed or collision detected
        if (curNode.totalMass <= 0.0 || isCollide(bi, curNode.centerMass))
        {
            return;
        }

        // Calculate vector from body to center of mass
        Vector rij = Vector(
            curNode.centerMass.x - bi.position.x,
            curNode.centerMass.y - bi.position.y,
            curNode.centerMass.z - bi.position.z);

        // Square distance with softening
        double r2 = rij.lengthSquared();
        double r = sqrt(r2 + (E * E));

        // Calculate gravitational force: G * m1 * m2 / r^3
        double f = (GRAVITY * bi.mass * curNode.totalMass) / (r * r * r);

        // Apply force to body acceleration
        bodies[bodyIndex].acceleration.x += (rij.x * f / bi.mass);
        bodies[bodyIndex].acceleration.y += (rij.y * f / bi.mass);
        bodies[bodyIndex].acceleration.z += (rij.z * f / bi.mass);

        return;
    }

    // Recursively traverse child nodes
    for (int i = 1; i <= 8; i++)
    {
        int childIndex = (nodeIndex * 8) + i;
        if (childIndex < nNodes)
        {
            ComputeForce(globalNodes, sharedNodes, bodies, childIndex, bodyIndex,
                               nNodes, nBodies, leafLimit, width / 2);
        }
    }
}

__global__ void ComputeForceKernel(
    Node *nodes, Body *bodies, int nNodes, int nBodies, int leafLimit)
{
    // Memoria compartida para nodos frecuentemente accedidos
    __shared__ Node sharedNodes[NODE_CACHE_SIZE];
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Cargar nodos cercanos a la raíz (accedidos por todos los hilos)
    if (threadIdx.x < NODE_CACHE_SIZE) {
        // Solo cargar nodos válidos
        if (threadIdx.x < nNodes) {
            sharedNodes[threadIdx.x] = nodes[threadIdx.x];
        }
    }
    
    __syncthreads();  // Sincronizar antes de usar la memoria compartida

    if (i >= nBodies) {
        return;
    }

    // Calcular ancho del dominio para criterio multipolo
    double width = max(
        fabs(nodes[0].botRightBack.x - nodes[0].topLeftFront.x),
        max(
            fabs(nodes[0].botRightBack.y - nodes[0].topLeftFront.y),
            fabs(nodes[0].botRightBack.z - nodes[0].topLeftFront.z)));

    Body &bi = bodies[i];

    // Solo procesar cuerpos dinámicos
    if (bi.isDynamic)
    {
        // Reiniciar aceleración
        bi.acceleration = Vector(0.0, 0.0, 0.0);

        // Recorrer el árbol para calcular fuerzas
        int nodeIndex = 0;
        double nodeWidth = width;
        
        // Cola para recorrido iterativo del árbol
        int stack[128];  // Tamaño máximo de pila para recorrido
        int stackSize = 0;
        
        stack[stackSize++] = 0;  // Iniciar con el nodo raíz
        
        // Recorrido iterativo en lugar de recursivo
        while (stackSize > 0) {
            nodeIndex = stack[--stackSize];
            
            // Verificar si el nodo está en caché
            Node curNode;
            if (nodeIndex < NODE_CACHE_SIZE) {
                curNode = sharedNodes[nodeIndex];
            } else {
                curNode = nodes[nodeIndex];
            }
            
            // Saltar nodos vacíos
            if (curNode.start == -1 && curNode.end == -1) {
                continue;
            }
            
            // Evitar auto-interacción
            if (curNode.isLeaf && curNode.start == i && curNode.end == i) {
                continue;
            }
            
            // Calcular ancho del nodo actual
            double curWidth = nodeWidth;
            for (int level = 0; level < 30; level++) {
                if ((nodeIndex >> (3 * level)) == 0) {
                    curWidth = width / (1 << level);
                    break;
                }
            }
            
            // Verificar criterio de aproximación multipolar
            bool useMultipole = false;
            if (!curNode.isLeaf && curNode.totalMass > 0.0) {
                double dist = getDistance(bi.position, curNode.centerMass);
                if (dist > 0.0 && curWidth / dist < THETA) {
                    useMultipole = true;
                }
            }
            
            // Aplicar fuerza si el nodo es hoja o podemos usar aproximación multipolar
            if (curNode.isLeaf || useMultipole) {
                // Omitir si no se ha calculado el centro de masa o hay colisión
                if (curNode.totalMass <= 0.0 || isCollide(bi, curNode.centerMass)) {
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
            else {
                // Añadir hijos a la pila para procesamiento
                for (int c = 8; c >= 1; c--) {
                    int childIndex = (nodeIndex * 8) + c;
                    if (childIndex < nNodes) {
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