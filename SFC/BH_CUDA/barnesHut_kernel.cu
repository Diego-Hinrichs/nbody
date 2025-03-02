#ifndef BARNES_HUT_KERNEL_
#define BARNES_HUT_KERNEL_
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <vector>
#include "constants.h"
#include "barnesHutCuda.cuh"
#include "barnesHut_kernel.cuh"
#include "sfc_utils.cuh"

/*
----------------------------------------------------------------------------------------
RESET KERNEL
----------------------------------------------------------------------------------------
*/
__global__ void ResetKernel(Node *node, int *mutex, int nNodes, int nBodies)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < nNodes)
    {
        // Resetear nodo a valores iniciales
        node[idx].topLeftFront = {INFINITY, INFINITY, INFINITY};
        node[idx].botRightBack = {-INFINITY, -INFINITY, -INFINITY};
        node[idx].centerMass = {0.0, 0.0, 0.0};
        node[idx].totalMass = 0.0;
        node[idx].isLeaf = true;
        node[idx].start = -1;
        node[idx].end = -1;
        node[idx].sfcCode = 0; // Initialize SFC code
        mutex[idx] = 0;
    }
    // El primer thread inicializa el nodo raíz
    if (idx == 0)
    {
        node[0].start = 0;
        node[0].end = nBodies - 1;
    }
}

#endif

/*
----------------------------------------------------------------------------------------
COMPUTE BOUNDING BOX
----------------------------------------------------------------------------------------
*/
__global__ void ComputeBoundingBoxKernel(Node *node, Body *bodies, int *mutex, int nBodies)
{
    // Memoria compartida para cada dimensión
    __shared__ double topLeftFrontX[BLOCK_SIZE];
    __shared__ double topLeftFrontY[BLOCK_SIZE];
    __shared__ double topLeftFrontZ[BLOCK_SIZE];
    __shared__ double botRightBackX[BLOCK_SIZE];
    __shared__ double botRightBackY[BLOCK_SIZE];
    __shared__ double botRightBackZ[BLOCK_SIZE];

    int tx = threadIdx.x;
    int b = blockIdx.x * blockDim.x + tx;

    // Inicialización correcta para encontrar mínimos y máximos
    topLeftFrontX[tx] = INFINITY;  // Para encontrar mínimo en X
    topLeftFrontY[tx] = -INFINITY; // Para encontrar mínimo en Y
    topLeftFrontZ[tx] = INFINITY;  // Para encontrar mínimo en Z

    botRightBackX[tx] = -INFINITY; // Para encontrar máximo en X
    botRightBackY[tx] = INFINITY;  // Para encontrar máximo en Y
    botRightBackZ[tx] = -INFINITY; // Para encontrar máximo en Z

    __syncthreads();

    // Cargar datos del cuerpo si está dentro del rango
    if (b < nBodies)
    {
        Body body = bodies[b];
        topLeftFrontX[tx] = body.position.x;
        topLeftFrontY[tx] = body.position.y;
        topLeftFrontZ[tx] = body.position.z;

        botRightBackX[tx] = body.position.x;
        botRightBackY[tx] = body.position.y;
        botRightBackZ[tx] = body.position.z;
    }

    // Reducción para encontrar mínimos y máximos
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        __syncthreads();
        if (tx < s)
        {
            // Encontrar mínimos para topLeftFront
            topLeftFrontX[tx] = fminf(topLeftFrontX[tx], topLeftFrontX[tx + s]);
            topLeftFrontY[tx] = fminf(topLeftFrontY[tx], topLeftFrontY[tx + s]);
            topLeftFrontZ[tx] = fminf(topLeftFrontZ[tx], topLeftFrontZ[tx + s]);

            // Encontrar máximos para botRightBack
            botRightBackX[tx] = fmaxf(botRightBackX[tx], botRightBackX[tx + s]);
            botRightBackY[tx] = fmaxf(botRightBackY[tx], botRightBackY[tx + s]);
            botRightBackZ[tx] = fmaxf(botRightBackZ[tx], botRightBackZ[tx + s]);
        }
    }

    // Actualización del nodo raíz con mutex
    if (tx == 0)
    {
        while (atomicCAS(mutex, 0, 1) != 0)
            ;
        // Actualizar mínimos con margen
        node[0].topLeftFront.x = fminf(node[0].topLeftFront.x, topLeftFrontX[0] - 1.0e10);
        node[0].botRightBack.y = fmaxf(node[0].botRightBack.y, botRightBackY[0] - 1.0e10);

        node[0].topLeftFront.y = fminf(node[0].topLeftFront.y, topLeftFrontY[0] + 1.0e10);
        node[0].botRightBack.x = fmaxf(node[0].botRightBack.x, botRightBackX[0] + 1.0e10);

        node[0].topLeftFront.z = fminf(node[0].topLeftFront.z, topLeftFrontZ[0] + 1.0e10);
        node[0].botRightBack.z = fmaxf(node[0].botRightBack.z, botRightBackZ[0] - 1.0e10);

        atomicExch(mutex, 0);
    }
}

/*
----------------------------------------------------------------------------------------
CONSTRUCT OCTREE
----------------------------------------------------------------------------------------
*/
__device__ int getOctant(Vector topLeftFront, Vector botRightBack, double x, double y, double z)
{
    int octant = 1;
    double midX = (topLeftFront.x + botRightBack.x) / 2;
    double midY = (topLeftFront.y + botRightBack.y) / 2;
    double midZ = (topLeftFront.z + botRightBack.z) / 2;

    // El problema podría estar aquí en la asignación de octantes
    if (x <= midX)
    {
        if (y >= midY)
        {
            if (z <= midZ)
                octant = 1;
            else
                octant = 5;
        }
        else
        {
            if (z <= midZ)
                octant = 3;
            else
                octant = 7;
        }
    }
    else
    {
        if (y >= midY)
        {
            if (z <= midZ)
                octant = 2;
            else
                octant = 6;
        }
        else
        {
            if (z <= midZ)
                octant = 4;
            else
                octant = 8;
        }
    }
    return octant;
}

// Calculate SFC code for a node based on its center
__device__ uint64_t calculateNodeSFCCode(Node &node, SFCType sfcType, Vector min, Vector max)
{
    // Calculate center of node
    Vector center = {
        (node.topLeftFront.x + node.botRightBack.x) / 2.0,
        (node.topLeftFront.y + node.botRightBack.y) / 2.0,
        (node.topLeftFront.z + node.botRightBack.z) / 2.0};

    // Calculate SFC code based on center
    if (sfcType == MORTON)
    {
        return positionToMortonCode(center, min, max);
    }
    else if (sfcType == HILBERT)
    {
        return positionToHilbertCode(center, min, max);
    }

    return 0; // Default for NO_SFC
}

__device__ void UpdateChildBound(Vector &tlf, Vector &brb, Node &childNode, int octant)
{
    double midX = (tlf.x + brb.x) / 2;
    double midY = (tlf.y + brb.y) / 2;
    double midZ = (tlf.z + brb.z) / 2;

    switch (octant)
    {
    case 1: // top-left-front
        childNode.topLeftFront = tlf;
        childNode.botRightBack = {midX, midY, midZ};
        break;
    case 2: // top-right-front
        childNode.topLeftFront = {midX, tlf.y, tlf.z};
        childNode.botRightBack = {brb.x, midY, midZ};
        break;
    case 3: // bottom-left-front
        childNode.topLeftFront = {tlf.x, midY, tlf.z};
        childNode.botRightBack = {midX, brb.y, midZ};
        break;
    case 4: // bottom-right-front
        childNode.topLeftFront = {midX, midY, tlf.z};
        childNode.botRightBack = {brb.x, brb.y, midZ};
        break;
    case 5: // top-left-back
        childNode.topLeftFront = {tlf.x, tlf.y, midZ};
        childNode.botRightBack = {midX, midY, brb.z};
        break;
    case 6: // top-right-back
        childNode.topLeftFront = {midX, tlf.y, midZ};
        childNode.botRightBack = {brb.x, midY, brb.z};
        break;
    case 7: // bottom-left-back
        childNode.topLeftFront = {tlf.x, midY, midZ};
        childNode.botRightBack = {midX, brb.y, brb.z};
        break;
    case 8: // bottom-right-back
        childNode.topLeftFront = {midX, midY, midZ};
        childNode.botRightBack = brb;
        break;
    }
}

__device__ void warpReduce(volatile double *totalMass, volatile double3 *centerMass, int tx)
{
    totalMass[tx] += totalMass[tx + 32];
    centerMass[tx].x += centerMass[tx + 32].x;
    centerMass[tx].y += centerMass[tx + 32].y;
    centerMass[tx].z += centerMass[tx + 32].z;

    totalMass[tx] += totalMass[tx + 16];
    centerMass[tx].x += centerMass[tx + 16].x;
    centerMass[tx].y += centerMass[tx + 16].y;
    centerMass[tx].z += centerMass[tx + 16].z;

    totalMass[tx] += totalMass[tx + 8];
    centerMass[tx].x += centerMass[tx + 8].x;
    centerMass[tx].y += centerMass[tx + 8].y;
    centerMass[tx].z += centerMass[tx + 8].z;

    totalMass[tx] += totalMass[tx + 4];
    centerMass[tx].x += centerMass[tx + 4].x;
    centerMass[tx].y += centerMass[tx + 4].y;
    centerMass[tx].z += centerMass[tx + 4].z;

    totalMass[tx] += totalMass[tx + 2];
    centerMass[tx].x += centerMass[tx + 2].x;
    centerMass[tx].y += centerMass[tx + 2].y;
    centerMass[tx].z += centerMass[tx + 2].z;

    totalMass[tx] += totalMass[tx + 1];
    centerMass[tx].x += centerMass[tx + 1].x;
    centerMass[tx].y += centerMass[tx + 1].y;
    centerMass[tx].z += centerMass[tx + 1].z;
}

__device__ void ComputeCenterMass(Node &curNode, Body *bodies, double *totalMass, double3 *centerMass, int start, int end)
{
    int tx = threadIdx.x;
    int total = end - start + 1;
    int sz = ceil((double)total / blockDim.x);
    int s = tx * sz + start;
    double M = 0.0;
    double3 R = make_double3(0.0, 0.0, 0.0);

    for (int i = s; i < s + sz; ++i)
    {
        if (i <= end)
        {
            Body &body = bodies[i];
            M += body.mass;
            R.x += body.mass * body.position.x;
            R.y += body.mass * body.position.y;
            R.z += body.mass * body.position.z;
        }
    }

    totalMass[tx] = M;
    centerMass[tx] = R;

    for (unsigned int stride = blockDim.x / 2; stride > 32; stride >>= 1)
    {
        __syncthreads();
        if (tx < stride)
        {
            totalMass[tx] += totalMass[tx + stride];
            centerMass[tx].x += centerMass[tx + stride].x;
            centerMass[tx].y += centerMass[tx + stride].y;
            centerMass[tx].z += centerMass[tx + stride].z;
        }
    }

    if (tx < 32)
    {
        warpReduce(totalMass, centerMass, tx);
    }

    __syncthreads();

    if (tx == 0)
    {
        centerMass[0].x /= totalMass[0];
        centerMass[0].y /= totalMass[0];
        centerMass[0].z /= totalMass[0];
        curNode.totalMass = totalMass[0];
        curNode.centerMass = {centerMass[0].x, centerMass[0].y, centerMass[0].z};
    }
}

__device__ void CountBodies(Body *bodies, Vector topLeftFront, Vector botRightBack, int *count, int start, int end, int nBodies)
{
    int tx = threadIdx.x;
    if (tx < 8)
        count[tx] = 0;
    __syncthreads();

    for (int i = start + tx; i <= end; i += blockDim.x)
    {
        Body body = bodies[i];
        int oct = getOctant(topLeftFront, botRightBack, body.position.x, body.position.y, body.position.z);
        atomicAdd(&count[oct - 1], 1);
    }
    __syncthreads();
}

__device__ void ComputeOffset(int *count, int start)
{
    int tx = threadIdx.x;
    if (tx < 8)
    {
        int offset = start;
        for (int i = 0; i < tx; ++i)
        {
            offset += count[i];
        }
        count[tx + 8] = offset;
    }
    __syncthreads();
}

__device__ void GroupBodies(Body *bodies, Body *buffer, Vector topLeftFront, Vector botRightBack, int *workOffset, int start, int end, int nBodies)
{
    for (int i = start + threadIdx.x; i <= end; i += blockDim.x)
    {
        Body body = bodies[i];
        int oct = getOctant(topLeftFront, botRightBack, body.position.x, body.position.y, body.position.z);
        int dest = atomicAdd(&workOffset[oct - 1], 1);
        buffer[dest] = body;
    }
    __syncthreads();
}

// Struct to store child node indices and their SFC codes for sorting
struct ChildNode
{
    int index;
    uint64_t sfcCode;
};

// Compare two ChildNode structs by SFC code
__device__ int compareChildNodes(const void *a, const void *b)
{
    const ChildNode *nodeA = (const ChildNode *)a;
    const ChildNode *nodeB = (const ChildNode *)b;

    if (nodeA->sfcCode < nodeB->sfcCode)
        return -1;
    if (nodeA->sfcCode > nodeB->sfcCode)
        return 1;
    return 0;
}

// Kernel para construir el octree
__global__ void ConstructOctTreeKernel(Node *node, Body *bodies, Body *buffer, int nodeIndex, int nNodes, int nBodies, int leafLimit, SFCType sfcType, OrderTarget orderTarget)
{
    // Reservamos memoria compartida para 8 contadores y 8 offsets (total 16 enteros)
    __shared__ int count[16]; // count[0..7]: cantidad de cuerpos por octante; count[8..15]: offsets base
    __shared__ double totalMass[BLOCK_SIZE];
    __shared__ double3 centerMass[BLOCK_SIZE];
    // Memory for storing child nodes for SFC sorting
    __shared__ ChildNode childNodes[8];

    int tx = threadIdx.x;
    // Ajustar el índice del nodo según el bloque
    nodeIndex += blockIdx.x;
    if (nodeIndex >= nNodes)
        return;

    Node &curNode = node[nodeIndex];
    int start = curNode.start;
    int end = curNode.end;
    Vector topLeftFront = curNode.topLeftFront;
    Vector botRightBack = curNode.botRightBack;

    // Get global min/max for SFC calculations
    Vector globalMin = node[0].topLeftFront;
    Vector globalMax = node[0].botRightBack;

    if (start == -1 && end == -1)
        return;

    // Calcula el centro de masa para el nodo actual (actualiza curNode)
    ComputeCenterMass(curNode, bodies, totalMass, centerMass, start, end);

    // If using SFC, calculate the SFC code for this node
    if (sfcType != NO_SFC)
    {
        if (sfcType == MORTON)
        {
            curNode.sfcCode = positionToMortonCode(curNode.centerMass, globalMin, globalMax);
        }
        else if (sfcType == HILBERT)
        {
            curNode.sfcCode = positionToHilbertCode(curNode.centerMass, globalMin, globalMax);
        }
    }

    // Si ya se alcanzó el límite de subdivisión o hay un único cuerpo, copiamos el bloque y retornamos
    if (nodeIndex >= leafLimit || start == end)
    {
        for (int i = start; i <= end; ++i)
        {
            buffer[i] = bodies[i];
        }
        return;
    }

    // Paso 1: contar la cantidad de cuerpos en cada octante.
    CountBodies(bodies, topLeftFront, botRightBack, count, start, end, nBodies);
    // Paso 2: calcular los offsets base a partir de 'start'
    ComputeOffset(count, start);

    // Copiar los offsets base (calculados en count[8..15]) a un arreglo compartido para usarlos en la asignación de nodos hijos.
    __shared__ int baseOffset[8];
    __shared__ int workOffset[8]; // copia que se usará para las operaciones atómicas en GroupBodies
    if (tx < 8)
    {
        baseOffset[tx] = count[tx + 8];  // guardar el offset original para el octante tx
        workOffset[tx] = baseOffset[tx]; // inicializar la copia de trabajo
    }
    __syncthreads();

    // Paso 3: agrupar cuerpos en el buffer según su octante, usando el arreglo workOffset.
    GroupBodies(bodies, buffer, topLeftFront, botRightBack, workOffset, start, end, nBodies);

    // Paso 4: asignar los rangos a los nodos hijos (únicamente en tx==0)
    if (tx == 0)
    {
        // For SFC ordering of octants
        bool usingSFCOrdering = (sfcType != NO_SFC && orderTarget == ORDER_OCTANTS);

        // For each octant
        for (int i = 0; i < 8; i++)
        {
            // Calculate child node index
            int childIdx = nodeIndex * 8 + (i + 1);

            // Setup the child node
            Node &childNode = node[childIdx];

            // Update child bounds
            UpdateChildBound(topLeftFront, botRightBack, childNode, i + 1);

            // Assign body range
            if (count[i] > 0)
            {
                childNode.start = baseOffset[i];
                childNode.end = childNode.start + count[i] - 1;
            }
            else
            {
                childNode.start = -1;
                childNode.end = -1;
            }

            // If using SFC ordering for octants, calculate SFC code and store in array for sorting
            if (usingSFCOrdering)
            {
                // Calculate center of child node for SFC code
                Vector childCenter = {
                    (childNode.topLeftFront.x + childNode.botRightBack.x) / 2.0,
                    (childNode.topLeftFront.y + childNode.botRightBack.y) / 2.0,
                    (childNode.topLeftFront.z + childNode.botRightBack.z) / 2.0};

                // Calculate SFC code
                uint64_t code = 0;
                if (sfcType == MORTON)
                {
                    code = positionToMortonCode(childCenter, globalMin, globalMax);
                }
                else if (sfcType == HILBERT)
                {
                    code = positionToHilbertCode(childCenter, globalMin, globalMax);
                }

                // Store in array for sorting
                childNodes[i].index = i;
                childNodes[i].sfcCode = code;
                childNode.sfcCode = code;
            }
        }

        // If using SFC ordering for octants, sort child nodes by SFC code
        if (usingSFCOrdering)
        {
            // Simple insertion sort (for small arrays like this, it's efficient enough)
            for (int i = 1; i < 8; i++)
            {
                ChildNode key = childNodes[i];
                int j = i - 1;

                while (j >= 0 && childNodes[j].sfcCode > key.sfcCode)
                {
                    childNodes[j + 1] = childNodes[j];
                    j--;
                }

                childNodes[j + 1] = key;
            }
        }

        curNode.isLeaf = false;

        // Launch recursion for children nodes
        // If using SFC ordering for octants, launch in sorted order
        if (usingSFCOrdering)
        {
            // Launch child kernels in the SFC-sorted order
            for (int i = 0; i < 8; i++)
            {
                int childOctant = childNodes[i].index;
                int childIdx = nodeIndex * 8 + (childOctant + 1);

                // Only launch if this child has bodies
                if (node[childIdx].start != -1 && node[childIdx].end != -1)
                {
                    ConstructOctTreeKernel<<<1, BLOCK_SIZE>>>(
                        node, buffer, bodies, childIdx, nNodes, nBodies, leafLimit, sfcType, orderTarget);
                }
            }
        }
        else
        {
            // Launch all child kernels at once (original behavior)
            ConstructOctTreeKernel<<<8, BLOCK_SIZE>>>(
                node, buffer, bodies, nodeIndex * 8 + 1, nNodes, nBodies, leafLimit, sfcType, orderTarget);
        }
    }
}

/*
----------------------------------------------------------------------------------------
COMPUTE FORCE
----------------------------------------------------------------------------------------
*/
__device__ double getDistance(Vector pos1, Vector pos2)
{
    return sqrt(pow(pos1.x - pos2.x, 2) + pow(pos1.y - pos2.y, 2) + pow(pos1.z - pos2.z, 2));
}

__device__ bool isCollide(Body &b1, Vector cm)
{
    double d = getDistance(b1.position, cm);
    double threshold = b1.radius * 2 + COLLISION_TH;
    return threshold > d;
}

__device__ void ComputeForce(Node *node, Body *bodies, int nodeIndex, int bodyIndex,
                             int nNodes, int nBodies, int leafLimit, double width)
{
    if (nodeIndex >= nNodes)
        return;

    Node curNode = node[nodeIndex];
    Body bi = bodies[bodyIndex];

    // Caso de nodo hoja: usar el centro de masa para aproximar la fuerza
    if (curNode.isLeaf)
    {
        // Verificar que el centro de masa es válido (asumiendo que -1 significa no válido)
        if (curNode.centerMass.x != -1 && !isCollide(bi, curNode.centerMass))
        {
            Vector rij = {
                curNode.centerMass.x - bi.position.x,
                curNode.centerMass.y - bi.position.y,
                curNode.centerMass.z - bi.position.z};
            // Calcular r² sin suavizado
            double r2 = (rij.x * rij.x) + (rij.y * rij.y) + (rij.z * rij.z);
            // Usar la fórmula: (r^2 + E^2)^(3/2)
            double r = sqrt(r2 + (E * E));
            double f = (GRAVITY * bi.mass * curNode.totalMass) / (r * r * r + (E * E));
            Vector force = {rij.x * f, rij.y * f, rij.z * f};

            bodies[bodyIndex].acceleration.x += (force.x / bi.mass);
            bodies[bodyIndex].acceleration.y += (force.y / bi.mass);
            bodies[bodyIndex].acceleration.z += (force.z / bi.mass);
        }
        return;
    }

    // Caso de aproximación multipolo
    double distance = getDistance(bi.position, curNode.centerMass);
    double sd = width / distance; // TAMANIO DE LA REGION / DISTANCIA, kd-tree
    if (sd < THETA)
    {
        if (!isCollide(bi, curNode.centerMass))
        {
            Vector rij = {
                curNode.centerMass.x - bi.position.x,
                curNode.centerMass.y - bi.position.y,
                curNode.centerMass.z - bi.position.z};

            double r2 = (rij.x * rij.x) + (rij.y * rij.y) + (rij.z * rij.z);
            double r = sqrt(r2 + (E * E));
            double f = (GRAVITY * bi.mass * curNode.totalMass) / (r * r * r + (E * E));
            Vector force = {rij.x * f, rij.y * f, rij.z * f};

            bodies[bodyIndex].acceleration.x += (force.x / bi.mass);
            bodies[bodyIndex].acceleration.y += (force.y / bi.mass);
            bodies[bodyIndex].acceleration.z += (force.z / bi.mass);
        }
        return;
    }

    // Si no se cumple la condición de aproximación, se recorre recursivamente a los 8 hijos.
    for (int i = 1; i <= 8; i++)
    {
        ComputeForce(node, bodies, (nodeIndex * 8) + i, bodyIndex, nNodes, nBodies, leafLimit, width / 2);
    }
}

__global__ void ComputeForceKernel(Node *node, Body *bodies, int nNodes, int nBodies, int leafLimit)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double width = node[0].botRightBack.x - node[0].topLeftFront.x;
    if (i < nBodies)
    {
        Body &bi = bodies[i];
        if (bi.isDynamic)
        {
            // Reiniciar la aceleración
            bi.acceleration = {0.0, 0.0, 0.0};

            // Compute the force recursively
            ComputeForce(node, bodies, 0, i, nNodes, nBodies, leafLimit, width);

            // Update velocity and position with integration (Euler)
            bi.velocity.x += bi.acceleration.x * DT;
            bi.velocity.y += bi.acceleration.y * DT;
            bi.velocity.z += bi.acceleration.z * DT;

            bi.position.x += bi.velocity.x * DT;
            bi.position.y += bi.velocity.y * DT;
            bi.position.z += bi.velocity.z * DT;
        }
    }
}
