#ifndef BARNES_HUT_KERNEL_
#define BARNES_HUT_KERNEL_
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <vector>
#include "barnes_hut.cuh"

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
        mutex[idx] = 0;
    }
    // El primer thread inicializa el nodo raíz
    if (idx == 0)
    {
        node[0].start = 0;
        node[0].end = nBodies - 1;
    }
}

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

    // Inicializar con valores extremos
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

__device__ int getOctant(Vector topLeftFront, Vector botRightBack, double x, double y, double z)
{
    int octant = 1;
    double midX = (topLeftFront.x + botRightBack.x) / 2;
    double midY = (topLeftFront.y + botRightBack.y) / 2;
    double midZ = (topLeftFront.z + botRightBack.z) / 2;

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
    // printf("Thread %d: totalMass = %f, centerMass = (%f, %f, %f)\n", tx, totalMass[tx], centerMass[tx].x, centerMass[tx].y, centerMass[tx].z);
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

// Kernel para construir el octree
__global__ void ConstructOctTreeKernel(Node *node, Body *bodies, Body *buffer, int nodeIndex, int nNodes, int nBodies, int leafLimit)
{
    // Reservamos memoria compartida para 8 contadores y 8 offsets (total 16 enteros)
    __shared__ int count[16]; // count[0..7]: cantidad de cuerpos por octante; count[8..15]: offsets base
    __shared__ double totalMass[BLOCK_SIZE];
    __shared__ double3 centerMass[BLOCK_SIZE];

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

    if (start == -1 && end == -1)
        return;

    // Calcula el centro de masa para el nodo actual (actualiza curNode)
    ComputeCenterMass(curNode, bodies, totalMass, centerMass, start, end);

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
        // Para cada uno de los 8 octantes (i de 0 a 7)
        for (int i = 0; i < 8; i++)
        {
            // El hijo correspondiente se ubica en: (nodeIndex * 8 + (i+1))
            Node &childNode = node[nodeIndex * 8 + (i + 1)];
            // Actualizar los límites (bounding box) del hijo
            UpdateChildBound(topLeftFront, botRightBack, childNode, i + 1);
            if (count[i] > 0)
            {
                // Asignar el rango usando el offset base
                childNode.start = baseOffset[i];
                childNode.end = childNode.start + count[i] - 1;
            }
            else
            {
                childNode.start = -1;
                childNode.end = -1;
            }
        }

        curNode.isLeaf = false;
        // Lanzar la recursión para los hijos: se usan 8 bloques
        ConstructOctTreeKernel<<<8, BLOCK_SIZE>>>(node, buffer, bodies, nodeIndex * 8 + 1, nNodes, nBodies, leafLimit);
    }
}

__device__ double getDistance(Vector pos1, Vector pos2)
{
    return sqrt((pos1.x - pos2.x) * (pos1.x - pos2.x) +
                (pos1.y - pos2.y) * (pos1.y - pos2.y) +
                (pos1.z - pos2.z) * (pos1.z - pos2.z));
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

    // Skip empty nodes
    if (curNode.start == -1 && curNode.end == -1)
        return;

    // Skip if computing force on itself
    if (curNode.isLeaf && curNode.start == bodyIndex && curNode.end == bodyIndex)
        return;

    // If it's a leaf or can use multipole approximation
    if (curNode.isLeaf || (curNode.totalMass > 0.0 && width / getDistance(bi.position, curNode.centerMass) < THETA))
    {
        // Skip if collision detected
        if (curNode.totalMass <= 0.0 || isCollide(bi, curNode.centerMass))
            return;

        // Calculate force
        Vector rij = {
            curNode.centerMass.x - bi.position.x,
            curNode.centerMass.y - bi.position.y,
            curNode.centerMass.z - bi.position.z};

        double r2 = (rij.x * rij.x) + (rij.y * rij.y) + (rij.z * rij.z);
        double r = sqrt(r2 + (E * E));

        // Use correct formula: G * m1 * m2 / r^3
        double f = (GRAVITY * bi.mass * curNode.totalMass) / (r * r * r);

        // Apply force
        bodies[bodyIndex].acceleration.x += (rij.x * f / bi.mass);
        bodies[bodyIndex].acceleration.y += (rij.y * f / bi.mass);
        bodies[bodyIndex].acceleration.z += (rij.z * f / bi.mass);

        return;
    }

    // Recurse through child nodes
    for (int i = 1; i <= 8; i++)
    {
        int childIndex = (nodeIndex * 8) + i;
        if (childIndex < nNodes)
        {
            ComputeForce(node, bodies, childIndex, bodyIndex, nNodes, nBodies, leafLimit, width / 2);
        }
    }
}

__global__ void ComputeForceKernel(Node *node, Body *bodies, int nNodes, int nBodies, int leafLimit)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= nBodies)
        return;

    // Calculate domain width (needed for multipole criterion)
    double width = max(
        fabs(node[0].botRightBack.x - node[0].topLeftFront.x),
        max(
            fabs(node[0].botRightBack.y - node[0].topLeftFront.y),
            fabs(node[0].botRightBack.z - node[0].topLeftFront.z)));

    Body &bi = bodies[i];

    if (bi.isDynamic)
    {
        // Reset acceleration
        bi.acceleration = {0.0, 0.0, 0.0};

        // Compute forces from octree
        ComputeForce(node, bodies, 0, i, nNodes, nBodies, leafLimit, width);

        // Update velocity (Euler integration)
        bi.velocity.x += bi.acceleration.x * DT;
        bi.velocity.y += bi.acceleration.y * DT;
        bi.velocity.z += bi.acceleration.z * DT;

        // Update position
        bi.position.x += bi.velocity.x * DT;
        bi.position.y += bi.velocity.y * DT;
        bi.position.z += bi.velocity.z * DT;
    }
}

#endif // BARNES_HUT_KERNEL_