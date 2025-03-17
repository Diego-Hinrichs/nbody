#include "../../include/common/types.cuh"
#include "../../include/common/constants.cuh"

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
    // printf("Thread %d: totalMass = %f, centerMass = (%f, %f, %f)\n", tx, totalMass[tx], centerMass[tx].x, centerMass[tx].y, centerMass[tx].z);
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

// __global__ void ConstructOctTreeKernel(Node *node, Body *bodies, Body *buffer, int nodeIndex, int nNodes, int nBodies, int leafLimit)
// {
//     // Reservamos memoria compartida para 8 contadores y 8 offsets (total 16 enteros)
//     __shared__ int count[16]; // count[0..7]: cantidad de cuerpos por octante; count[8..15]: offsets base
//     __shared__ double totalMass[BLOCK_SIZE];
//     __shared__ double3 centerMass[BLOCK_SIZE];

//     int tx = threadIdx.x;
//     // Ajustar el índice del nodo según el bloque
//     nodeIndex += blockIdx.x;
//     if (nodeIndex >= nNodes)
//         return;

//     Node &curNode = node[nodeIndex];
//     int start = curNode.start;
//     int end = curNode.end;
//     Vector topLeftFront = curNode.topLeftFront;
//     Vector botRightBack = curNode.botRightBack;

//     if (start == -1 && end == -1)
//         return;

//     // Calcula el centro de masa para el nodo actual (actualiza curNode)
//     ComputeCenterMass(curNode, bodies, totalMass, centerMass, start, end);

//     // Si ya se alcanzó el límite de subdivisión o hay un único cuerpo, copiamos el bloque y retornamos
//     if (nodeIndex >= leafLimit || start == end)
//     {
//         for (int i = start; i <= end; ++i)
//         {
//             buffer[i] = bodies[i];
//         }
//         return;
//     }

//     // Paso 1: contar la cantidad de cuerpos en cada octante.
//     CountBodies(bodies, topLeftFront, botRightBack, count, start, end, nBodies);
//     // Paso 2: calcular los offsets base a partir de 'start'
//     ComputeOffset(count, start);

//     // Copiar los offsets base (calculados en count[8..15]) a un arreglo compartido para usarlos en la asignación de nodos hijos.
//     __shared__ int baseOffset[8];
//     __shared__ int workOffset[8]; // copia que se usará para las operaciones atómicas en GroupBodies
//     if (tx < 8)
//     {
//         baseOffset[tx] = count[tx + 8];  // guardar el offset original para el octante tx
//         workOffset[tx] = baseOffset[tx]; // inicializar la copia de trabajo
//     }
//     __syncthreads();

//     // Paso 3: agrupar cuerpos en el buffer según su octante, usando el arreglo workOffset.
//     GroupBodies(bodies, buffer, topLeftFront, botRightBack, workOffset, start, end, nBodies);

//     // Paso 4: asignar los rangos a los nodos hijos (únicamente en tx==0)
//     if (tx == 0)
//     {
//         // Para cada uno de los 8 octantes (i de 0 a 7)
//         for (int i = 0; i < 8; i++)
//         {
//             // El hijo correspondiente se ubica en: (nodeIndex * 8 + (i+1))
//             Node &childNode = node[nodeIndex * 8 + (i + 1)];
//             // Actualizar los límites (bounding box) del hijo
//             UpdateChildBound(topLeftFront, botRightBack, childNode, i + 1);
//             if (count[i] > 0)
//             {
//                 // Asignar el rango usando el offset base
//                 childNode.start = baseOffset[i];
//                 childNode.end = childNode.start + count[i] - 1;
//             }
//             else
//             {
//                 childNode.start = -1;
//                 childNode.end = -1;
//             }
//         }

//         curNode.isLeaf = false;
        
//         // Implementación de paralelismo dinámico
//         // Lanzar kernels hijo para cada octante que contenga cuerpos
//         for (int i = 0; i < 8; i++) {
//             int childIndex = nodeIndex * 8 + (i + 1);
//             if (childIndex < nNodes && count[i] > 0) {
//                 // Lanzar kernel hijo con un solo bloque para este nodo hijo
//                 ConstructOctTreeKernel<<<1, BLOCK_SIZE>>>(
//                     node, buffer, bodies, childIndex, nNodes, nBodies, leafLimit);
//             }
//         }
//     }
// }

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
        // ConstructOctTreeKernel<<<8, BLOCK_SIZE>>>(node, buffer, bodies, nodeIndex * 8 + 1, nNodes, nBodies, leafLimit);
    }
}
