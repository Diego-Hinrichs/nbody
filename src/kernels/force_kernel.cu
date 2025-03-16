#include "../../include/common/types.cuh"
#include "../../include/common/constants.cuh"

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

    if (curNode.isLeaf)
    {
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
            bi.acceleration = {0.0, 0.0, 0.0};

            ComputeForce(node, bodies, 0, i, nNodes, nBodies, leafLimit, width);

            bi.velocity.x += bi.acceleration.x * DT;
            bi.velocity.y += bi.acceleration.y * DT;
            bi.velocity.z += bi.acceleration.z * DT;

            bi.position.x += bi.velocity.x * DT;
            bi.position.y += bi.velocity.y * DT;
            bi.position.z += bi.velocity.z * DT;

        }
    }
}