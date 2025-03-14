#include <iostream>
#include <cmath>
#include "barnes_hut_kernel.cuh"
#include "constants.h"
#include "err.h"
#include <vector>

BarnesHutCuda::BarnesHutCuda(int n) : nBodies(n)
{
    nNodes = MAX_NODES;
    leafLimit = MAX_NODES - N_LEAF;
    h_b = new Body[nBodies];
    h_node = new Node[nNodes];

    CHECK_CUDA_ERROR(cudaMalloc(&d_b, n * sizeof(Body)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_node, nNodes * sizeof(Node)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_mutex, nNodes * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b_buffer, n * sizeof(Body)));
}

BarnesHutCuda::~BarnesHutCuda()
{
    delete[] h_b;
    delete[] h_node;
    CHECK_CUDA_ERROR(cudaFree(d_b));
    CHECK_CUDA_ERROR(cudaFree(d_node));
    CHECK_CUDA_ERROR(cudaFree(d_mutex));
    CHECK_CUDA_ERROR(cudaFree(d_b_buffer));
}

void BarnesHutCuda::resetCUDA()
{
    int blockSize = BLOCK_SIZE;
    dim3 gridSize = ceil((float)nNodes / blockSize);
    ResetKernel<<<gridSize, blockSize>>>(d_node, d_mutex, nNodes, nBodies);

    // cudaDeviceSynchronize();

    // // Verificar errores
    // cudaError_t error = cudaGetLastError();
    // if (error != cudaSuccess)
    // {
    //     printf("Error en ResetKernel: %s\n", cudaGetErrorString(error));
    // }
}

void BarnesHutCuda::computeBoundingBoxCUDA()
{
    int blockSize = BLOCK_SIZE;
    dim3 gridSize = ceil((float)nBodies / blockSize);
    ComputeBoundingBoxKernel<<<gridSize, blockSize>>>(d_node, d_b, d_mutex, nBodies);

    // cudaDeviceSynchronize();
    // cudaError_t error = cudaGetLastError();
    // if (error != cudaSuccess)
    // {
    //     printf("Error en ComputeBoundingBoxKernel: %s\n", cudaGetErrorString(error));
    // }
}

void BarnesHutCuda::constructOctreeCUDA()
{   
    int blockSize = BLOCK_SIZE;
    ConstructOctTreeKernel<<<1, blockSize>>>(d_node, d_b, d_b_buffer, 0, nNodes, nBodies, leafLimit);
}

void BarnesHutCuda::computeForceCUDA()
{
    int blockSize = 32;
    dim3 gridSize = ceil((float)nBodies / blockSize);
    ComputeForceKernel<<<gridSize, blockSize>>>(d_node, d_b, nNodes, nBodies, leafLimit);
}

void BarnesHutCuda::setBody(int i, bool isDynamic, double mass, double radius, Vector position, Vector velocity, Vector acceleration)
{
    h_b[i].isDynamic = isDynamic;
    h_b[i].mass = mass;
    h_b[i].radius = radius;
    h_b[i].position = position;
    h_b[i].velocity = velocity;
    h_b[i].acceleration = acceleration;
}

Body *BarnesHutCuda::getBodies()
{
    return h_b;
}

void BarnesHutCuda::readDeviceBodies()
{
    // Copia de la información de los cuerpos de la GPU al host
    CHECK_CUDA_ERROR(cudaMemcpy(h_b, d_b, sizeof(Body) * nBodies, cudaMemcpyDeviceToHost));
}

void BarnesHutCuda::setup(int sim)
{
    // Inicializa los cuerpos de manera aleatoria (excepto el sol)
    initRandomBodies();

    // Copia la información al dispositivo
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b, sizeof(Body) * nBodies, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_node, h_node, sizeof(Node) * nNodes, cudaMemcpyHostToDevice));
}

// void BarnesHutCuda::update()
// {
//     resetCUDA();
//     computeBoundingBoxCUDA();
//     constructOctreeCUDA();
//     computeForceCUDA();
//     CHECK_LAST_CUDA_ERROR();
// }

UpdateTimes BarnesHutCuda::update()
{
    // Eventos para medir cada fase
    cudaEvent_t startAll, afterReset, afterBBox, afterOctree, afterForce;
    cudaEventCreate(&startAll);
    cudaEventCreate(&afterReset);
    cudaEventCreate(&afterBBox);
    cudaEventCreate(&afterOctree);
    cudaEventCreate(&afterForce);

    // Marcar el inicio de todas las operaciones GPU de update()
    cudaEventRecord(startAll);

    // 1. resetCUDA()
    resetCUDA();
    cudaEventRecord(afterReset);

    // 2. computeBoundingBoxCUDA()
    computeBoundingBoxCUDA();
    cudaEventRecord(afterBBox);

    // 3. constructOctreeCUDA()
    constructOctreeCUDA();
    cudaEventRecord(afterOctree);

    // 4. computeForceCUDA()
    computeForceCUDA();
    cudaEventRecord(afterForce);

    // Sincronizar para asegurarnos de que los kernels terminaron
    cudaEventSynchronize(afterForce);

    // Calcular tiempos entre eventos (ms)
    UpdateTimes times;
    cudaEventElapsedTime(&times.resetTimeMs,   startAll,     afterReset);
    cudaEventElapsedTime(&times.bboxTimeMs,    afterReset,   afterBBox);
    cudaEventElapsedTime(&times.octreeTimeMs,  afterBBox,    afterOctree);
    cudaEventElapsedTime(&times.forceTimeMs,   afterOctree,  afterForce);

    // Liberar eventos
    cudaEventDestroy(startAll);
    cudaEventDestroy(afterReset);
    cudaEventDestroy(afterBBox);
    cudaEventDestroy(afterOctree);
    cudaEventDestroy(afterForce);

    // Chequeo de errores de CUDA (si corresponde)
    CHECK_LAST_CUDA_ERROR();

    // Retornar la estructura con los tiempos
    return times;
}

void BarnesHutCuda::initRandomBodies()
{
    // Inicializar la semilla para números aleatorios
    srand(time(NULL));

    double maxDistance = MAX_DIST;
    double minDistance = MIN_DIST;

    Vector centerPos = {CENTERX, CENTERY, CENTERZ};

    // Generar cuerpos (por ejemplo, planetas) de forma dinámica
    for (int i = 0; i < nBodies; ++i)
    {
        // Generar dos números aleatorios uniformes entre 0 y 1
        double u = rand() / (double)RAND_MAX; // Para theta
        double v = rand() / (double)RAND_MAX; // Para phi

        // Theta: ángulo en el plano XY (0 a 2π)
        double theta = 2.0 * M_PI * u;
        // Phi: ángulo desde el eje Z; para una distribución uniforme sobre la esfera,
        // se usa phi = acos(2*v - 1)
        double phi = acos(2.0 * v - 1.0);

        // Generar un radio aleatorio entre minDistance y maxDistance
        double radius = (maxDistance - minDistance) * (rand() / (double)RAND_MAX) + minDistance;

        // Convertir de coordenadas esféricas a cartesianas:
        double x = centerPos.x + radius * sin(phi) * cos(theta);
        double y = centerPos.y + radius * sin(phi) * sin(theta);
        double z = centerPos.z + radius * cos(phi);

        Vector position = {x, y, z};

        // Configurar el cuerpo (por ejemplo, un planeta) como dinámico
        h_b[i].isDynamic    = true;
        h_b[i].mass         = SUN_MASS;
        h_b[i].radius       = SUN_DIA;
        h_b[i].position     = position;
        h_b[i].velocity     = {0.0, 0.0, 0.0};
        h_b[i].acceleration = {0.0, 0.0, 0.0};
    }

    // El último cuerpo se asigna como el sol (o cuerpo central)
    // h_b[nBodies - 1].isDynamic    = false;
    // h_b[nBodies - 1].mass         = SUN_MASS;
    // h_b[nBodies - 1].radius       = SUN_DIA;
    // h_b[nBodies - 1].position     = centerPos;
    // h_b[nBodies - 1].velocity     = {0.0, 0.0, 0.0};
    // h_b[nBodies - 1].acceleration = {0.0, 0.0, 0.0};
}


void BarnesHutCuda::debugPrintDeviceBodies()
{
    // Crear un vector temporal en host para almacenar los cuerpos.
    std::vector<Body> tempBodies(nBodies);

    // Copiar los cuerpos desde el dispositivo (d_b) al host.
    cudaError_t err = cudaMemcpy(tempBodies.data(), d_b, nBodies * sizeof(Body), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        std::cerr << "Error al copiar cuerpos desde el dispositivo: "
                  << cudaGetErrorString(err) << std::endl;
        return;
    }

    std::cout << "===== Cuerpos en el Dispositivo =====" << std::endl;
    for (int i = 0; i < nBodies; i++)
    {
        std::cout << "Cuerpo " << i << ":" << std::endl;
        std::cout << "\tPosición: ("
                  << tempBodies[i].position.x << ", "
                  << tempBodies[i].position.y << ", "
                  << tempBodies[i].position.z << ")" << std::endl;
        std::cout << "\tVelocidad: ("
                  << tempBodies[i].velocity.x << ", "
                  << tempBodies[i].velocity.y << ", "
                  << tempBodies[i].velocity.z << ")" << std::endl;
        std::cout << "\tAceleración: ("
                  << tempBodies[i].acceleration.x << ", "
                  << tempBodies[i].acceleration.y << ", "
                  << tempBodies[i].acceleration.z << ")" << std::endl;
        std::cout << "\tMasa: " << tempBodies[i].mass
                  << ", Radio: " << tempBodies[i].radius
                  << ", Dinámico: " << (tempBodies[i].isDynamic ? "Sí" : "No")
                  << std::endl;
    }
}

//---------------------------------------------------------------------
// Función: debugPrintTree()
//---------------------------------------------------------------------
// Copia el árbol Barnes-Hut (almacenado en d_node) desde la memoria
// del dispositivo a una estructura temporal en el host y lo imprime.
// Se asume que 'nNodes' contiene la cantidad de nodos creados y que
// el nodo 0 corresponde a la raíz (con el bounding box global).
//---------------------------------------------------------------------
void BarnesHutCuda::debugPrintTree()
{
    if (nNodes <= 0)
    {
        std::cout << "El árbol Barnes-Hut está vacío. nNodes = " << nNodes << std::endl;
        return;
    }

    // Crear un vector temporal en host para almacenar los nodos del árbol.
    std::vector<Node> tempNodes(nNodes);

    // Copiar los nodos desde el dispositivo (d_node) al host.
    cudaError_t err = cudaMemcpy(tempNodes.data(), d_node, nNodes * sizeof(Node), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        std::cerr << "Error al copiar nodos desde el dispositivo: "
                  << cudaGetErrorString(err) << std::endl;
        return;
    }

    // Obtener el arreglo de cuerpos del host.
    Body *hostBodies = getBodies(); // Se asume que getBodies() devuelve el arreglo actualizado

    std::cout << "===== Árbol Barnes-Hut =====" << std::endl;
    std::cout << "Cantidad de nodos: " << nNodes << std::endl;

    // Imprimir el bounding box global usando el nodo raíz (índice 0)
    std::cout << "Bounding Box Global (nodo raíz):" << std::endl;
    std::cout << "\tTop Left Front: ("
              << tempNodes[0].topLeftFront.x << ", "
              << tempNodes[0].topLeftFront.y << ", "
              << tempNodes[0].topLeftFront.z << ")" << std::endl;
    std::cout << "\tBot Right Back: ("
              << tempNodes[0].botRightBack.x << ", "
              << tempNodes[0].botRightBack.y << ", "
              << tempNodes[0].botRightBack.z << ")" << std::endl;

    // Recorrer e imprimir cada nodo
    for (int i = 0; i < nNodes; i++)
    {
        std::cout << "Nodo " << i << ":" << std::endl;
        std::cout << "\tTop Left Front: ("
                  << tempNodes[i].topLeftFront.x << ", "
                  << tempNodes[i].topLeftFront.y << ", "
                  << tempNodes[i].topLeftFront.z << ")" << std::endl;
        std::cout << "\tBot Right Back: ("
                  << tempNodes[i].botRightBack.x << ", "
                  << tempNodes[i].botRightBack.y << ", "
                  << tempNodes[i].botRightBack.z << ")" << std::endl;
        std::cout << "\tCentro de Masa: ("
                  << tempNodes[i].centerMass.x << ", "
                  << tempNodes[i].centerMass.y << ", "
                  << tempNodes[i].centerMass.z << ")" << std::endl;
        std::cout << "\tMasa Total: " << tempNodes[i].totalMass << std::endl;
        std::cout << "\tEs Hoja: " << (tempNodes[i].isLeaf ? "Sí" : "No") << std::endl;
        std::cout << "\tRango de cuerpos: inicio = " << tempNodes[i].start
                  << ", fin = " << tempNodes[i].end << std::endl;

        // Si el nodo es hoja y tiene un rango válido de cuerpos, imprimir los cuerpos contenidos.
        if (tempNodes[i].isLeaf && tempNodes[i].start != -1 && tempNodes[i].end != -1)
        {
            std::cout << "\tCuerpos en este nodo:" << std::endl;
            for (int j = tempNodes[i].start; j <= tempNodes[i].end; j++)
            {
                // Imprimir detalles básicos del cuerpo; puedes ampliar la información si lo deseas.
                std::cout << "\t\tCuerpo " << j << ": Posición ("
                          << hostBodies[j].position.x << ", "
                          << hostBodies[j].position.y << ", "
                          << hostBodies[j].position.z << ")";
                std::cout << ", Masa " << hostBodies[j].mass;
                std::cout << ", Radio " << hostBodies[j].radius;
                std::cout << ", Dinámico: " << (hostBodies[j].isDynamic ? "Sí" : "No") << std::endl;
            }
        }
    }
}
