
#ifndef BARNES_HUT_CUH
#define BARNES_HUT_CUH

#include "../../base/base.cuh"
#include <cstring>
class BarnesHut : public SimulationBase
{
protected:
    int nNodes;    // Total number of nodes in the octree
    int leafLimit; // Leaf node threshold

    Node *h_nodes; // Host nodes array
    Node *d_nodes; // Device nodes array
    int *d_mutex;  // Device mutex for synchronization

    Body *d_bodiesBuffer; // Temporary buffer for octree construction

    virtual void resetOctree();
    virtual void computeBoundingBox();
    virtual void constructOctree();
    virtual void computeForces();

public:
    BarnesHut(int numBodies,
              BodyDistribution dist = BodyDistribution::SOLAR_SYSTEM,
              unsigned int seed = static_cast<unsigned int>(time(nullptr)));
    virtual ~BarnesHut();

    bool getOctreeNodes(Node* destNodes, int maxNodes) {
        if (!d_nodes || nNodes <= 0)
            return false;
            
        try {
            // Sincronizar antes de copiar para asegurar que los datos estén actualizados
            cudaDeviceSynchronize();
            
            // Aquí está el problema: Necesitamos asegurar que h_nodes esté inicializado
            if (!h_nodes) {
                std::cerr << "Error: h_nodes no está inicializado" << std::endl;
                return false;
            }
            
            // Depuración: imprimir información antes de la copia
            std::cout << "Copying " << nNodes << " nodes from device (max: " << maxNodes << ")" << std::endl;
            
            // Copiar desde la GPU a la CPU con verificación de errores
            cudaError_t err = cudaMemcpy(h_nodes, d_nodes, nNodes * sizeof(Node), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                std::cerr << "CUDA Error copying nodes from device: " << cudaGetErrorString(err) << std::endl;
                return false;
            }
            
            // Verificar el nodo raíz antes de copiarlo
            if (nNodes > 0) {
                std::cout << "Root node bounds: (" 
                          << h_nodes[0].topLeftFront.x << ", " 
                          << h_nodes[0].topLeftFront.y << ", " 
                          << h_nodes[0].topLeftFront.z << ") to ("
                          << h_nodes[0].botRightBack.x << ", " 
                          << h_nodes[0].botRightBack.y << ", " 
                          << h_nodes[0].botRightBack.z << ")" << std::endl;
                          
                // Si los valores son infinitos, inicializar con un valor por defecto
                if (std::isinf(h_nodes[0].topLeftFront.x) || std::isinf(h_nodes[0].botRightBack.x)) {
                    std::cout << "Warning: Root node has infinite bounds, setting default values" << std::endl;
                    h_nodes[0].topLeftFront = Vector(-1.0e11, 1.0e11, -1.0e11);
                    h_nodes[0].botRightBack = Vector(1.0e11, -1.0e11, 1.0e11);
                }
            }
            
            // Determinar cuántos nodos copiar
            int nodesToCopy = std::min(maxNodes, nNodes);
            std::cout << "Copying " << nodesToCopy << " nodes to destination array" << std::endl;
            
            // Copiar nodos a la estructura de destino
            memcpy(destNodes, h_nodes, nodesToCopy * sizeof(Node));
            return true;
        } catch(const std::exception& e) {
            std::cerr << "Exception in getOctreeNodes: " << e.what() << std::endl;
            return false;
        }
    }
    
    int getNumNodes() const {
        return nNodes;
    }
    
    int getRootNodeIndex() const {
        return 0;
    }

    virtual void update() override;
};

#endif // BARNES_HUT_CUH