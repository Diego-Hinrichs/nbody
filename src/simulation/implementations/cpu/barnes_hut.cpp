#include "../../include/simulation/implementations/cpu/barnes_hut.hpp"
#include "../../include/common/types.cuh"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <limits>

CPUBarnesHut::CPUBarnesHut(int numBodies, bool useParallelization, int threads,
                         BodyDistribution dist, unsigned int seed,
                         bool enableSFC, SFCOrderingMode sfcOrderingMode, int reorderFreq)
    : SimulationBase(numBodies, dist, seed),
      useOpenMP(useParallelization),
      root(nullptr),
      useSFC(enableSFC),
      orderingMode(sfcOrderingMode),
      reorderFrequency(reorderFreq),
      iterationCounter(0)
{
    // Initialize thread count
    setThreadCount(threads);

    // Set initial bounds to invalid values to force computation
    minBound = Vector(std::numeric_limits<double>::max(), 
                      std::numeric_limits<double>::max(), 
                      std::numeric_limits<double>::max());
    maxBound = Vector(std::numeric_limits<double>::lowest(), 
                      std::numeric_limits<double>::lowest(), 
                      std::numeric_limits<double>::lowest());

    // Log configuration
    std::cout << "CPU Barnes-Hut Simulation created with " << numBodies << " bodies." << std::endl;
    if (useOpenMP) {
        std::cout << "OpenMP enabled with " << numThreads << " threads." << std::endl;
    } else {
        std::cout << "OpenMP disabled, using single-threaded mode." << std::endl;
    }
}

CPUBarnesHut::~CPUBarnesHut()
{
    // Node cleanup is handled by the unique_ptr
}

void CPUBarnesHut::computeBoundingBox()
{
    // Measure execution time
    CudaTimer timer(metrics.bboxTimeMs);
    
    // Copy bodies to host if they're not already there
    copyBodiesFromDevice();
    
    // Reset bounds
    minBound = Vector(std::numeric_limits<double>::max(), 
                      std::numeric_limits<double>::max(), 
                      std::numeric_limits<double>::max());
    maxBound = Vector(std::numeric_limits<double>::lowest(), 
                      std::numeric_limits<double>::lowest(), 
                      std::numeric_limits<double>::lowest());
    
    // Find the minimum and maximum coordinates
    if (useOpenMP) {
        // Set the number of threads
        omp_set_num_threads(numThreads);
        
        // Shared variables for reduction
        Vector localMin[numThreads];
        Vector localMax[numThreads];
        
        // Initialize local bounds
        for (int i = 0; i < numThreads; i++) {
            localMin[i] = Vector(std::numeric_limits<double>::max(), 
                                std::numeric_limits<double>::max(), 
                                std::numeric_limits<double>::max());
            localMax[i] = Vector(std::numeric_limits<double>::lowest(), 
                                std::numeric_limits<double>::lowest(), 
                                std::numeric_limits<double>::lowest());
        }
        
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            
            #pragma omp for
            for (int i = 0; i < nBodies; i++) {
                // Update local min coords
                localMin[tid].x = std::min(localMin[tid].x, h_bodies[i].position.x);
                localMin[tid].y = std::min(localMin[tid].y, h_bodies[i].position.y);
                localMin[tid].z = std::min(localMin[tid].z, h_bodies[i].position.z);
                
                // Update local max coords
                localMax[tid].x = std::max(localMax[tid].x, h_bodies[i].position.x);
                localMax[tid].y = std::max(localMax[tid].y, h_bodies[i].position.y);
                localMax[tid].z = std::max(localMax[tid].z, h_bodies[i].position.z);
            }
        }
        
        // Combine results from all threads
        for (int i = 0; i < numThreads; i++) {
            minBound.x = std::min(minBound.x, localMin[i].x);
            minBound.y = std::min(minBound.y, localMin[i].y);
            minBound.z = std::min(minBound.z, localMin[i].z);
            
            maxBound.x = std::max(maxBound.x, localMax[i].x);
            maxBound.y = std::max(maxBound.y, localMax[i].y);
            maxBound.z = std::max(maxBound.z, localMax[i].z);
        }
    } else {
        // Single-threaded computation
        for (int i = 0; i < nBodies; i++) {
            // Update minimum bounds
            minBound.x = std::min(minBound.x, h_bodies[i].position.x);
            minBound.y = std::min(minBound.y, h_bodies[i].position.y);
            minBound.z = std::min(minBound.z, h_bodies[i].position.z);
            
            // Update maximum bounds
            maxBound.x = std::max(maxBound.x, h_bodies[i].position.x);
            maxBound.y = std::max(maxBound.y, h_bodies[i].position.y);
            maxBound.z = std::max(maxBound.z, h_bodies[i].position.z);
        }
    }
    
    // Add some padding to avoid edge cases
    double padding = std::max(1.0e10, (maxBound.x - minBound.x) * 0.01);
    minBound.x -= padding;
    minBound.y -= padding;
    minBound.z -= padding;
    maxBound.x += padding;
    maxBound.y += padding;
    maxBound.z += padding;
    
    // Ensure the bounding box is a cube (same width in all dimensions)
    double sizeX = maxBound.x - minBound.x;
    double sizeY = maxBound.y - minBound.y;
    double sizeZ = maxBound.z - minBound.z;
    double maxSize = std::max(std::max(sizeX, sizeY), sizeZ);
    
    // Adjust bounds to make a cube
    double centerX = (minBound.x + maxBound.x) * 0.5;
    double centerY = (minBound.y + maxBound.y) * 0.5;
    double centerZ = (minBound.z + maxBound.z) * 0.5;
    
    minBound.x = centerX - maxSize * 0.5;
    minBound.y = centerY - maxSize * 0.5;
    minBound.z = centerZ - maxSize * 0.5;
    
    maxBound.x = centerX + maxSize * 0.5;
    maxBound.y = centerY + maxSize * 0.5;
    maxBound.z = centerZ + maxSize * 0.5;
}

void CPUBarnesHut::buildOctree()
{
    // Measure execution time
    CudaTimer timer(metrics.octreeTimeMs);
    
    // Clear any existing tree
    root.reset(new CPUOctreeNode());
    
    // Set the root node properties
    Vector center = Vector(
        (minBound.x + maxBound.x) * 0.5,
        (minBound.y + maxBound.y) * 0.5,
        (minBound.z + maxBound.z) * 0.5
    );
    double halfWidth = std::max(
        std::max(maxBound.x - center.x, maxBound.y - center.y),
        maxBound.z - center.z
    );
    
    root->center = center;
    root->halfWidth = halfWidth;
    
    // Insert all bodies into the octree
    for (int i = 0; i < nBodies; i++) {
        // Start at the root node
        CPUOctreeNode* node = root.get();
        
        // Keep track of the current node's level for subdivision limits
        int level = 0;
        const int MAX_LEVEL = 20; // Prevent excessive subdivision
        
        while (!node->isLeaf && level < MAX_LEVEL) {
            // Non-leaf node: determine which child octant the body belongs to
            int octant = node->getOctant(h_bodies[i].position);
            
            // Create the child if it doesn't exist
            if (!node->children[octant]) {
                node->children[octant] = new CPUOctreeNode();
                node->children[octant]->center = node->getOctantCenter(octant);
                node->children[octant]->halfWidth = node->halfWidth * 0.5;
            }
            
            // Move to the child node
            node = node->children[octant];
            level++;
        }
        
        // We've reached a leaf node
        if (node->bodyIndex == -1) {
            // Empty leaf node, store the body
            node->bodyIndex = i;
        } else {
            // Node already contains a body, subdivide further
            int existingIndex = node->bodyIndex;
            node->bodyIndex = -1;
            node->isLeaf = false;
            
            // Add the existing body to a child
            int octant1 = node->getOctant(h_bodies[existingIndex].position);
            if (!node->children[octant1]) {
                node->children[octant1] = new CPUOctreeNode();
                node->children[octant1]->center = node->getOctantCenter(octant1);
                node->children[octant1]->halfWidth = node->halfWidth * 0.5;
            }
            node->children[octant1]->bodyIndex = existingIndex;
            
            // Add the new body to a child
            int octant2 = node->getOctant(h_bodies[i].position);
            if (!node->children[octant2]) {
                node->children[octant2] = new CPUOctreeNode();
                node->children[octant2]->center = node->getOctantCenter(octant2);
                node->children[octant2]->halfWidth = node->halfWidth * 0.5;
            }
            
            // Handle the case where both bodies go to the same octant
            if (octant1 == octant2) {
                // We need to recursively insert into the child
                // Reset bodyIndex and call buildOctree again
                node->children[octant1]->bodyIndex = -1;
                node->children[octant1]->isLeaf = false;
                
                // Create grandchild for the existing body
                int subOctant1 = node->children[octant1]->getOctant(h_bodies[existingIndex].position);
                if (!node->children[octant1]->children[subOctant1]) {
                    node->children[octant1]->children[subOctant1] = new CPUOctreeNode();
                    node->children[octant1]->children[subOctant1]->center = 
                        node->children[octant1]->getOctantCenter(subOctant1);
                    node->children[octant1]->children[subOctant1]->halfWidth = 
                        node->children[octant1]->halfWidth * 0.5;
                }
                node->children[octant1]->children[subOctant1]->bodyIndex = existingIndex;
                
                // Create grandchild for the new body
                int subOctant2 = node->children[octant1]->getOctant(h_bodies[i].position);
                if (!node->children[octant1]->children[subOctant2]) {
                    node->children[octant1]->children[subOctant2] = new CPUOctreeNode();
                    node->children[octant1]->children[subOctant2]->center = 
                        node->children[octant1]->getOctantCenter(subOctant2);
                    node->children[octant1]->children[subOctant2]->halfWidth = 
                        node->children[octant1]->halfWidth * 0.5;
                }
                
                // If they still end up in the same octant, we'll handle it in the next iteration
                if (subOctant1 != subOctant2) {
                    node->children[octant1]->children[subOctant2]->bodyIndex = i;
                } else {
                    // Here we could continue the recursion, but for simplicity,
                    // we'll just add to a list of bodies in this node
                    if (node->children[octant1]->children[subOctant1]->bodies.empty()) {
                        node->children[octant1]->children[subOctant1]->bodies.push_back(existingIndex);
                    }
                    node->children[octant1]->children[subOctant1]->bodies.push_back(i);
                    node->children[octant1]->children[subOctant1]->isLeaf = true;
                    node->children[octant1]->children[subOctant1]->bodyIndex = -1;
                }
            } else {
                // Bodies go to different octants, simple case
                node->children[octant2]->bodyIndex = i;
            }
        }
    }
    
    // Compute centers of mass recursively
    computeCenterOfMass(root.get());
}

void CPUBarnesHut::computeCenterOfMass(CPUOctreeNode* node)
{
    if (!node) return;
    
    // Reset center of mass and total mass
    node->centerOfMass = Vector(0.0, 0.0, 0.0);
    node->totalMass = 0.0;
    
    if (node->isLeaf) {
        if (node->bodyIndex != -1) {
            // Single body in a leaf
            node->centerOfMass = h_bodies[node->bodyIndex].position;
            node->totalMass = h_bodies[node->bodyIndex].mass;
        } else if (!node->bodies.empty()) {
            // Multiple bodies in a leaf (edge case)
            for (int idx : node->bodies) {
                node->centerOfMass.x += h_bodies[idx].position.x * h_bodies[idx].mass;
                node->centerOfMass.y += h_bodies[idx].position.y * h_bodies[idx].mass;
                node->centerOfMass.z += h_bodies[idx].position.z * h_bodies[idx].mass;
                node->totalMass += h_bodies[idx].mass;
            }
            
            if (node->totalMass > 0.0) {
                node->centerOfMass.x /= node->totalMass;
                node->centerOfMass.y /= node->totalMass;
                node->centerOfMass.z /= node->totalMass;
            }
        }
    } else {
        // Internal node: recursively compute centers of mass for children
        for (int i = 0; i < 8; i++) {
            if (node->children[i]) {
                computeCenterOfMass(node->children[i]);
                
                // Incorporate child's mass and center of mass
                if (node->children[i]->totalMass > 0.0) {
                    node->centerOfMass.x += node->children[i]->centerOfMass.x * node->children[i]->totalMass;
                    node->centerOfMass.y += node->children[i]->centerOfMass.y * node->children[i]->totalMass;
                    node->centerOfMass.z += node->children[i]->centerOfMass.z * node->children[i]->totalMass;
                    node->totalMass += node->children[i]->totalMass;
                }
            }
        }
        
        if (node->totalMass > 0.0) {
            node->centerOfMass.x /= node->totalMass;
            node->centerOfMass.y /= node->totalMass;
            node->centerOfMass.z /= node->totalMass;
        }
    }
}

void CPUBarnesHut::computeForceFromNode(Body& body, const CPUOctreeNode* node)
{
    if (!node) return;
    
    if (node->totalMass <= 0.0) return; // Skip empty nodes
    
    // Calculate distance between body and node's center of mass
    Vector r = node->centerOfMass - body.position;
    double distSqr = r.lengthSquared();
    
    // Check if node is a leaf or if it's far enough for approximation
    if (node->isLeaf || (node->halfWidth * 2.0) / sqrt(distSqr) < THETA) {
        // Either a leaf or far enough to use approximation
        
        // Skip self-interaction
        if (node->isLeaf && node->bodyIndex != -1 && 
            body.position.x == h_bodies[node->bodyIndex].position.x &&
            body.position.y == h_bodies[node->bodyIndex].position.y &&
            body.position.z == h_bodies[node->bodyIndex].position.z) {
            return;
        }
        
        // Apply softening to avoid numerical instability
        double dist = sqrt(distSqr + (E * E));
        
        // Skip if bodies are too close (collision)
        if (dist < COLLISION_TH) return;
        
        // Gravitational force: G * m1 * m2 / r^3 * r_vector
        double forceMag = GRAVITY * body.mass * node->totalMass / (dist * dist * dist);
        
        // Update acceleration (F = ma, so a = F/m)
        body.acceleration.x += (r.x * forceMag) / body.mass;
        body.acceleration.y += (r.y * forceMag) / body.mass;
        body.acceleration.z += (r.z * forceMag) / body.mass;
    } else {
        // Internal node that's too close: recursively visit children
        for (int i = 0; i < 8; i++) {
            if (node->children[i]) {
                computeForceFromNode(body, node->children[i]);
            }
        }
    }
}

void CPUBarnesHut::computeForces()
{
    // Measure execution time
    CudaTimer timer(metrics.forceTimeMs);
    
    // Make sure the tree is built
    if (!root) {
        std::cerr << "Error: Octree not built before force computation" << std::endl;
        return;
    }
    
    // Compute forces using Barnes-Hut approximation
    if (useOpenMP) {
        // Set the number of threads
        omp_set_num_threads(numThreads);
        
        #pragma omp parallel for
        for (int i = 0; i < nBodies; i++) {
            // Skip non-dynamic bodies
            if (!h_bodies[i].isDynamic) continue;
            
            // Reset acceleration
            h_bodies[i].acceleration = Vector(0.0, 0.0, 0.0);
            
            // Compute force from the octree
            computeForceFromNode(h_bodies[i], root.get());
            
            // Update velocity (Euler integration)
            h_bodies[i].velocity.x += h_bodies[i].acceleration.x * DT;
            h_bodies[i].velocity.y += h_bodies[i].acceleration.y * DT;
            h_bodies[i].velocity.z += h_bodies[i].acceleration.z * DT;
            
            // Update position
            h_bodies[i].position.x += h_bodies[i].velocity.x * DT;
            h_bodies[i].position.y += h_bodies[i].velocity.y * DT;
            h_bodies[i].position.z += h_bodies[i].velocity.z * DT;
        }
    } else {
        // Single-threaded computation
        for (int i = 0; i < nBodies; i++) {
            // Skip non-dynamic bodies
            if (!h_bodies[i].isDynamic) continue;
            
            // Reset acceleration
            h_bodies[i].acceleration = Vector(0.0, 0.0, 0.0);
            
            // Compute force from the octree
            computeForceFromNode(h_bodies[i], root.get());
            
            // Update velocity (Euler integration)
            h_bodies[i].velocity.x += h_bodies[i].acceleration.x * DT;
            h_bodies[i].velocity.y += h_bodies[i].acceleration.y * DT;
            h_bodies[i].velocity.z += h_bodies[i].acceleration.z * DT;
            
            // Update position
            h_bodies[i].position.x += h_bodies[i].velocity.x * DT;
            h_bodies[i].position.y += h_bodies[i].velocity.y * DT;
            h_bodies[i].position.z += h_bodies[i].velocity.z * DT;
        }
    }
    
    // Copy updated bodies back to device for rendering
    copyBodiesToDevice();
}

void CPUBarnesHut::update()
{
    // Ensure initialization
    checkInitialization();

    // Measure total execution time
    CudaTimer timer(metrics.totalTimeMs);

    // Execute the Barnes-Hut algorithm steps
    computeBoundingBox();
    buildOctree();
    computeForces();
}