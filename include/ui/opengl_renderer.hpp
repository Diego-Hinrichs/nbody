#ifndef OPENGL_RENDERER_H
#define OPENGL_RENDERER_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <vector>
#include <functional>

#include "../common/types.cuh"
#include "../ui/simulation_state.hpp"

class OpenGLRenderer
{
public:
    OpenGLRenderer(SimulationState &simulationState);
    ~OpenGLRenderer();

    // Initialize OpenGL resources
    void init();

    // Update body data for rendering
    void updateBodies(Body *bodies, int numBodies);

    // Render the scene
    void render(float aspectRatio);
    void setParticleSize(float size) { particleSize = size; }
    float getParticleSize() const { return particleSize; }

    // New methods for octree visualization
    void updateOctreeVisualization(Node* nodes, int numNodes, int rootIndex, int maxDepth);
    void renderOctree(float aspectRatio);

private:
    // Reference to simulation state for dynamic parameters
    SimulationState &simulationState_;

    // OpenGL shader and buffer objects
    GLuint shaderProgram_;
    GLuint VBO_, VAO_;

    // Body position and mass data
    std::vector<glm::vec4> bodyPositions_;
    int numBodies_;
    float particleSize = 5.0f;

    // Octree visualization resources
    GLuint octreeShaderProgram_;
    GLuint octreeVBO_, octreeVAO_;
    int octreeVertexCount_;
    
    // Rendering state
    float lastAspectRatio_;

    // Shader compilation helpers
    GLuint compileShader(GLenum type, const char *source);
    void createShaderProgram();
    void setupBuffers();
    
    // Octree shader creation
    void initOctreeRenderer();
    
    // Helper method to add a node to the octree visualization
    void addNodeToOctreeVisualization(
        std::vector<float>& vertices,
        const Node& node,
        float scaleFactor,
        int currentDepth,
        int maxDepth
    );
    
    // Helper method to recursively process octree nodes
    void processOctreeNode(
        std::vector<float>& vertices,
        Node* nodes, 
        int numNodes,
        int nodeIndex, 
        float scaleFactor,
        int currentDepth, 
        int maxDepth
    );
};

#endif // OPENGL_RENDERER_H