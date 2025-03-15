#ifndef OPENGL_RENDERER_H
#define OPENGL_RENDERER_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <vector>

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

    // Shader compilation helpers
    GLuint compileShader(GLenum type, const char *source);
    void createShaderProgram();
    void setupBuffers();
};

#endif // OPENGL_RENDERER_H