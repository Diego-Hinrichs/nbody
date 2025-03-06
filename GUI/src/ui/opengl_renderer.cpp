#include "opengl_renderer.h"
#include <iostream>

// Vertex shader
const char* vertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec4 aPosition; // x, y, z, mass

    uniform mat4 uProjection;
    uniform mat4 uView;
    uniform float uPointSize;

    out float vMass;

    void main() {
        gl_Position = uProjection * uView * vec4(aPosition.xyz, 1.0);
        gl_PointSize = uPointSize * (aPosition.w / 1e24); // Size based on mass
        vMass = aPosition.w;
    }
)";

// Fragment shader
const char* fragmentShaderSource = R"(
    #version 330 core
    in float vMass;
    out vec4 FragColor;

    void main() {
        // Create a circular point with soft edges
        vec2 circCoord = 2.0 * gl_PointCoord - 1.0;
        float distSquared = dot(circCoord, circCoord);
        
        if (distSquared > 1.0) {
            discard;
        }
        
        // Color gradient based on mass
        float normalizedMass = clamp(log(vMass / 1e24) / 5.0, 0.0, 1.0);
        vec3 bodyColor = mix(
            vec3(0.2, 0.4, 1.0),   // Blue for smaller bodies
            vec3(1.0, 0.6, 0.2),   // Orange for larger bodies
            normalizedMass
        );
        
        // Soft circular gradient
        float alpha = 1.0 - smoothstep(0.7, 1.0, distSquared);
        
        FragColor = vec4(bodyColor, alpha);
    }
)";

OpenGLRenderer::OpenGLRenderer(SimulationState& simulationState) 
    : simulationState_(simulationState), 
      shaderProgram_(0), 
      VBO_(0), 
      VAO_(0), 
      numBodies_(0) 
{
}

OpenGLRenderer::~OpenGLRenderer() 
{
    // Clean up OpenGL resources
    if (VAO_) glDeleteVertexArrays(1, &VAO_);
    if (VBO_) glDeleteBuffers(1, &VBO_);
    if (shaderProgram_) glDeleteProgram(shaderProgram_);
}

GLuint OpenGLRenderer::compileShader(GLenum type, const char* source) 
{
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    // Error checking
    GLint success;
    GLchar infoLog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(shader, sizeof(infoLog), nullptr, infoLog);
        std::cerr << "Shader compilation failed: " << infoLog << std::endl;
        return 0;
    }

    return shader;
}

void OpenGLRenderer::createShaderProgram() 
{
    // Compile shaders
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);

    // Create shader program
    shaderProgram_ = glCreateProgram();
    glAttachShader(shaderProgram_, vertexShader);
    glAttachShader(shaderProgram_, fragmentShader);
    glLinkProgram(shaderProgram_);

    // Error checking
    GLint success;
    GLchar infoLog[512];
    glGetProgramiv(shaderProgram_, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram_, sizeof(infoLog), nullptr, infoLog);
        std::cerr << "Shader program linking failed: " << infoLog << std::endl;
    }

    // Clean up individual shaders
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

void OpenGLRenderer::setupBuffers() 
{
    // Create VAO and VBO
    glGenVertexArrays(1, &VAO_);
    glGenBuffers(1, &VBO_);

    glBindVertexArray(VAO_);
    glBindBuffer(GL_ARRAY_BUFFER, VBO_);
    
    // Allocate buffer space
    glBufferData(GL_ARRAY_BUFFER, bodyPositions_.size() * sizeof(glm::vec4), 
                 bodyPositions_.data(), GL_DYNAMIC_DRAW);

    // Position and mass attribute
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void OpenGLRenderer::init() 
{
    // Create and compile shaders
    createShaderProgram();
}

void OpenGLRenderer::updateBodies(Body* bodies, int numBodies) 
{
    // Resize and update body positions
    bodyPositions_.resize(numBodies);
    numBodies_ = numBodies;

    for (int i = 0; i < numBodies; ++i) {
        bodyPositions_[i] = glm::vec4(
            bodies[i].position.x, 
            bodies[i].position.y, 
            bodies[i].position.z, 
            bodies[i].mass
        );
    }

    // If buffers exist, update them
    if (VAO_ && VBO_) {
        glBindBuffer(GL_ARRAY_BUFFER, VBO_);
        glBufferSubData(GL_ARRAY_BUFFER, 0, bodyPositions_.size() * sizeof(glm::vec4), bodyPositions_.data());
    } else {
        // Setup buffers if not already done
        setupBuffers();
    }
}

void OpenGLRenderer::render(float aspectRatio) 
{
    if (numBodies_ == 0) return;

    // Use the shader program
    glUseProgram(shaderProgram_);

    // Create projection matrix (perspective projection)
    glm::mat4 projection = glm::perspective(
        glm::radians(45.0f),  // Field of view
        aspectRatio,           // Aspect ratio
        1.0f,                 // Near plane
        1.0e13f               // Far plane (adjust based on simulation scale)
    );

    // Create view matrix (camera positioning)
    float zoomFactor = simulationState_.zoomFactor.load();
    glm::mat4 view = glm::lookAt(
        glm::vec3(0.0f, 0.0f, 5.0e11f / zoomFactor),  // Camera position
        glm::vec3(0.0f, 0.0f, 0.0f),                  // Look at origin
        glm::vec3(0.0f, 1.0f, 0.0f)                   // Up vector
    );

    // Set uniforms
    GLint projLoc = glGetUniformLocation(shaderProgram_, "uProjection");
    GLint viewLoc = glGetUniformLocation(shaderProgram_, "uView");
    GLint pointSizeLoc = glGetUniformLocation(shaderProgram_, "uPointSize");

    glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
    glUniform1f(pointSizeLoc, 10.0f);  // Base point size

    // Enable blending for point transparency
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Bind VAO and draw points
    glBindVertexArray(VAO_);
    glDrawArrays(GL_POINTS, 0, numBodies_);

    // Cleanup
    glBindVertexArray(0);
    glUseProgram(0);
}