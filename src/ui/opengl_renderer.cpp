#include "../../include/ui/opengl_renderer.hpp"
#include <iostream>
#include <limits>
#include <cmath>
#include <queue>
#include <tuple>

// Vertex Shader para partículas
const char *vertexShaderSource = R"(
    #version 420 core
    layout (location = 0) in vec3 aPosition;
    layout (location = 1) in float aMass;

    uniform mat4 uProjection;
    uniform mat4 uView;
    uniform float uPointSize;
    uniform float uScaleFactor;

    out float vMass;
    out float vDistance;

    void main() {
        // Scale coordinates - apply scale factor to handle astronomical distances
        vec3 scaledPos = aPosition * uScaleFactor;
        
        // Calculate position in view space
        vec4 viewPos = uView * vec4(scaledPos, 1.0);
        
        // Calculate distance from camera (for falloff effects)
        vDistance = length(viewPos.xyz);
        
        // Calculate final position
        gl_Position = uProjection * viewPos;
        
        // Adjust point size based on mass and distance
        // For the sun (very massive objects)
        float massScale = 1.0;
        if (aMass > 1.0e28) {
            massScale = 4.0; // Make the sun larger
        }
        else if (aMass > 1.0e24) {
            massScale = 2.0; // Make planets visible
        }
        else {
            // For smaller objects, scale by mass logarithmically
            massScale = log(aMass / 1.0e20) * 0.2;
            if (massScale < 0.1) massScale = 0.1;
        }
        
        // Apply distance attenuation - closer objects appear larger
        float distanceScale = 10.0 / (1.0 + vDistance * 0.00001);
        
        // Combine all scaling factors
        gl_PointSize = min(uPointSize * 1.0 * clamp(distanceScale, 0.1, 15.0), 10.0);
        
        // Pass mass to fragment shader
        vMass = aMass;
    }
)";

// Fragment Shader para partículas
const char *fragmentShaderSource = R"(
    #version 420 core
    in float vMass;
    in float vDistance;
    out vec4 FragColor;
    
    void main() {
        // Use a pure white color for all particles
        vec3 bodyColor = vec3(1.0, 1.0, 1.0);
        
        // Create circular point
        vec2 circCoord = 2.0 * gl_PointCoord - 1.0;
        float distSquared = dot(circCoord, circCoord);
        
        // Discard fragments outside the circle
        if (distSquared > 1.0) {
            discard;
        }
        
        // Simply output white with full opacity
        FragColor = vec4(bodyColor, 1.0);
    }
)";

const char *octreeVertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec3 aPosition;
    
    uniform mat4 uProjection;
    uniform mat4 uView;
    
    void main() {
        gl_Position = uProjection * uView * vec4(aPosition, 1.0);
    }
)";

// Fragment Shader simplificado para octree con color fijo brillante
const char *octreeFragmentShaderSource = R"(
    #version 330 core
    out vec4 FragColor;
    
    void main() {
        // Color brillante para asegurar visibilidad
        FragColor = vec4(0.0, 1.0, 1.0, 1.0); // Cian brillante
    }
)";

OpenGLRenderer::OpenGLRenderer(SimulationState &simulationState)
    : simulationState_(simulationState),
      shaderProgram_(0),
      VBO_(0),
      VAO_(0),
      numBodies_(0),
      octreeShaderProgram_(0),
      octreeVBO_(0),
      octreeVAO_(0),
      octreeVertexCount_(0),
      lastAspectRatio_(1.0f)
{
}

OpenGLRenderer::~OpenGLRenderer()
{
    // Clean up OpenGL resources
    if (VAO_)
        glDeleteVertexArrays(1, &VAO_);
    if (VBO_)
        glDeleteBuffers(1, &VBO_);
    if (shaderProgram_)
        glDeleteProgram(shaderProgram_);

    // Limpiar recursos del octree
    if (octreeVAO_)
        glDeleteVertexArrays(1, &octreeVAO_);
    if (octreeVBO_)
        glDeleteBuffers(1, &octreeVBO_);
    if (octreeShaderProgram_)
        glDeleteProgram(octreeShaderProgram_);
}

GLuint OpenGLRenderer::compileShader(GLenum type, const char *source)
{
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    // Error checking
    GLint success;
    GLchar infoLog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
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

    if (!vertexShader || !fragmentShader)
    {
        std::cerr << "Failed to compile shaders" << std::endl;
        return;
    }

    // Create shader program
    shaderProgram_ = glCreateProgram();
    glAttachShader(shaderProgram_, vertexShader);
    glAttachShader(shaderProgram_, fragmentShader);
    glLinkProgram(shaderProgram_);

    // Check for linking errors
    GLint success;
    GLchar infoLog[512];
    glGetProgramiv(shaderProgram_, GL_LINK_STATUS, &success);
    if (!success)
    {
        glGetProgramInfoLog(shaderProgram_, sizeof(infoLog), nullptr, infoLog);
        std::cerr << "ERROR: Shader program linking failed\n"
                  << infoLog << std::endl;
    }

    // Clean up individual shaders
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

void OpenGLRenderer::setupBuffers()
{
    // Create VAO and VBO for particles
    glGenVertexArrays(1, &VAO_);
    glGenBuffers(1, &VBO_);
}

void OpenGLRenderer::initOctreeRenderer()
{
    // Compile the shaders for octree
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, octreeVertexShaderSource);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, octreeFragmentShaderSource);

    // Verify compilation
    if (!vertexShader || !fragmentShader)
    {
        std::cerr << "ERROR: Failed to compile octree shaders" << std::endl;
        return;
    }

    // Create shader program
    octreeShaderProgram_ = glCreateProgram();
    glAttachShader(octreeShaderProgram_, vertexShader);
    glAttachShader(octreeShaderProgram_, fragmentShader);
    glLinkProgram(octreeShaderProgram_);

    // Verify linking
    GLint success;
    GLchar infoLog[512];
    glGetProgramiv(octreeShaderProgram_, GL_LINK_STATUS, &success);
    if (!success)
    {
        glGetProgramInfoLog(octreeShaderProgram_, sizeof(infoLog), nullptr, infoLog);
        std::cerr << "ERROR: Octree shader program linking failed\n"
                  << infoLog << std::endl;
        octreeShaderProgram_ = 0;
    }
    else
    {
        std::cout << "Octree shader program compiled and linked successfully: ID="
                  << octreeShaderProgram_ << std::endl;
    }

    // Clean up shaders
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // Create VAO and VBO for the octree
    glGenVertexArrays(1, &octreeVAO_);
    glGenBuffers(1, &octreeVBO_);
}

void OpenGLRenderer::init()
{
    // Check OpenGL capabilities
    GLint maxVertexAttribs, maxTextureUnits, maxTextureSize;
    glGetIntegerv(GL_MAX_VERTEX_ATTRIBS, &maxVertexAttribs);
    glGetIntegerv(GL_MAX_TEXTURE_UNITS, &maxTextureUnits);
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxTextureSize);

    std::cout << "Max Vertex Attributes: " << maxVertexAttribs << std::endl;
    std::cout << "Max Texture Units: " << maxTextureUnits << std::endl;
    std::cout << "Max Texture Size: " << maxTextureSize << std::endl;

    // Validate OpenGL error state before initialization
    GLenum err = glGetError();
    if (err != GL_NO_ERROR)
    {
        std::cerr << "OpenGL error before initialization: " << err << std::endl;
    }

    // Advanced OpenGL configuration
    try
    {
        // Basic OpenGL configuration with extensive error checking
        glEnable(GL_DEPTH_TEST);
        err = glGetError();
        if (err != GL_NO_ERROR)
            std::cerr << "Error enabling depth test: " << err << std::endl;

        glEnable(GL_PROGRAM_POINT_SIZE);
        err = glGetError();
        if (err != GL_NO_ERROR)
            std::cerr << "Error enabling point size: " << err << std::endl;

        glEnable(GL_BLEND);
        err = glGetError();
        if (err != GL_NO_ERROR)
            std::cerr << "Error enabling blending: " << err << std::endl;

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        err = glGetError();
        if (err != GL_NO_ERROR)
            std::cerr << "Error setting blend function: " << err << std::endl;

        glfwSwapInterval(1);

        // Create the shader program with comprehensive error checking
        createShaderProgram();

        // Setup buffers with detailed logging
        setupBuffers();

        // Validate vertex array and buffer creation
        if (VAO_ == 0)
            std::cerr << "Failed to create Vertex Array Object (VAO)" << std::endl;
        if (VBO_ == 0)
            std::cerr << "Failed to create Vertex Buffer Object (VBO)" << std::endl;

        // Initial camera/view settings
        simulationState_.zoomFactor.store(0.1);
        simulationState_.offsetX = 0.0;
        simulationState_.offsetY = 0.0;

        std::cout << "OpenGL Renderer initialized successfully" << std::endl;
        std::cout << "Shader program ID: " << shaderProgram_ << std::endl;

        // Initialize octree renderer
        initOctreeRenderer();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception during OpenGL initialization: "
                  << e.what() << std::endl;
    }

    // Final error check
    err = glGetError();
    if (err != GL_NO_ERROR)
    {
        std::cerr << "Unhandled OpenGL error during initialization: " << err << std::endl;
    }
}

void OpenGLRenderer::render(float aspectRatio)
{
    lastAspectRatio_ = aspectRatio;

    if (numBodies_ == 0 || shaderProgram_ == 0)
    {
        return;
    }

    // Create projection matrix
    glm::mat4 projection = glm::perspective(
        glm::radians(45.0f), // Field of view
        aspectRatio,         // Aspect ratio
        0.1f,                // Near plane
        100.0f               // Far plane
    );

    // Get camera parameters from simulation state
    float zoomFactor = simulationState_.zoomFactor.load();
    float offsetX = simulationState_.offsetX;
    float offsetY = simulationState_.offsetY;

    // Create view matrix with adjusted camera position
    glm::mat4 view = glm::lookAt(
        glm::vec3(offsetX, offsetY, 10.0f / zoomFactor), // Camera position
        glm::vec3(offsetX, offsetY, 0.0f),              // Look at offset origin
        glm::vec3(0.0f, 1.0f, 0.0f)                     // Up vector
    );

    // Clear buffers with a nice dark background
    glClearColor(41.0f / 255.0f, 41.0f / 255.0f, 40.0f / 255.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    // Primero, dibuja las partículas
    // Use shader program
    glUseProgram(shaderProgram_);

    // Locate uniform variables
    GLint projLoc = glGetUniformLocation(shaderProgram_, "uProjection");
    GLint viewLoc = glGetUniformLocation(shaderProgram_, "uView");
    GLint pointSizeLoc = glGetUniformLocation(shaderProgram_, "uPointSize");
    GLint scaleFactorLoc = glGetUniformLocation(shaderProgram_, "uScaleFactor");

    // Set uniform values
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
    
    float adaptiveSize = particleSize * (0.1f + (0.01f / zoomFactor));
    adaptiveSize = std::min(adaptiveSize, 10.0f);
    
    glUniform1f(pointSizeLoc, 5.0f + (zoomFactor * particleSize));
    glUniform1f(scaleFactorLoc, 1.0f * zoomFactor);
    
    // Bind VAO and draw points
    glBindVertexArray(VAO_);
    glDrawArrays(GL_POINTS, 0, numBodies_);
    glBindVertexArray(0);
    glUseProgram(0);

    // Ahora, si está habilitado, dibuja el octree encima
    if (simulationState_.showOctree && octreeShaderProgram_ != 0 && octreeVertexCount_ > 0) {
        // Usar shader del octree
        glUseProgram(octreeShaderProgram_);
        
        // Configurar uniforms
        GLint octreeProjLoc = glGetUniformLocation(octreeShaderProgram_, "uProjection");
        GLint octreeViewLoc = glGetUniformLocation(octreeShaderProgram_, "uView");
        
        if (octreeProjLoc != -1)
            glUniformMatrix4fv(octreeProjLoc, 1, GL_FALSE, glm::value_ptr(projection));
        else
            std::cout << "Warning: uProjection uniform not found in octree shader" << std::endl;
            
        if (octreeViewLoc != -1)
            glUniformMatrix4fv(octreeViewLoc, 1, GL_FALSE, glm::value_ptr(view));
        else
            std::cout << "Warning: uView uniform not found in octree shader" << std::endl;
        
        // Debug: Imprimir número de vértices del octree
        std::cout << "Drawing octree with " << octreeVertexCount_ << " vertices" << std::endl;
        
        // Establecer grosor de línea más visible
        glLineWidth(2.0f);
        
        // Dibujar el octree
        glBindVertexArray(octreeVAO_);
        glDrawArrays(GL_LINES, 0, octreeVertexCount_);
        glBindVertexArray(0);
        glUseProgram(0);
    }
}

void OpenGLRenderer::updateBodies(Body *bodies, int numBodies)
{
    if (numBodies <= 0 || !bodies)
    {
        std::cerr << "Warning: No bodies to update or invalid data." << std::endl;
        return;
    }

    // Find coordinate center to help with scaling
    Vector centerOfMass(0, 0, 0);
    double totalMass = 0.0;

    double minX = std::numeric_limits<double>::max();
    double maxX = std::numeric_limits<double>::lowest();
    double minY = std::numeric_limits<double>::max();
    double maxY = std::numeric_limits<double>::lowest();
    double minZ = std::numeric_limits<double>::max();
    double maxZ = std::numeric_limits<double>::lowest();

    for (int i = 0; i < numBodies; ++i)
    {
        // Accumulate center of mass
        centerOfMass.x += bodies[i].position.x * bodies[i].mass;
        centerOfMass.y += bodies[i].position.y * bodies[i].mass;
        centerOfMass.z += bodies[i].position.z * bodies[i].mass;
        totalMass += bodies[i].mass;

        // Track coordinate ranges
        minX = std::min(minX, bodies[i].position.x);
        maxX = std::max(maxX, bodies[i].position.x);
        minY = std::min(minY, bodies[i].position.y);
        maxY = std::max(maxY, bodies[i].position.y);
        minZ = std::min(minZ, bodies[i].position.z);
        maxZ = std::max(maxZ, bodies[i].position.z);
    }

    // Normalize center of mass
    centerOfMass.x /= totalMass;
    centerOfMass.y /= totalMass;
    centerOfMass.z /= totalMass;

    // Prepare vector for combined position and mass data
    std::vector<float> combinedData;
    combinedData.reserve(numBodies * 4); // xyz + mass

    // Normalization factors
    double rangeX = maxX - minX;
    double rangeY = maxY - minY;
    double rangeZ = maxZ - minZ;
    double maxRange = std::max(std::max(rangeX, rangeY), rangeZ);

    for (int i = 0; i < numBodies; ++i)
    {
        // Normalize coordinates relative to center of mass
        float normalizedX = static_cast<float>((bodies[i].position.x - centerOfMass.x) / maxRange);
        float normalizedY = static_cast<float>((bodies[i].position.y - centerOfMass.y) / maxRange);
        float normalizedZ = static_cast<float>((bodies[i].position.z - centerOfMass.z) / maxRange);

        // Add normalized coordinates
        combinedData.push_back(normalizedX);
        combinedData.push_back(normalizedY);
        combinedData.push_back(normalizedZ);

        // Add mass value for shader
        float normalizedMass = static_cast<float>(bodies[i].mass);
        combinedData.push_back(normalizedMass);
    }

    numBodies_ = numBodies;

    // Bind and update buffer data
    glBindVertexArray(VAO_);
    glBindBuffer(GL_ARRAY_BUFFER, VBO_);

    // Reallocate buffer if size changed
    glBufferData(GL_ARRAY_BUFFER,
                 combinedData.size() * sizeof(float),
                 combinedData.data(), GL_DYNAMIC_DRAW);

    // Configure vertex attributes
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
                          4 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);

    // Configure mass attribute
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE,
                          4 * sizeof(float), (void *)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void OpenGLRenderer::addNodeToOctreeVisualization(
    std::vector<float> &vertices,
    const Node &node,
    float scaleFactor,
    int currentDepth,
    int maxDepth)
{
    if (currentDepth > maxDepth)
    {
        return;
    }

    // Extract bounding box corners and scale them
    float minX = node.topLeftFront.x * scaleFactor;
    float maxX = node.botRightBack.x * scaleFactor;
    float minY = node.botRightBack.y * scaleFactor; // Note: Y is inverted in your implementation
    float maxY = node.topLeftFront.y * scaleFactor;
    float minZ = node.topLeftFront.z * scaleFactor;
    float maxZ = node.botRightBack.z * scaleFactor;

    // Add front face edges
    vertices.push_back(minX);
    vertices.push_back(minY);
    vertices.push_back(minZ);
    vertices.push_back(maxX);
    vertices.push_back(minY);
    vertices.push_back(minZ);

    vertices.push_back(maxX);
    vertices.push_back(minY);
    vertices.push_back(minZ);
    vertices.push_back(maxX);
    vertices.push_back(maxY);
    vertices.push_back(minZ);

    vertices.push_back(maxX);
    vertices.push_back(maxY);
    vertices.push_back(minZ);
    vertices.push_back(minX);
    vertices.push_back(maxY);
    vertices.push_back(minZ);

    vertices.push_back(minX);
    vertices.push_back(maxY);
    vertices.push_back(minZ);
    vertices.push_back(minX);
    vertices.push_back(minY);
    vertices.push_back(minZ);

    // Add back face edges
    vertices.push_back(minX);
    vertices.push_back(minY);
    vertices.push_back(maxZ);
    vertices.push_back(maxX);
    vertices.push_back(minY);
    vertices.push_back(maxZ);

    vertices.push_back(maxX);
    vertices.push_back(minY);
    vertices.push_back(maxZ);
    vertices.push_back(maxX);
    vertices.push_back(maxY);
    vertices.push_back(maxZ);

    vertices.push_back(maxX);
    vertices.push_back(maxY);
    vertices.push_back(maxZ);
    vertices.push_back(minX);
    vertices.push_back(maxY);
    vertices.push_back(maxZ);

    vertices.push_back(minX);
    vertices.push_back(maxY);
    vertices.push_back(maxZ);
    vertices.push_back(minX);
    vertices.push_back(minY);
    vertices.push_back(maxZ);

    // Add connecting edges
    vertices.push_back(minX);
    vertices.push_back(minY);
    vertices.push_back(minZ);
    vertices.push_back(minX);
    vertices.push_back(minY);
    vertices.push_back(maxZ);

    vertices.push_back(maxX);
    vertices.push_back(minY);
    vertices.push_back(minZ);
    vertices.push_back(maxX);
    vertices.push_back(minY);
    vertices.push_back(maxZ);

    vertices.push_back(maxX);
    vertices.push_back(maxY);
    vertices.push_back(minZ);
    vertices.push_back(maxX);
    vertices.push_back(maxY);
    vertices.push_back(maxZ);

    vertices.push_back(minX);
    vertices.push_back(maxY);
    vertices.push_back(minZ);
    vertices.push_back(minX);
    vertices.push_back(maxY);
    vertices.push_back(maxZ);
}

void OpenGLRenderer::processOctreeNode(
    std::vector<float> &vertices,
    Node *nodes,
    int numNodes,
    int nodeIndex,
    float scaleFactor,
    int currentDepth,
    int maxDepth)
{
    if (nodeIndex < 0 || nodeIndex >= numNodes || currentDepth > maxDepth)
    {
        return;
    }

    // Add the current node to visualization
    addNodeToOctreeVisualization(vertices, nodes[nodeIndex], scaleFactor, currentDepth, maxDepth);

    // For leaf nodes we stop here
    if (nodes[nodeIndex].isLeaf || currentDepth >= maxDepth)
    {
        return;
    }

    // Otherwise, process children
    // Note: In the Barnes-Hut implementation, children are at indices (nodeIndex*8 + 1) through (nodeIndex*8 + 8)
    for (int i = 1; i <= 8; i++)
    {
        int childIndex = nodeIndex * 8 + i;
        if (childIndex < numNodes &&
            nodes[childIndex].start != -1 &&
            nodes[childIndex].end != -1)
        {
            processOctreeNode(vertices, nodes, numNodes, childIndex, scaleFactor, currentDepth + 1, maxDepth);
        }
    }
}

// Modifica esta parte en OpenGLRenderer::updateOctreeVisualization

void OpenGLRenderer::updateOctreeVisualization(Node* nodes, int numNodes, int rootIndex, int maxDepth)
{
    if (numNodes <= 0 || !nodes || rootIndex < 0 || rootIndex >= numNodes) {
        std::cout << "Invalid octree data for visualization" << std::endl;
        octreeVertexCount_ = 0;
        return;
    }
    
    std::vector<float> vertices;
    
    // Problema potencial: Factor de escala inadecuado
    // Probar diferentes valores de escalado para hacer visible el octree
    // float scaleFactor = 1.0e-11; // Valor original
    float scaleFactor = 1.0e-12; // Intentar un valor más pequeño
    
    // También imprimir información sobre el tamaño del nodo raíz para depuración
    Node rootNode = nodes[rootIndex];
    std::cout << "Root node dimensions: " 
              << "Width: " << fabs(rootNode.botRightBack.x - rootNode.topLeftFront.x) * scaleFactor
              << ", Height: " << fabs(rootNode.topLeftFront.y - rootNode.botRightBack.y) * scaleFactor
              << ", Depth: " << fabs(rootNode.botRightBack.z - rootNode.topLeftFront.z) * scaleFactor
              << std::endl;
    
    // Procesar el octree recursivamente comenzando desde la raíz
    processOctreeNode(vertices, nodes, numNodes, rootIndex, scaleFactor, 0, maxDepth);
    
    // Actualizar el conteo de vértices
    octreeVertexCount_ = vertices.size() / 3;
    
    if (octreeVertexCount_ == 0) {
        std::cout << "No octree nodes to visualize" << std::endl;
        return;
    }
    
    std::cout << "Generated octree visualization with " << octreeVertexCount_ << " vertices" << std::endl;
    
    // Actualizar VBO y VAO
    glBindVertexArray(octreeVAO_);
    glBindBuffer(GL_ARRAY_BUFFER, octreeVBO_);
    
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_DYNAMIC_DRAW);
    
    // Configurar atributos de vértices
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void OpenGLRenderer::renderOctree(float aspectRatio)
{
    if (octreeVertexCount_ <= 0 || octreeShaderProgram_ == 0)
    {
        return;
    }

    // Use the same projection and view as in the main render method
    // Create projection matrix
    glm::mat4 projection = glm::perspective(
        glm::radians(45.0f), // Field of view
        aspectRatio,         // Aspect ratio
        0.1f,                // Near plane
        100.0f               // Far plane
    );

    // Get camera parameters from simulation state
    float zoomFactor = simulationState_.zoomFactor.load();
    float offsetX = simulationState_.offsetX;
    float offsetY = simulationState_.offsetY;

    // Create view matrix with adjusted camera position
    glm::mat4 view = glm::lookAt(
        glm::vec3(offsetX, offsetY, 10.0f / zoomFactor), // Camera position
        glm::vec3(offsetX, offsetY, 0.0f),               // Look at offset origin
        glm::vec3(0.0f, 1.0f, 0.0f)                      // Up vector
    );

    // Use shader del octree
    glUseProgram(octreeShaderProgram_);

    // Configurar uniforms
    GLint projLoc = glGetUniformLocation(octreeShaderProgram_, "uProjection");
    GLint viewLoc = glGetUniformLocation(octreeShaderProgram_, "uView");
    GLint opacityLoc = glGetUniformLocation(octreeShaderProgram_, "uOpacity");

    glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));

    // Set opacity from simulation state
    if (opacityLoc != -1)
    {
        glUniform1f(opacityLoc, simulationState_.octreeOpacity);
    }

    // Enable blending for transparency
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Keep depth test enabled for proper ordering
    glEnable(GL_DEPTH_TEST);

    // Draw lines with appropriate width
    glLineWidth(1.0f);

    // Draw the octree
    glBindVertexArray(octreeVAO_);
    glDrawArrays(GL_LINES, 0, octreeVertexCount_);

    // Cleanup
    glBindVertexArray(0);
    glUseProgram(0);
}
