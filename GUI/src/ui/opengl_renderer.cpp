#include "../../include/ui/opengl_renderer.h"
#include <iostream>
#include <limits>
#include <cmath>

const char *vertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec3 aPosition;
    layout (location = 1) in float aMass;

    uniform mat4 uProjection;
    uniform mat4 uView;
    uniform float uPointSize;
    uniform float uScaleFactor;

    out float vMass;
    out float vDistance;

    void main() {
        // The position is already normalized
        vec4 viewPos = uView * vec4(aPosition, 1.0);
        
        // Calculate distance from camera (for falloff effects)
        vDistance = length(viewPos.xyz);
        
        // Calculate final position
        gl_Position = uProjection * viewPos;
        
        // Adjust point size based on normalized mass
        // Higher values will appear larger
        float massScale = max(aMass * 1.0, 0.01);
        
        // Apply distance attenuation - closer objects appear larger
        float distanceScale = 10.0 / (1.0 + vDistance * 0.1);
        
        // Combine all scaling factors
        gl_PointSize = uPointSize * massScale * clamp(distanceScale, 0.1, 5.0);
        
        // Pass mass to fragment shader (already scaled)
        vMass = aMass;
    }
)";

// Fragment shader
const char *fragmentShaderSource = R"(
    #version 330 core
    in float vMass;
    in float vDistance;
    out vec4 FragColor;

    void main() {
        // Determine body type based on normalized mass
        vec3 bodyColor;
        float glowIntensity;
        float coreBrightness;
        
        // Higher mass bodies get brighter colors
        if (vMass > 1.0) {
            // Bright yellow for most massive bodies
            bodyColor = vec3(1.0, 0.9, 0.7);
            glowIntensity = 0.8;
            coreBrightness = 1.0;
        }
        else if (vMass > 0.8) {
            // Orange for medium-mass bodies
            bodyColor = vec3(1.0, 0.5, 0.2);
            glowIntensity = 0.5;
            coreBrightness = 0.8;
        }
        else {
            // Blue-gray for smaller bodies
            bodyColor = vec3(0.4, 0.6, 0.8);
            glowIntensity = 0.3;
            coreBrightness = 0.6;
        }
        
        // Create circular point with glow effect
        vec2 circCoord = 2.0 * gl_PointCoord - 1.0;
        float distSquared = dot(circCoord, circCoord);
        
        if (distSquared > 1.0) {
            discard;
        }
        
        // Core and glow effect
        float coreFactor = 1.0 - smoothstep(0.0, 0.4, distSquared);
        float glowFactor = 1.0 - smoothstep(0.4, 1.0, distSquared);
        
        // Combine core and glow
        vec3 finalColor = mix(
            bodyColor * glowIntensity,
            bodyColor * coreBrightness,
            coreFactor
        );
        
        // Adjust alpha based on distance from center
        float alpha = mix(glowIntensity, 1.0, coreFactor);
        
        // Distance fade effect
        alpha *= clamp(1.0 - (vDistance * 0.1), 0.1, 1.0);
        
        FragColor = vec4(finalColor, alpha);
    }
)";

OpenGLRenderer::OpenGLRenderer(SimulationState &simulationState)
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
    if (VAO_)
        glDeleteVertexArrays(1, &VAO_);
    if (VBO_)
        glDeleteBuffers(1, &VBO_);
    if (shaderProgram_)
        glDeleteProgram(shaderProgram_);
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
    // Create VAO and VBO
    glGenVertexArrays(1, &VAO_);
    glGenBuffers(1, &VBO_);
}

void OpenGLRenderer::init()
{
    // Extensive OpenGL context and capability checking
    std::cout << "OpenGL Initialization Details:" << std::endl;

    // Print OpenGL context information
    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "OpenGL Renderer: " << glGetString(GL_RENDERER) << std::endl;
    std::cout << "OpenGL Vendor: " << glGetString(GL_VENDOR) << std::endl;
    std::cout << "GLSL Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;

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

        // Point rendering optimizations
        // glEnable(GL_POINT_SMOOTH);
        // glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
        // err = glGetError();
        // if (err != GL_NO_ERROR)
        //     std::cerr << "Error configuring point rendering: " << err << std::endl;

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
        simulationState_.zoomFactor.store(1.0);
        simulationState_.offsetX = 0.0;
        simulationState_.offsetY = 0.0;

        std::cout << "OpenGL Renderer initialized successfully" << std::endl;
        std::cout << "Shader program ID: " << shaderProgram_ << std::endl;
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
    if (numBodies_ == 0 || shaderProgram_ == 0)
    {
        std::cerr << "Rendering conditions not met: "
                  << "numBodies=" << numBodies_
                  << ", shaderProgram=" << shaderProgram_ << std::endl;
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
        glm::vec3(offsetX, offsetY, 2.0f / zoomFactor), // Camera position
        glm::vec3(offsetX, offsetY, 0.0f),              // Look at offset origin
        glm::vec3(0.0f, 1.0f, 0.0f)                     // Up vector
    );

    // Scale factor for adjusting normalized coordinates
    float scaleFactor = 1.0f * zoomFactor;
    
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
    glUniform1f(pointSizeLoc, 5.0f + (zoomFactor * 0.5f));
    glUniform1f(scaleFactorLoc, scaleFactor);

    // Clear buffers with a nice dark background
    glClearColor(0.0f, 0.0f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Bind VAO and draw points
    glBindVertexArray(VAO_);
    glDrawArrays(GL_POINTS, 0, numBodies_);

    // Cleanup
    glBindVertexArray(0);
    glUseProgram(0);
}

void OpenGLRenderer::updateBodies(Body *bodies, int numBodies)
{   
    if (numBodies <= 0 || !bodies)
    {
        std::cerr << "Warning: No bodies to update or invalid data." << std::endl;
        return;
    }

    std::cout << "Updating " << numBodies << " bodies" << std::endl;

    float massScaleFactor = simulationState_.massScaleFactor.load();

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

        // Print first 10 bodies in detail
        if (i < 10)
        {
            std::cout << "Body " << i
                      << ": x=" << bodies[i].position.x
                      << ", y=" << bodies[i].position.y
                      << ", z=" << bodies[i].position.z
                      << ", mass=" << bodies[i].mass << std::endl;
        }
    }

    // Normalize center of mass
    centerOfMass.x /= totalMass;
    centerOfMass.y /= totalMass;
    centerOfMass.z /= totalMass;

    std::cout << "Center of Mass: "
              << "x=" << centerOfMass.x
              << ", y=" << centerOfMass.y
              << ", z=" << centerOfMass.z << std::endl;

    std::cout << "Coordinate Ranges:" << std::endl;
    std::cout << "X: " << minX << " to " << maxX << std::endl;
    std::cout << "Y: " << minY << " to " << maxY << std::endl;
    std::cout << "Z: " << minZ << " to " << maxZ << std::endl;

    // Prepare vector for combined position and mass data
    std::vector<float> combinedData;
    combinedData.reserve(numBodies * 4); // xyz + mass

    // Normalization factors
    double rangeX = maxX - minX;
    double rangeY = maxY - minY;
    double rangeZ = maxZ - minZ;
    double maxRange = std::max({rangeX, rangeY, rangeZ});

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

        // Normalize mass logarithmically for visibility
        float normalizedMass = static_cast<float>(log(std::max(bodies[i].mass, 1.0)) * massScaleFactor);
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

    // Debug print for rendering
    std::cout << "Buffer updated with " << numBodies << " bodies" << std::endl;
}
