#include "opengl_renderer.h"
#include <iostream>

const char *vertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec3 aPosition;
    layout (location = 1) in float aMass;

    uniform mat4 uProjection;
    uniform mat4 uView;
    uniform float uPointSize;
    uniform float uScaleFactor;

    out float vMass;

    void main() {
        // Escalar las coordenadas
        vec3 scaledPos = aPosition * uScaleFactor;
        gl_Position = uProjection * uView * vec4(scaledPos, 1.0);
        
        // Ajustar tamaño del punto basado en la masa
        float pointScale = log(aMass + 1.0) * 0.5;
        gl_PointSize = uPointSize * (1.0 + pointScale);
        
        vMass = aMass;
    }
)";

// Fragment shader
const char *fragmentShaderSource = R"(
    #version 330 core
    in float vMass;
    out vec4 FragColor;

    void main() {
        // Gradiente de color basado en la masa
        float normalizedMass = clamp(log(vMass / 1e24) / 5.0, 0.0, 1.0);
        vec3 bodyColor = mix(
            vec3(0.2, 0.4, 1.0),   // Azul para cuerpos pequeños
            vec3(1.0, 0.6, 0.2),   // Naranja para cuerpos grandes
            normalizedMass
        );
        
        // Crear punto circular suave
        vec2 circCoord = 2.0 * gl_PointCoord - 1.0;
        float distSquared = dot(circCoord, circCoord);
        
        if (distSquared > 1.0) {
            discard;
        }
        
        // Gradiente de transparencia suave
        float alpha = 1.0 - smoothstep(0.7, 1.0, distSquared);
        
        FragColor = vec4(bodyColor, alpha);
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
    // Compilar shaders
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);

    // Crear programa de shader
    shaderProgram_ = glCreateProgram();
    glAttachShader(shaderProgram_, vertexShader);
    glAttachShader(shaderProgram_, fragmentShader);
    glLinkProgram(shaderProgram_);

    // Verificar errores de enlace
    GLint success;
    GLchar infoLog[512];
    glGetProgramiv(shaderProgram_, GL_LINK_STATUS, &success);
    if (!success)
    {
        glGetProgramInfoLog(shaderProgram_, sizeof(infoLog), nullptr, infoLog);
        std::cerr << "ERROR: Shader program linking failed\n"
                  << infoLog << std::endl;
    }

    // Limpiar shaders individuales
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

void OpenGLRenderer::setupBuffers()
{
    // Crear VAO y VBO
    glGenVertexArrays(1, &VAO_);
    glGenBuffers(1, &VBO_);

    glBindVertexArray(VAO_);
    glBindBuffer(GL_ARRAY_BUFFER, VBO_);

    // Alocar espacio para posiciones y masa
    glBufferData(GL_ARRAY_BUFFER,
                 bodyPositions_.size() * (3 * sizeof(float) + sizeof(float)),
                 nullptr, GL_DYNAMIC_DRAW);

    // Configurar atributos de posición
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
                          4 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);

    // Configurar atributos de masa
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE,
                          4 * sizeof(float), (void *)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void OpenGLRenderer::init()
{
    // Configuraciones básicas de OpenGL
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void OpenGLRenderer::updateBodies(Body *bodies, int numBodies)
{
    // Preparar vector para datos combinados de posición y masa
    std::vector<float> combinedData;
    combinedData.reserve(numBodies * 4);

    for (int i = 0; i < numBodies; ++i)
    {
        // Añadir coordenadas
        combinedData.push_back(bodies[i].position.x);
        combinedData.push_back(bodies[i].position.y);
        combinedData.push_back(bodies[i].position.z);

        // Añadir masa
        combinedData.push_back(bodies[i].mass);
    }

    numBodies_ = numBodies;

    // Si los buffers ya existen, actualizar
    if (VAO_ && VBO_)
    {
        glBindBuffer(GL_ARRAY_BUFFER, VBO_);
        glBufferSubData(GL_ARRAY_BUFFER, 0,
                        combinedData.size() * sizeof(float),
                        combinedData.data());
    }
    else
    {
        // Preparar buffers si aún no se han creado
        bodyPositions_.resize(numBodies);
        setupBuffers();

        glBindBuffer(GL_ARRAY_BUFFER, VBO_);
        glBufferSubData(GL_ARRAY_BUFFER, 0,
                        combinedData.size() * sizeof(float),
                        combinedData.data());
    }
}

void OpenGLRenderer::render(float aspectRatio)
{
    if (numBodies_ == 0)
        return;

    std::cout << "Renderizando " << numBodies_ << " cuerpos" << std::endl;
    std::cout << "Primera posición: ("
        << bodyPositions_[2].x << ", "
        << bodyPositions_[2].y << ", "
        << bodyPositions_[2].z << ")" << std::endl;

    // Limpiar buffers
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Usar programa de shader
    glUseProgram(shaderProgram_);

    // Crear matriz de proyección
    glm::mat4 projection = glm::perspective(
        glm::radians(45.0f), // Campo de visión
        aspectRatio,         // Relación de aspecto
        1.0f,                // Plano cercano
        1.0e16f              // Plano lejano extendido
    );

    // Crear matriz de vista
    float zoomFactor = simulationState_.zoomFactor.load();
    glm::mat4 view = glm::lookAt(
        glm::vec3(0.0f, 0.0f, 5.0e14f / zoomFactor), // Posición de la cámara
        glm::vec3(0.0f, 0.0f, 0.0f),                 // Mirar al origen
        glm::vec3(0.0f, 1.0f, 0.0f)                  // Vector hacia arriba
    );

    // Factor de escala para ajustar las coordenadas
    float scaleFactor = 1.0e-11f;

    // Configurar uniforms
    GLint projLoc = glGetUniformLocation(shaderProgram_, "uProjection");
    GLint viewLoc = glGetUniformLocation(shaderProgram_, "uView");
    GLint pointSizeLoc = glGetUniformLocation(shaderProgram_, "uPointSize");
    GLint scaleFactorLoc = glGetUniformLocation(shaderProgram_, "uScaleFactor");

    glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
    glUniform1f(pointSizeLoc, 10.0f);         // Tamaño base de punto
    glUniform1f(scaleFactorLoc, scaleFactor); // Factor de escala

    // Dibujar puntos
    glBindVertexArray(VAO_);
    glDrawArrays(GL_POINTS, 0, numBodies_);

    // Limpiar
    glBindVertexArray(0);
    glUseProgram(0);
}