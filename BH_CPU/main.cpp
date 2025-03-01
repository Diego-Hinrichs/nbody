/******************************************************
 * main.cpp
 * Barnes-Hut 3D + Guardar Video con OpenCV (2D proyección)
 ******************************************************/

#include <iostream>
#include <memory>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

//----------- OpenCV -----------
#include <opencv2/opencv.hpp>

//==================================================
// 1) Definiciones de Constantes (antes constants.h)
//==================================================
#define NUM_BODIES 10000
#define WINDOW_WIDTH 1600
#define WINDOW_HEIGHT 1600
#define NBODY_WIDTH 1.0e12 // ancho "físico" a simular (ej: 1e12 ~ 1e3 Gm)
#define NBODY_HEIGHT 1.0e12
#define CENTERX 0
#define CENTERY 0
#define GRAVITY 6.67e-11
#define COLLISION_TH 1.0e10
#define MIN_DIST 2.0e10
#define MAX_DIST 5.0e11
#define SUN_MASS 1.9890e30
#define SUN_DIA 1.3927e6
#define EARTH_MASS 5.974e24
#define EARTH_DIA 12756

// Umbral para distinguir "estrellas" de "planetas" al dibujar (ejemplo)
#define HBL 1.0e29

//==================================================
// 2) Clase Vector3
//==================================================
class Vector3
{
public:
    double x, y, z;

    Vector3() : x(0.0), y(0.0), z(0.0) {}
    Vector3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
    Vector3(const Vector3 &v) : x(v.x), y(v.y), z(v.z) {}
    Vector3(Vector3 &&v) noexcept : x(v.x), y(v.y), z(v.z) {}

    double mod() const
    {
        return std::sqrt(x * x + y * y + z * z);
    }

    Vector3 operator+(const Vector3 &rhs) const
    {
        return Vector3(x + rhs.x, y + rhs.y, z + rhs.z);
    }

    Vector3 operator-(const Vector3 &rhs) const
    {
        return Vector3(x - rhs.x, y - rhs.y, z - rhs.z);
    }

    Vector3 operator*(double v) const
    {
        return Vector3(x * v, y * v, z * v);
    }

    Vector3 operator/(double v) const
    {
        return Vector3(x / v, y / v, z / v);
    }

    Vector3 &operator+=(const Vector3 &rhs)
    {
        x += rhs.x;
        y += rhs.y;
        z += rhs.z;
        return *this;
    }

    Vector3 &operator=(const Vector3 &rhs)
    {
        x = rhs.x;
        y = rhs.y;
        z = rhs.z;
        return *this;
    }

    bool operator==(const Vector3 &rhs) const
    {
        return (x == rhs.x && y == rhs.y && z == rhs.z);
    }

    friend std::ostream &operator<<(std::ostream &os, const Vector3 &v)
    {
        os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
        return os;
    }
};

//==================================================
// 3) Clase Body3D
//==================================================
class Body3D
{
public:
    bool isDynamic;
    double mass;
    double radius;
    Vector3 position;
    Vector3 velocity;
    Vector3 acceleration;

    Body3D(double m, double r, const Vector3 &p,
           const Vector3 &v, const Vector3 &a, bool d = true)
        : isDynamic(d), mass(m), radius(r),
          position(p), velocity(v), acceleration(a) {}

    friend std::ostream &operator<<(std::ostream &os, const Body3D &b)
    {
        os << "[Body3D] m=" << b.mass
           << ", r=" << b.radius
           << ", pos=" << b.position
           << ", vel=" << b.velocity
           << ", acc=" << b.acceleration
           << ", dyn=" << b.isDynamic;
        return os;
    }
};

//==================================================
// 4) Clase Octree (barnes-hut en 3D)
//==================================================
class Octree
{
public:
    Vector3 minCorner;
    Vector3 maxCorner;

    Vector3 centerMass;
    double totalMass;

    bool isLeaf;
    std::shared_ptr<Body3D> b;

    std::unique_ptr<Octree> children[8];

    Octree(const Vector3 &minC, const Vector3 &maxC)
        : minCorner(minC), maxCorner(maxC),
          centerMass(0.0, 0.0, 0.0), totalMass(0.0),
          isLeaf(true), b(nullptr)
    {
        for (int i = 0; i < 8; i++)
            children[i] = nullptr;
    }

    void insert(std::shared_ptr<Body3D> body);
    int getOctant(const Vector3 &pos) const;
    bool inBoundary(const Vector3 &pos) const;
    double getMaxSize() const;
};

//--------------------------------------------------
// Métodos de Octree
//--------------------------------------------------
int Octree::getOctant(const Vector3 &pos) const
{
    double midx = 0.5 * (minCorner.x + maxCorner.x);
    double midy = 0.5 * (minCorner.y + maxCorner.y);
    double midz = 0.5 * (minCorner.z + maxCorner.z);

    int octIndex = 0;

    if (pos.x < midx)
    {
        if (pos.y < midy)
        {
            if (pos.z < midz)
                octIndex = 0; // 000
            else
                octIndex = 1; // 001
        }
        else
        {
            if (pos.z < midz)
                octIndex = 2; // 010
            else
                octIndex = 3; // 011
        }
    }
    else
    {
        if (pos.y < midy)
        {
            if (pos.z < midz)
                octIndex = 4; // 100
            else
                octIndex = 5; // 101
        }
        else
        {
            if (pos.z < midz)
                octIndex = 6; // 110
            else
                octIndex = 7; // 111
        }
    }
    return octIndex;
}

bool Octree::inBoundary(const Vector3 &pos) const
{
    return (pos.x >= minCorner.x && pos.x <= maxCorner.x &&
            pos.y >= minCorner.y && pos.y <= maxCorner.y &&
            pos.z >= minCorner.z && pos.z <= maxCorner.z);
}

double Octree::getMaxSize() const
{
    double dx = maxCorner.x - minCorner.x;
    double dy = maxCorner.y - minCorner.y;
    double dz = maxCorner.z - minCorner.z;
    return std::max(std::max(dx, dy), dz);
}

void Octree::insert(std::shared_ptr<Body3D> body)
{
    if (!body)
        return;
    if (!inBoundary(body->position))
        return; // fuera del volumen

    if (isLeaf && b == nullptr)
    {
        // nodo hoja vacío
        b = body;
        return;
    }

    if (isLeaf && b != nullptr)
    {
        // subdividir
        isLeaf = false;

        double midx = 0.5 * (minCorner.x + maxCorner.x);
        double midy = 0.5 * (minCorner.y + maxCorner.y);
        double midz = 0.5 * (minCorner.z + maxCorner.z);

        // Crear los 8 hijos
        children[0] = std::make_unique<Octree>(
            Vector3(minCorner.x, minCorner.y, minCorner.z),
            Vector3(midx, midy, midz));
        children[1] = std::make_unique<Octree>(
            Vector3(minCorner.x, minCorner.y, midz),
            Vector3(midx, midy, maxCorner.z));
        children[2] = std::make_unique<Octree>(
            Vector3(minCorner.x, midy, minCorner.z),
            Vector3(midx, maxCorner.y, midz));
        children[3] = std::make_unique<Octree>(
            Vector3(minCorner.x, midy, midz),
            Vector3(midx, maxCorner.y, maxCorner.z));
        children[4] = std::make_unique<Octree>(
            Vector3(midx, minCorner.y, minCorner.z),
            Vector3(maxCorner.x, midy, midz));
        children[5] = std::make_unique<Octree>(
            Vector3(midx, minCorner.y, midz),
            Vector3(maxCorner.x, midy, maxCorner.z));
        children[6] = std::make_unique<Octree>(
            Vector3(midx, midy, minCorner.z),
            Vector3(maxCorner.x, maxCorner.y, midz));
        children[7] = std::make_unique<Octree>(
            Vector3(midx, midy, midz),
            Vector3(maxCorner.x, maxCorner.y, maxCorner.z));

        // reinsertar el body que ya estaba
        int oldOct = getOctant(b->position);
        children[oldOct]->insert(b);
        b = nullptr;

        // insertar el nuevo
        int newOct = getOctant(body->position);
        children[newOct]->insert(body);
    }
    else
    {
        // no hoja -> insertar en el hijo apropiado
        int oct = getOctant(body->position);
        children[oct]->insert(body);
    }
}

//==================================================
// 5) Funciones helper para actualizar COM
//==================================================
double getTotalMass(const std::unique_ptr<Octree> &node)
{
    if (!node)
        return 0.0;
    return node->totalMass;
}

void updateCenterMass(std::unique_ptr<Octree> &node)
{
    if (!node)
        return;

    // Hoja con body
    if (node->isLeaf && node->b)
    {
        node->totalMass = node->b->mass;
        node->centerMass = node->b->position;
        return;
    }

    double totalChildMass = 0.0;
    Vector3 weightedPos(0.0, 0.0, 0.0);

    for (int i = 0; i < 8; i++)
    {
        if (node->children[i])
        {
            updateCenterMass(node->children[i]);
            double cmass = getTotalMass(node->children[i]);
            Vector3 cpos = node->children[i]->centerMass;
            totalChildMass += cmass;
            weightedPos += cpos * cmass;
        }
    }

    node->totalMass = totalChildMass;
    if (totalChildMass > 0.0)
        node->centerMass = weightedPos / totalChildMass;
    else
        node->centerMass = Vector3(0, 0, 0);
}

//==================================================
// 6) Variables Globales y funciones para la simulación
//==================================================
std::vector<std::shared_ptr<Body3D>> bodies;

// Banderas/param
static const double THETA = 0.5; // criterio de apertura BH
static const double EPS = 1e10;  // suavizado
static const double DT = 3600;   // paso de tiempo (1 hora aprox)

void computeForceBarnesHut(const std::unique_ptr<Octree> &node, std::shared_ptr<Body3D> body)
{
    if (!node || node->totalMass <= 0.0)
        return;

    // nodo hoja con body != body actual => fuerza exacta
    if (node->isLeaf && node->b && node->b != body)
    {
        Vector3 dir = node->b->position - body->position;
        double dist = dir.mod() + EPS;
        double F = (GRAVITY * body->mass * node->b->mass) / (dist * dist * dist);
        body->acceleration += dir * (F / body->mass);
        return;
    }

    // criterio BH: s/d < THETA
    double s = node->getMaxSize();
    Vector3 dir = node->centerMass - body->position;
    double d = dir.mod() + EPS;

    if ((s / d) < THETA)
    {
        // toda la masa del nodo en centerMass
        double F = (GRAVITY * body->mass * node->totalMass) / (d * d * d);
        body->acceleration += dir * (F / body->mass);
    }
    else
    {
        // visitar hijos
        for (int i = 0; i < 8; i++)
        {
            if (node->children[i])
                computeForceBarnesHut(node->children[i], body);
        }
    }
}

void integrateEuler(std::shared_ptr<Body3D> body)
{
    if (!body->isDynamic)
        return;

    body->velocity += (body->acceleration * DT);
    body->position += (body->velocity * DT);
    body->acceleration = Vector3(0, 0, 0);
}

//==================================================
// 7) OpenCV: video y funciones para grabar la simulación
//    - Proyectaremos (x,y,z) -> (x,y), ignorando z
//    - Redimensionamos a la ventana (WINDOW_WIDTH x WINDOW_HEIGHT)
//==================================================

// Creamos el writer de video (30 fps, MJPG)
cv::VideoWriter video(
    "bh_cpu.avi",
    cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
    30,
    cv::Size(WINDOW_WIDTH, WINDOW_HEIGHT));

// Proyección y escalado 3D->2D. Ignoramos z,
// centrando en (0,0) y escalamos al [WINDOW_WIDTH, WINDOW_HEIGHT].
cv::Point scaleToWindow(const Vector3 &pos3D)
{
    // Ignorar z. Sólo usamos x,y.
    double x = pos3D.x;
    double y = pos3D.y;

    // Escalas
    double scaleX = (double)WINDOW_WIDTH / (double)(NBODY_WIDTH * 2.0);
    double scaleY = (double)WINDOW_HEIGHT / (double)(NBODY_HEIGHT * 2.0);

    // Trasladar (0,0) -> centro de la imagen
    // El "0" en x,y del mundo va al centro de la ventana
    // Suponemos que el espacio va de [-NBODY_WIDTH,+NBODY_WIDTH], etc.
    double screenX = (x + NBODY_WIDTH) * scaleX;
    double screenY = (y + NBODY_HEIGHT) * scaleY;

    return cv::Point((int)screenX, (int)(WINDOW_HEIGHT - screenY));
    // Se invierte Y para que +Y vaya hacia arriba, si se desea.
}

// Guarda un frame con los cuerpos dibujados
void storeFrame()
{
    // Lienzo negro
    cv::Mat image = cv::Mat::zeros(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC3);

    // Dibujar cada cuerpo
    for (auto &b : bodies)
    {
        cv::Point center = scaleToWindow(b->position);

        // Determinar color y radio
        cv::Scalar color(255, 255, 255); // blanco por defecto
        int radius = 2;

        // Si masa >= HBL, lo consideramos "estrella" => rojo, más grande
        if (b->mass >= HBL)
        {
            color = cv::Scalar(0, 0, 255); // BGR => rojo
            radius = 5;
        }

        cv::circle(image, center, radius, color, -1);
    }

    // Escribimos el frame en el video
    video.write(image);
}

//==================================================
// 8) Inicializar cuerpos (random esférico 3D)
//==================================================
void initRandomBodies3D()
{
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    bodies.clear();
    bodies.reserve(NUM_BODIES);

    double maxDistance = MAX_DIST;
    double minDistance = MIN_DIST;

    // Centro en (CENTERX, CENTERY, 0)
    Vector3 centerPos(CENTERX, CENTERY, 0.0);

    // Generar (NUM_BODIES - 1) cuerpos
    for (int i = 0; i < NUM_BODIES; i++)
    {
        double rnd01 = (double)std::rand() / (double)RAND_MAX;
        double r = minDistance + rnd01 * (maxDistance - minDistance);

        double rnd02 = (double)std::rand() / (double)RAND_MAX;
        double rnd03 = (double)std::rand() / (double)RAND_MAX;

        double theta = 2.0 * M_PI * rnd02;
        // Distribución uniforme en esfera:
        double phi = std::acos(2.0 * rnd03 - 1.0);

        double sinPhi = std::sin(phi);

        double x = r * sinPhi * std::cos(theta);
        double y = r * sinPhi * std::sin(theta);
        double z = r * std::cos(phi);

        Vector3 position = centerPos + Vector3(x, y, z);

        // "Tierra" random
        auto body = std::make_shared<Body3D>(
            SUN_MASS,
            SUN_DIA,
            position,
            Vector3(0, 0, 0),
            Vector3(0, 0, 0),
            true);
        bodies.push_back(body);
    }
}

//==================================================
// 9) MAIN
//==================================================
int main()
{
    // Inicializar
    initRandomBodies3D();

    // Cantidad de pasos de la simulación
    int NUM_STEPS = 1000; // por ejemplo

    for (int step = 0; step < NUM_STEPS; step++)
    {
        // 1) Crear Octree que cubra [-NBODY_WIDTH, +NBODY_WIDTH]^3
        auto root = std::make_unique<Octree>(
            Vector3(-NBODY_WIDTH, -NBODY_WIDTH, -NBODY_WIDTH),
            Vector3(NBODY_WIDTH, NBODY_WIDTH, NBODY_WIDTH));

        // 2) Insertar cuerpos
        for (auto &b : bodies)
            root->insert(b);

        // 3) Update center of mass
        updateCenterMass(root);

        // 4) Computar fuerza en cada cuerpo
        for (auto &b : bodies)
            computeForceBarnesHut(root, b);

        // 5) Integrar
        for (auto &b : bodies)
            integrateEuler(b);

        // 6) Guardar frame en el video
        storeFrame();

        // (Opcional) imprimir algo por consola
        if (step % 100 == 0)
        {
            std::cout << "Step " << step << " / " << NUM_STEPS << std::endl;
        }
    }

    // Cerrar el video
    video.release();

    std::cout << "Simulación finalizada. Video guardado en 'nbody3D.avi'\n";
    return 0;
}
