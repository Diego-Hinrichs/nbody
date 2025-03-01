#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <omp.h>
#include <opencv2/opencv.hpp>

//--------------------------------------------------
// Constantes (ajusta según necesites)
//--------------------------------------------------
#define MAX_DIST      5.0e11
#define MIN_DIST      2.0e10
#define CENTERX       0
#define CENTERY       0
#define GRAVITY       6.67e-11
#define E             0.5
#define DT            25000
#define SUN_MASS      1.9890e30
#define SUN_DIA       1.3927e6
#define COLLISION_TH  1.0e10

// Constantes para visualización (ventana y escala)
#define WINDOW_WIDTH  1600
#define WINDOW_HEIGHT 1600
#define NBODY_WIDTH   1.0e12
#define NBODY_HEIGHT  1.0e12
#define HBL           1.0e29

//--------------------------------------------------
// Estructuras
//--------------------------------------------------
struct Vector3 {
    double x, y, z;
};

struct Body {
    double mass;
    double radius;
    bool isDynamic;
    Vector3 position;
    Vector3 velocity;
    Vector3 acceleration;
};

//--------------------------------------------------
// Variables globales
//--------------------------------------------------
std::vector<Body> bodies;

// VideoWriter global para guardar el video
cv::VideoWriter video("simulation.avi",
                      cv::VideoWriter::fourcc('M','J','P','G'),
                      30, cv::Size(WINDOW_WIDTH, WINDOW_HEIGHT));

//--------------------------------------------------
// Funciones auxiliares
//--------------------------------------------------

// Convierte coordenadas 3D a 2D (se ignora z)
cv::Point scaleToWindow(const Vector3 &pos) {
    double scaleX = static_cast<double>(WINDOW_WIDTH) / (NBODY_WIDTH * 2.0);
    double scaleY = static_cast<double>(WINDOW_HEIGHT) / (NBODY_HEIGHT * 2.0);
    double screenX = (pos.x + NBODY_WIDTH) * scaleX;
    double screenY = (pos.y + NBODY_HEIGHT) * scaleY;
    return cv::Point(static_cast<int>(screenX), static_cast<int>(WINDOW_HEIGHT - screenY));
}

// Dibuja todos los cuerpos sobre una imagen negra y la guarda en el video
void storeFrame() {
    cv::Mat image = cv::Mat::zeros(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC3);
    for (size_t i = 0; i < bodies.size(); i++) {
        cv::Point center = scaleToWindow(bodies[i].position);
        cv::Scalar color(255, 255, 255); // blanco por defecto
        int radius = 2;
        if (bodies[i].mass >= HBL) { // por ejemplo, considerar "estrella"
            color = cv::Scalar(0, 0, 255); // rojo
            radius = 5;
        }
        cv::circle(image, center, radius, color, -1);
    }
    video.write(image);
}

// Calcula la distancia Euclidiana entre dos puntos en 3D
double getDistance(const Vector3 &pos1, const Vector3 &pos2) {
    double dx = pos1.x - pos2.x;
    double dy = pos1.y - pos2.y;
    double dz = pos1.z - pos2.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// Determina si dos cuerpos colisionan (según un umbral)
bool isCollide(Body &b1, Body &b2) {
    double dist = getDistance(b1.position, b2.position);
    return (b1.radius + b2.radius + COLLISION_TH) > dist;
}

// Inicializa los cuerpos de forma aleatoria en un volumen esférico
void initRandomBodies(int n) {
    bodies.clear();
    bodies.resize(n);
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    double maxDistance = MAX_DIST;
    double minDistance = MIN_DIST;
    Vector3 centerPos = {CENTERX, CENTERY, 0.0};

    for (int i = 0; i < n; i++) {
        double r = ((maxDistance - minDistance) * (std::rand() / (double)RAND_MAX)) + minDistance;
        double theta = 2.0 * M_PI * (std::rand() / (double)RAND_MAX);
        double phi = M_PI * (std::rand() / (double)RAND_MAX);
        double x = centerPos.x + r * std::sin(phi) * std::cos(theta);
        double y = centerPos.y + r * std::sin(phi) * std::sin(theta);
        double z = centerPos.z + r * std::cos(phi);

        bodies[i].isDynamic = true;
        bodies[i].mass = SUN_MASS;
        bodies[i].radius = SUN_DIA;
        bodies[i].position = {x, y, z};
        bodies[i].velocity = {0.0, 0.0, 0.0};
        bodies[i].acceleration = {0.0, 0.0, 0.0};
    }
}

//--------------------------------------------------
// Versión 1: Paralelismo simple (1 nivel)
//--------------------------------------------------
void directSum(Body *bodies_ptr, int n) {
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        Body &bi = bodies_ptr[i];
        double fx = 0.0, fy = 0.0, fz = 0.0;
        bi.acceleration = {0.0, 0.0, 0.0};

        for (int j = 0; j < n; j++) {
            if (j == i)
                continue;
            Body &bj = bodies_ptr[j];
            if (!isCollide(bi, bj) && bi.isDynamic) {
                double rx = bj.position.x - bi.position.x;
                double ry = bj.position.y - bi.position.y;
                double rz = bj.position.z - bi.position.z;
                double r = std::sqrt(rx * rx + ry * ry + rz * rz + (E * E));
                double f = (GRAVITY * bi.mass * bj.mass) / (r * r * r + (E * E));
                fx += (f * rx) / bi.mass;
                fy += (f * ry) / bi.mass;
                fz += (f * rz) / bi.mass;
            }
        }
        bi.acceleration.x = fx;
        bi.acceleration.y = fy;
        bi.acceleration.z = fz;
    }

#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        Body &bi = bodies_ptr[i];
        if (!bi.isDynamic)
            continue;
        bi.velocity.x += bi.acceleration.x * DT;
        bi.velocity.y += bi.acceleration.y * DT;
        bi.velocity.z += bi.acceleration.z * DT;
        bi.position.x += bi.velocity.x * DT;
        bi.position.y += bi.velocity.y * DT;
        bi.position.z += bi.velocity.z * DT;
    }
}

//--------------------------------------------------
// Versión 2: Paralelismo anidado
//--------------------------------------------------
void directSumNested(Body *bodies_ptr, int n) {
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        Body &bi = bodies_ptr[i];
        double fx = 0.0, fy = 0.0, fz = 0.0;
        bi.acceleration = {0.0, 0.0, 0.0};

#pragma omp parallel for reduction(+ : fx, fy, fz)
        for (int j = 0; j < n; j++) {
            if (j == i)
                continue;
            Body &bj = bodies_ptr[j];
            if (!isCollide(bi, bj) && bi.isDynamic) {
                double rx = bj.position.x - bi.position.x;
                double ry = bj.position.y - bi.position.y;
                double rz = bj.position.z - bi.position.z;
                double r = std::sqrt(rx * rx + ry * ry + rz * rz + (E * E));
                double f = (GRAVITY * bi.mass * bj.mass) / (r * r * r + (E * E));
                fx += (f * rx) / bi.mass;
                fy += (f * ry) / bi.mass;
                fz += (f * rz) / bi.mass;
            }
        }
        bi.acceleration.x = fx;
        bi.acceleration.y = fy;
        bi.acceleration.z = fz;
    }

#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        Body &bi = bodies_ptr[i];
        if (!bi.isDynamic)
            continue;
        bi.velocity.x += bi.acceleration.x * DT;
        bi.velocity.y += bi.acceleration.y * DT;
        bi.velocity.z += bi.acceleration.z * DT;
        bi.position.x += bi.velocity.x * DT;
        bi.position.y += bi.velocity.y * DT;
        bi.position.z += bi.velocity.z * DT;
    }
}

//--------------------------------------------------
// MAIN
//--------------------------------------------------
int main(int argc, char *argv[]) {
    if (argc < 5) {
        std::cerr << "Uso: " << argv[0] << " <nBodies> <iters> <mode> <nt>\n";
        std::cerr << "  mode=0 -> Paralelismo simple\n";
        std::cerr << "  mode=1 -> Paralelismo anidado\n";
        std::cerr << "  nt = número de threads\n";
        return -1;
    }

    int nBodies = std::atoi(argv[1]);
    int iters = std::atoi(argv[2]);
    int mode = std::atoi(argv[3]);  // 0: simple; 1: nested
    int nt = std::atoi(argv[4]);

    omp_set_num_threads(nt);
    if (mode == 1) {
        omp_set_nested(1);
        omp_set_max_active_levels(2);
    } else {
        omp_set_nested(0);
    }

    initRandomBodies(nBodies);

    using Clock = std::chrono::steady_clock;
    double sumTimeMs = 0.0;
    std::cout << "iteration,totalIterationMs" << std::endl;

    for (int i = 0; i < iters; i++) {
        auto start = Clock::now();

        if (mode == 0)
            directSum(bodies.data(), nBodies);
        else
            directSumNested(bodies.data(), nBodies);

        auto end = Clock::now();
        double iterMs = std::chrono::duration<double, std::milli>(end - start).count();
        sumTimeMs += iterMs;
        std::cout << i << "," << iterMs << std::endl;

        // Guarda el frame de la simulación
        storeFrame();
    }

    std::cout << "Mean iteration time: " << sumTimeMs / iters << " ms" << std::endl;
    video.release();
    return 0;
}
