#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <chrono>
#include "barnes_hut.cuh"
#include "constants.h"


cv::VideoWriter video("BH_CUDA.avi",
    cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
    30,
    cv::Size(WINDOW_WIDTH, WINDOW_HEIGHT));

cv::Point scaleToWindow(const Vector &pos3D)
{
    double scaleX = WINDOW_WIDTH / (NBODY_WIDTH * 2.0);
    double scaleY = WINDOW_HEIGHT / (NBODY_HEIGHT * 2.0);
    double screenX = (pos3D.x + NBODY_WIDTH) * scaleX;
    double screenY = (pos3D.y + NBODY_HEIGHT) * scaleY;
    return cv::Point((int)screenX, (int)(WINDOW_HEIGHT - screenY));
}

/**
 * Dibuja los cuerpos en una imagen y la escribe al video.
 */
void storeFrame(Body *bodies, int n, int id)
{
    cv::Mat image = cv::Mat::zeros(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC3);
    for (int i = 0; i < n; i++)
    {
        cv::Point center = scaleToWindow(bodies[i].position);
        cv::circle(image, center, 1, cv::Scalar(255, 255, 255), -1);
    }
    video.write(image);
    // Para guardar frames individuales como imagen:
    // cv::imwrite("frame3D_" + std::to_string(id) + ".jpg", image);
}

int main(int argc, char **argv)
{
    int nBodies = atoi(argv[1]);
    int sim = atoi(argv[2]);
    int iters = atoi(argv[3]);
    
    BarnesHutCuda *bh = new BarnesHutCuda(nBodies);
    bh->setup(sim);
    
    using Clock = std::chrono::steady_clock;

    // Imprimir el encabezado CSV (por ejemplo):
    std::cout << "iteration"
              << ",totalIterationMs"
              << std::endl;

    // Bucle de iteraciones
    for (int i = 0; i < iters; ++i)
    {
        // Marca de inicio de la iteración en CPU
        auto start = Clock::now();

        // Llamamos a update() (mide internamente los 4 kernels)
        UpdateTimes kernelTimes = bh->update();

        // Marca de fin de la iteración
        auto end = Clock::now();

        // Calcular el tiempo de esta iteración en milisegundos
        double iterationTime = std::chrono::duration<double, std::milli>(end - start).count();

        // Imprimir una línea CSV con todos los datos
        std::cout << i << ","      // Índice de la iteración
                  << iterationTime // ms totales (CPU + GPU)
                  << std::endl;
        
        bh->readDeviceBodies();
        storeFrame(bh->getBodies(), nBodies, i);
    }

    video.release();
    delete bh;

    return 0;
}
