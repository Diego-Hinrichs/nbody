#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <chrono>
#include "barnes_hut.cuh"

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

void storeFrame(Body *bodies, int n, int id)
{
    cv::Mat image = cv::Mat::zeros(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC3);
    for (int i = 0; i < n; i++)
    {
        cv::Point center = scaleToWindow(bodies[i].position);
        cv::circle(image, center, 1, cv::Scalar(255, 255, 255), -1);
    }
    video.write(image);
}

int main(int argc, char **argv)
{
    int nBodies = atoi(argv[1]);
    int iters = atoi(argv[2]);

    BarnesHutCuda *bh = new BarnesHutCuda(nBodies);
    bh->setup();

    using Clock = std::chrono::steady_clock;

    std::cout << "iteration" << ",totalIterationMs" << std::endl;

    for (int i = 0; i < iters; ++i)
    {
        auto start = Clock::now();

        bh->update();

        auto end = Clock::now();

        double iterationTime = std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << i << "," << iterationTime << std::endl;

        bh->readDeviceBodies();
        storeFrame(bh->getBodies(), nBodies, i);
    }

    video.release();
    delete bh;

    return 0;
}
