#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "barnes_hut.cuh"
#include "sfc_morton.cuh"

// Convert 3D position to 2D screen coordinates
cv::Point scaleToWindow(const Vector &pos3D)
{
    double scaleX = WINDOW_WIDTH / (NBODY_WIDTH * 2.0);
    double scaleY = WINDOW_HEIGHT / (NBODY_HEIGHT * 2.0);
    double screenX = (pos3D.x + NBODY_WIDTH) * scaleX;
    double screenY = (pos3D.y + NBODY_HEIGHT) * scaleY;
    return cv::Point((int)screenX, (int)(WINDOW_HEIGHT - screenY));
}

// Store frame in video
void storeFrame(Body *bodies, int n, cv::VideoWriter &video)
{
    cv::Mat image = cv::Mat::zeros(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC3);

    // Draw central body (sun) larger and brighter
    if (n > 0)
    {
        cv::Point center = scaleToWindow(bodies[0].position);
        cv::circle(image, center, 10, cv::Scalar(0, 150, 255), -1); // Orange for sun
    }

    // Draw other bodies
    for (int i = 1; i < n; i++)
    {
        cv::Point center = scaleToWindow(bodies[i].position);
        cv::circle(image, center, 2, cv::Scalar(255, 255, 255), -1); // White for planets
    }

    video.write(image);
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <nBodies> <iterations> [useSFC]" << std::endl;
        std::cerr << "  useSFC: 0 = Disabled, 1 = Enabled (default)" << std::endl;
        return 1;
    }

    int nBodies = atoi(argv[1]);
    int iters = atoi(argv[2]);

    // Default to SFC enabled unless specified otherwise
    bool useSFC = true;
    if (argc > 3)
    {
        useSFC = (atoi(argv[3]) != 0);
    }

    // Create descriptive filenames
    std::string simType = useSFC ? "morton_sfc" : "standard";
    std::string videoName = "simulation_" + simType + "_" + std::to_string(nBodies) + ".avi";
    std::string csvName = "timing_" + simType + "_" + std::to_string(nBodies) + ".csv";

    // Initialize video writer
    cv::VideoWriter video(videoName,
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          30,
                          cv::Size(WINDOW_WIDTH, WINDOW_HEIGHT));

    // Initialize CSV for timing data
    std::ofstream csv(csvName);
    csv << "iteration,timeMs" << std::endl;

    // Initialize simulation
    std::cout << "Running " << (useSFC ? "Morton SFC" : "standard") << " simulation with "
              << nBodies << " bodies for " << iters << " iterations" << std::endl;

    // Create appropriate simulation type
    BarnesHutCuda *simulation;
    if (useSFC)
    {
        simulation = new SFCBarnesHutCuda(nBodies, true);
    }
    else
    {
        simulation = new BarnesHutCuda(nBodies);
    }

    // Setup initial conditions
    simulation->setup();

    // Record initial state
    simulation->readDeviceBodies();
    storeFrame(simulation->getBodies(), nBodies, video);

    // Run simulation
    using Clock = std::chrono::steady_clock;

    for (int i = 0; i < iters; ++i)
    {
        auto start = Clock::now();

        simulation->update();

        auto end = Clock::now();
        double iterationTime = std::chrono::duration<double, std::milli>(end - start).count();

        // Record timing data
        csv << i << "," << iterationTime << std::endl;
        std::cout << "Iteration " << i << ": " << iterationTime << " ms" << std::endl;

        // Capture frame
        simulation->readDeviceBodies();
        storeFrame(simulation->getBodies(), nBodies, video);
    }

    // Cleanup
    csv.close();
    video.release();
    delete simulation;

    std::cout << "Simulation complete. Results saved to:" << std::endl;
    std::cout << "  Video: " << videoName << std::endl;
    std::cout << "  Timing data: " << csvName << std::endl;

    return 0;
}