/*
   Copyright 2023 Hsin-Hung Wu

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <opencv2/opencv.hpp>
#include "barnesHutCuda.cuh"
#include "constants.h"
#include "err.h"

// Video writer for storing frames
cv::VideoWriter video;

/**
 * Projects a 3D position to a 2D point in the window.
 * Uses a simple orthographic projection for this example.
 */
cv::Point scaleToWindow(const Vector &pos)
{
    double scaleX = WINDOW_WIDTH / (NBODY_WIDTH * 2.0);
    double scaleY = WINDOW_HEIGHT / (NBODY_HEIGHT * 2.0);
    double screenX = (pos.x + NBODY_WIDTH) * scaleX;
    double screenY = (pos.y + NBODY_HEIGHT) * scaleY;
    return cv::Point((int)screenX, (int)(WINDOW_HEIGHT - screenY));
}

/**
 * Generates a color based on the SFC code of a body for visualization
 */
cv::Scalar getColorFromSFCCode(uint64_t sfcCode)
{
    // Use the bits from SFC code to create a color
    unsigned char r = (sfcCode) & 0xFF;
    unsigned char g = (sfcCode >> 8) & 0xFF;
    unsigned char b = (sfcCode >> 16) & 0xFF;
    return cv::Scalar(b, g, r);
}

/**
 * Stores a frame of the simulation to video
 */
void storeFrame(Body *bodies, int nBodies, int frameNum)
{
    cv::Mat image = cv::Mat::zeros(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC3);

    // Draw the bodies on the image
    for (int i = 0; i < nBodies; i++)
    {
        cv::Point center = scaleToWindow(bodies[i].position);

        // Color based on SFC order for better visualization
        // For bodies not using SFC ordering, this would create consistent coloring
        // For bodies with SFC ordering, this shows the spatial partitioning
        cv::Scalar color = getColorFromSFCCode(i);

        // Size based on body mass (scaled logarithmically)
        double sizeScale = log10(bodies[i].mass / EARTH_MASS) * 0.5;
        int size = std::max(1, (int)(sizeScale));

        cv::circle(image, center, size, color, -1);
    }

    // Optional: Add frame number and other info
    cv::putText(image, "Frame: " + std::to_string(frameNum),
                cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(255, 255, 255), 2);

    // Write frame to video
    video.write(image);

    // Optionally save individual frames
    // cv::imwrite("frames/frame_" + std::to_string(frameNum) + ".png", image);
}

void printUsage()
{
    std::cerr << "Usage: ./barnesHut <nBodies> <sim> <iters> [sfc] [target] [video]" << std::endl;
    std::cerr << "  nBodies: Number of bodies in the simulation" << std::endl;
    std::cerr << "  sim: Simulation type" << std::endl;
    std::cerr << "  iters: Number of iterations to run" << std::endl;
    std::cerr << "  sfc (optional): Space-filling curve type" << std::endl;
    std::cerr << "    0: None (default)" << std::endl;
    std::cerr << "    1: Morton (Z-order)" << std::endl;
    std::cerr << "    2: Hilbert" << std::endl;
    std::cerr << "  target (optional): What to order using SFC" << std::endl;
    std::cerr << "    0: None (default)" << std::endl;
    std::cerr << "    1: Bodies" << std::endl;
    std::cerr << "    2: Octants" << std::endl;
    std::cerr << "  video (optional): Whether to generate video output" << std::endl;
    std::cerr << "    0: No video (default)" << std::endl;
    std::cerr << "    1: Generate video" << std::endl;
}

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        printUsage();
        return 1;
    }

    int nBodies = atoi(argv[1]);
    int sim = atoi(argv[2]);
    int iters = atoi(argv[3]);

    // Default values for SFC parameters
    SFCType sfcType = NO_SFC;
    OrderTarget orderTarget = ORDER_NONE;
    bool generateVideo = false;

    // Parse optional SFC type parameter
    if (argc >= 5)
    {
        int sfcParam = atoi(argv[4]);
        switch (sfcParam)
        {
        case 1:
            sfcType = MORTON;
            std::cout << "Using Morton (Z-order) curve" << std::endl;
            break;
        case 2:
            sfcType = HILBERT;
            std::cout << "Using Hilbert curve" << std::endl;
            break;
        default:
            sfcType = NO_SFC;
            std::cout << "No space-filling curve ordering" << std::endl;
            break;
        }
    }

    // Parse optional target parameter
    if (argc >= 6)
    {
        int targetParam = atoi(argv[5]);
        switch (targetParam)
        {
        case 1:
            orderTarget = ORDER_BODIES;
            std::cout << "Ordering bodies" << std::endl;
            break;
        case 2:
            orderTarget = ORDER_OCTANTS;
            std::cout << "Ordering octants" << std::endl;
            break;
        default:
            orderTarget = ORDER_NONE;
            std::cout << "No ordering" << std::endl;
            break;
        }
    }

    // Parse optional video parameter
    if (argc >= 7)
    {
        generateVideo = (atoi(argv[6]) > 0);
    }

    // Initialize video writer if requested
    if (generateVideo)
    {
        std::string videoFilename;

        if (sfcType == MORTON)
        {
            videoFilename = "barnes_hut_morton";
        }
        else if (sfcType == HILBERT)
        {
            videoFilename = "barnes_hut_hilbert";
        }
        else
        {
            videoFilename = "barnes_hut";
        }

        if (orderTarget == ORDER_BODIES)
        {
            videoFilename += "_bodies";
        }
        else if (orderTarget == ORDER_OCTANTS)
        {
            videoFilename += "_octants";
        }

        videoFilename += ".avi";

        std::cout << "Generating video output to: " << videoFilename << std::endl;

        // Initialize video writer with MJPG codec
        video.open(videoFilename,
                   cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                   30, // Frames per second
                   cv::Size(WINDOW_WIDTH, WINDOW_HEIGHT));

        if (!video.isOpened())
        {
            std::cerr << "Error: Could not open video writer." << std::endl;
            return 1;
        }
    }

    // Initialize Barnes-Hut with SFC parameters
    BarnesHutCuda *bh = new BarnesHutCuda(nBodies, sfcType, orderTarget);
    bh->setup(sim);

    using Clock = std::chrono::steady_clock;

    // Print CSV header for timing information
    std::cout << "iteration"
              << ",totalIterationMs"
              << ",resetTimeMs"
              << ",bboxTimeMs"
              << ",sfcTimeMs" // Added SFC timing
              << ",octreeTimeMs"
              << ",forceTimeMs"
              << std::endl;

    // Main simulation loop
    for (int i = 0; i < iters; ++i)
    {
        // CPU timer for overall iteration
        auto start = Clock::now();

        // Run a simulation step and get timing information
        UpdateTimes kernelTimes = bh->update();

        // If generating video, get body data from device
        if (generateVideo)
        {
            bh->readDeviceBodies();
            storeFrame(bh->getBodies(), nBodies, i);
        }

        // End CPU timer
        auto end = Clock::now();

        // Calculate total iteration time in milliseconds
        double iterationTime = std::chrono::duration<double, std::milli>(end - start).count();

        // Print CSV line with timing information
        std::cout << i << ","             // Iteration index
                  << iterationTime << "," // Total time (CPU + GPU)
                  << kernelTimes.resetTimeMs << ","
                  << kernelTimes.bboxTimeMs << ","
                  << kernelTimes.sfcTimeMs << "," // Added SFC timing
                  << kernelTimes.octreeTimeMs << ","
                  << kernelTimes.forceTimeMs
                  << std::endl;
    }

    // Close video if open
    if (generateVideo && video.isOpened())
    {
        video.release();
        std::cout << "Video generation complete." << std::endl;
    }

    // Cleanup
    delete bh;

    return 0;
}