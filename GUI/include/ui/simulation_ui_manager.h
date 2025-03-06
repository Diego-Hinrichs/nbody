#ifndef SIMULATION_UI_MANAGER_H
#define SIMULATION_UI_MANAGER_H

// Disable certain warnings that might interfere with OpenCV
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#endif

// C++ standard library headers
#include <vector>
#include <atomic>
#include <mutex>
#include <sstream>
#include <iomanip>
#include <string>

// Forward declarations to minimize dependencies
struct SimulationState;
struct Vector;

// Temporarily undefine any conflicting macros
#undef MAX_DIST
#undef E

// OpenCV headers
#include <opencv2/opencv.hpp>

// Restore any undefined macros
#include "../common/constants.cuh"
#include "../common/types.cuh"
#include "simulation_state.h"

// Restore diagnostic settings
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

class SimulationUIManager
{
public:
    SimulationUIManager(SimulationState &state);

    void drawBodies(cv::Mat &image);
    void drawCommandMenu(cv::Mat &image);
    void handleKeyboardEvents(int key);
    void setupWindow(bool fullscreen);

    static void mouseCallback(int event, int x, int y, int flags, void *userdata);

private:
    SimulationState &simulationState_;

    cv::Point scaleToWindow(const Vector &pos3D, double zoomFactor, double offsetX, double offsetY);
    void executeMenuCommand(int commandIndex);
};

#endif // SIMULATION_UI_MANAGER_H