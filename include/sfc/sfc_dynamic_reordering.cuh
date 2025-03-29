#ifndef SFC_DYNAMIC_REORDERING_CUH
#define SFC_DYNAMIC_REORDERING_CUH

#include <deque>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <cfloat>

class SFCDynamicReorderingStrategy
{
private:
    // Parameters for the optimization formula
    double reorderTime;        // Time to reorder (equivalent to Rt)
    double postReorderSimTime; // Simulation time right after reordering (equivalent to Rq)
    double updateTime;         // Time to update without reordering (equivalent to Ut, typically 0 for SFC)
    double degradationRate;    // Average performance degradation per iteration (equivalent to dQ)

    int iterationsSinceReorder;  // Counter for iterations since last reorder
    int currentOptimalFrequency; // Current calculated optimal frequency

    // Tracking metrics for dynamic calculation
    int metricsWindowSize;
    std::deque<double> reorderTimeHistory;
    std::deque<double> postReorderSimTimeHistory;
    std::deque<double> simulationTimeHistory;

    // Calculate the optimal reordering frequency
    int computeOptimalFrequency(int totalIterations)
    {
        // Using the formula: ((nU*nU*dQ/2) + nU*(Ut+Rq) + (Rt+Rq)) * Nit/(nU+1)
        // The optimal frequency is where the derivative = 0

        double determinant = 1.0 - 2.0 * (updateTime - reorderTime) / degradationRate;

        // If determinant is negative, use a default value
        if (determinant < 0)
            return 10; // Default to 10 as a reasonable value

        double optNu = -1.0 + sqrt(determinant);

        // Convert to integer values and check which one is better
        int nu1 = static_cast<int>(optNu);
        int nu2 = nu1 + 1;

        if (nu1 <= 0)
            return 1; // Avoid negative or zero values

        // Calculate total time with nu1 and nu2
        double time1 = ((nu1 * nu1 * degradationRate / 2.0) + nu1 * (updateTime + postReorderSimTime) +
                        (reorderTime + postReorderSimTime)) *
                       totalIterations / (nu1 + 1.0);
        double time2 = ((nu2 * nu2 * degradationRate / 2.0) + nu2 * (updateTime + postReorderSimTime) +
                        (reorderTime + postReorderSimTime)) *
                       totalIterations / (nu2 + 1.0);

        return time1 < time2 ? nu1 : nu2;
    }

    // Update metrics with new timing information
    void updateMetrics(double newReorderTime, double newSimTime)
    {
        // Update reorder time if available
        if (newReorderTime > 0)
        {
            reorderTimeHistory.push_back(newReorderTime);
            if (reorderTimeHistory.size() > metricsWindowSize)
            {
                reorderTimeHistory.pop_front();
            }

            // Recalculate average reorder time
            reorderTime = std::accumulate(reorderTimeHistory.begin(), reorderTimeHistory.end(), 0.0) /
                          reorderTimeHistory.size();
        }

        // Track simulation times to calculate degradation
        simulationTimeHistory.push_back(newSimTime);
        if (simulationTimeHistory.size() > metricsWindowSize)
        {
            simulationTimeHistory.pop_front();
        }

        // If first simulation after reorder, update postReorderSimTime
        if (iterationsSinceReorder == 1)
        {
            postReorderSimTimeHistory.push_back(newSimTime);
            if (postReorderSimTimeHistory.size() > metricsWindowSize)
            {
                postReorderSimTimeHistory.pop_front();
            }

            postReorderSimTime = std::accumulate(postReorderSimTimeHistory.begin(),
                                                 postReorderSimTimeHistory.end(), 0.0) /
                                 postReorderSimTimeHistory.size();
        }

        // Calculate degradation rate if we have enough data
        if (simulationTimeHistory.size() >= 3)
        {
            // Simple linear regression on the most recent simulation times
            // to estimate the degradation rate
            int n = std::min(5, static_cast<int>(simulationTimeHistory.size()));
            double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;

            for (int i = 0; i < n; i++)
            {
                double x = i;
                double y = simulationTimeHistory[simulationTimeHistory.size() - n + i];
                sumX += x;
                sumY += y;
                sumXY += x * y;
                sumX2 += x * x;
            }

            // Calculate slope (degradation rate)
            double slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
            if (slope > 0)
            {
                degradationRate = slope;
            }
        }
    }

public:
    SFCDynamicReorderingStrategy(int windowSize = 10)
        : reorderTime(0.0),
          postReorderSimTime(0.0),
          updateTime(0.0),
          degradationRate(0.001), // Initial small degradation assumption
          iterationsSinceReorder(0),
          currentOptimalFrequency(10), // Start with a reasonable default
          metricsWindowSize(windowSize)
    {
    }

    // Check if reordering is needed based on current metrics
    bool shouldReorder(double lastSimTime, double lastReorderTime = 0.0)
    {
        iterationsSinceReorder++;

        // Update metrics with new timing information
        updateMetrics(lastReorderTime, lastSimTime);

        // Recalculate optimal frequency periodically
        if (iterationsSinceReorder % 10 == 0)
        {
            currentOptimalFrequency = computeOptimalFrequency(1000); // Assuming 1000 total iterations

            // Ensure frequency is reasonable
            currentOptimalFrequency = std::max(1, std::min(100, currentOptimalFrequency));
        }

        // Decide if we should reorder based on current counter and optimal frequency
        bool shouldReorder = iterationsSinceReorder >= currentOptimalFrequency;

        // Reset counter if reordering
        if (shouldReorder)
        {
            iterationsSinceReorder = 0;
        }

        return shouldReorder;
    }

    // Get the current optimal frequency
    int getOptimalFrequency() const
    {
        return currentOptimalFrequency;
    }

    // Get the current degradation rate estimate
    double getDegradationRate() const
    {
        return degradationRate;
    }

    // Reset the strategy
    void reset()
    {
        iterationsSinceReorder = 0;
        reorderTimeHistory.clear();
        postReorderSimTimeHistory.clear();
        simulationTimeHistory.clear();
    }
};

#endif