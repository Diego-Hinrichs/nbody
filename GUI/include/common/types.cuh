#ifndef TYPES_CUH
#define TYPES_CUH

#include <cuda_runtime.h>
#include <math.h>

/**
 * @brief 3D vector structure with basic operations
 */
struct Vector
{
    double x;
    double y;
    double z;

    // Default constructor
    __host__ __device__ Vector() : x(0.0), y(0.0), z(0.0) {}

    // Constructor with initial values
    __host__ __device__ Vector(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    // Vector addition
    __host__ __device__ Vector operator+(const Vector &other) const
    {
        return Vector(x + other.x, y + other.y, z + other.z);
    }

    // Vector subtraction
    __host__ __device__ Vector operator-(const Vector &other) const
    {
        return Vector(x - other.x, y - other.y, z - other.z);
    }

    // Scalar multiplication
    __host__ __device__ Vector operator*(double scalar) const
    {
        return Vector(x * scalar, y * scalar, z * scalar);
    }

    // Dot product
    __host__ __device__ double dot(const Vector &other) const
    {
        return x * other.x + y * other.y + z * other.z;
    }

    // Vector length squared
    __host__ __device__ double lengthSquared() const
    {
        return x * x + y * y + z * z;
    }

    // Vector length
    __host__ __device__ double length() const
    {
        return sqrt(lengthSquared());
    }

    // Vector normalization
    __host__ __device__ Vector normalize() const
    {
        double len = length();
        if (len > 0.0)
        {
            return Vector(x / len, y / len, z / len);
        }
        return *this;
    }

    // Distance between two vectors
    __host__ __device__ static double distance(const Vector &a, const Vector &b)
    {
        return (a - b).length();
    }

    // Distance squared between two vectors (more efficient)
    __host__ __device__ static double distanceSquared(const Vector &a, const Vector &b)
    {
        return (a - b).lengthSquared();
    }
};

/**
 * @brief Body structure representing a celestial body
 */
struct Body
{
    bool isDynamic;      // Whether the body moves or is static
    double mass;         // Mass of the body
    double radius;       // Radius of the body
    Vector position;     // Position in 3D space
    Vector velocity;     // Velocity vector
    Vector acceleration; // Acceleration vector

    // Default constructor
    __host__ __device__ Body() : isDynamic(true),
                                 mass(0.0),
                                 radius(0.0),
                                 position(),
                                 velocity(),
                                 acceleration() {}
};

/**
 * @brief Node structure for the octree
 */
struct Node
{
    Vector topLeftFront; // Minimum corner of bounding box
    Vector botRightBack; // Maximum corner of bounding box
    Vector centerMass;   // Center of mass
    double totalMass;    // Total mass of bodies in this node
    bool isLeaf;         // Whether this is a leaf node
    int start;           // Start index of bodies in this node
    int end;             // End index of bodies in this node

    // Default constructor
    __host__ __device__ Node() : topLeftFront(),
                                 botRightBack(),
                                 centerMass(),
                                 totalMass(0.0),
                                 isLeaf(true),
                                 start(-1),
                                 end(-1) {}

    // Get the center of the node's bounding box
    __host__ __device__ Vector getCenter() const
    {
        return Vector(
            (topLeftFront.x + botRightBack.x) * 0.5,
            (topLeftFront.y + botRightBack.y) * 0.5,
            (topLeftFront.z + botRightBack.z) * 0.5);
    }

    // Get the width of the node's bounding box (max side length)
    __host__ __device__ double getWidth() const
    {
        return fmax(
            fabs(botRightBack.x - topLeftFront.x),
            fmax(
                fabs(botRightBack.y - topLeftFront.y),
                fabs(botRightBack.z - topLeftFront.z)));
    }

    // Check if the node contains a point
    __host__ __device__ bool contains(const Vector &point) const
    {
        return (
            point.x >= topLeftFront.x && point.x <= botRightBack.x &&
            point.y <= topLeftFront.y && point.y >= botRightBack.y &&
            point.z >= topLeftFront.z && point.z <= botRightBack.z);
    }
};

/**
 * @brief Performance metrics for simulation timing
 */
struct SimulationMetrics
{
    float resetTimeMs;
    float bboxTimeMs;
    float octreeTimeMs;
    float forceTimeMs;
    float totalTimeMs;

    SimulationMetrics() : resetTimeMs(0.0f),
                          bboxTimeMs(0.0f),
                          octreeTimeMs(0.0f),
                          forceTimeMs(0.0f),
                          totalTimeMs(0.0f) {}
};

#endif // TYPES_CUH