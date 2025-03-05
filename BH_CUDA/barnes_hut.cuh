#ifndef BARNES_HUT_CUDA_H_
#define BARNES_HUT_CUDA_H_

// Para medir los tiempos de cada kernel
struct UpdateTimes {
    float resetTimeMs;   // tiempo en ms de resetCUDA()
    float bboxTimeMs;    // tiempo en ms de computeBoundingBoxCUDA()
    float octreeTimeMs;  // tiempo en ms de constructOctreeCUDA()
    float forceTimeMs;   // tiempo en ms de computeForceCUDA()
};

typedef struct
{
    double x;
    double y;
    double z;
} Vector;

typedef struct
{
    bool isDynamic;
    double mass;
    double radius;
    Vector position;
    Vector velocity;
    Vector acceleration;

} Body;

typedef struct
{
    Vector topLeftFront;
    Vector botRightBack;
    Vector centerMass;
    double totalMass;
    bool isLeaf;
    int start;
    int end;

} Node;

class BarnesHutCuda
{
    int nBodies;
    int nNodes;
    int leafLimit;

    Body *h_b;
    Node *h_node;

    Body *d_b;
    Body *d_b_buffer;
    Node *d_node;
    int *d_mutex;

    void initRandomBodies();
    void setBody(int i, bool isDynamic, double mass, double radius, Vector position, Vector velocity, Vector acceleration);
    void resetCUDA();
    void computeBoundingBoxCUDA();
    void constructOctreeCUDA();
    void computeForceCUDA();

public:
    BarnesHutCuda(int n);
    ~BarnesHutCuda();
    UpdateTimes update();
    void setup(int sim);
    void readDeviceBodies();
    void debugPrintDeviceBodies();
    void debugPrintTree();
    Body *getBodies();
};

#endif