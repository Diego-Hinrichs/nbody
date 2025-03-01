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
#include "barnesHutCuda.cuh"
#include "constants.h"


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
              //   << ",resetTimeMs"
              //   << ",bboxTimeMs"
              //   << ",octreeTimeMs"
              //   << ",forceTimeMs"
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
        //   << kernelTimes.resetTimeMs << ","
        //   << kernelTimes.bboxTimeMs << ","
        //   << kernelTimes.octreeTimeMs << ","
        //   << kernelTimes.forceTimeMs
    }

    delete bh;

    return 0;
}
