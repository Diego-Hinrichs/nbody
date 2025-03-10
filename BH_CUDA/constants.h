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

#ifndef CONSTANTS_H
#define CONSTANTS_H

#define HBL 1.6e29
#define WINDOW_WIDTH 2048
#define WINDOW_HEIGHT 2048
#define NBODY_WIDTH 10.0e11
#define NBODY_HEIGHT 10.0e11
// Parámetros de la simulación
#define GRAVITY     6.67430e-11
#define E           0.5
#define DT          25000.0
#define THETA       0.5
#define COLLISION_TH 1.0e10

// Kernel/Bloques
#define BLOCK_SIZE  1024
#define MAX_NODES   349525
#define N_LEAF      262144

// Constantes astronómicas para normalizar
#define MAX_DIST    5.0e11  // Distancia máxima (usada en initRandomBodies)
#define MIN_DIST    2.0e10
#define EARTH_MASS  5.974e24
#define EARTH_DIA   12756.0
#define SUN_MASS    1.989e30
#define SUN_DIA     1.3927e6
#define CENTERX 0
#define CENTERY 0
#define CENTERZ 0
#endif