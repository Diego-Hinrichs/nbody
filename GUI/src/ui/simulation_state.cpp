#include "../../include/ui/simulation_state.h"

SimulationState::SimulationState() : running(true),
                                     restart(false),
                                     useSFC(false),
                                     isPaused(false),
                                     numBodies(1000),
                                     zoomFactor(1.0),
                                     offsetX(0.0),
                                     offsetY(0.0),
                                     sharedBodies(nullptr),
                                     currentBodiesCount(0),
                                     fps(0.0),
                                     lastIterationTime(0.0),
                                     showCommandMenu(false),
                                     selectedCommandIndex(0),
                                     selectedParticleOption(0)
{
    // Default command options
    commandOptions = {
        "Reiniciar Simulacion",
        "Numero de Particulas",
        "Activar/Desactivar SFC",
        "Pausar/Reanudar",
        "Cerrar Menu"};

    // Default particle options
    particleOptions = {"1000", "5000", "10000", "15000", "20000"};
}

SimulationState::~SimulationState()
{
    // Free shared bodies if allocated
    if (sharedBodies != nullptr)
    {
        delete[] sharedBodies;
        sharedBodies = nullptr;
    }
}