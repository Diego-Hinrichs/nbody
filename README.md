# TODOS 21/03/2025

- [x] Implement Morton and Hilbert as GPU-accelerated methods
- [x] Set all masses to be equal for now
- [x] Add a new option to choose the mass distribution: uniform or normal
- [ ] Optimize the frequency at which the points are re-ordered
- [ ] Implement the SPH method with fixed radius (parameter `r`)
- [ ] Allow the program to be run via the terminal with essential options:
```bash
./prog \
  -n <particulas> \  
  -sort <0: nada, 1: hilbert, 2: morton> \    
  -steps <pasos> \
  -mdist <0: uniforme, 1: normal> \
  -alg <0: cpu-direct-sum, 1: cpu-barnes, 2: gpu-direct-sum,  3: gpu-barnes-hut> \
  -theta <float> \
  -visual <0: off, 1: on> \  
  -energy <archivo-salida.txt> \  
  -nt <num-threads>
  -bs <block-size>
  ```
  