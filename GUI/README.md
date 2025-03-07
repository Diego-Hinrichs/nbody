cmake -S . -B build \                          
      -DCMAKE_C_COMPILER=$(which gcc) \
      -DCMAKE_CXX_COMPILER=$(which g++) \
      -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.6