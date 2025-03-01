# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/diegoh/tesis/Finales/N-Body/BH_CUDA/build/_deps/glm-src"
  "/home/diegoh/tesis/Finales/N-Body/BH_CUDA/build/_deps/glm-build"
  "/home/diegoh/tesis/Finales/N-Body/BH_CUDA/build/_deps/glm-subbuild/glm-populate-prefix"
  "/home/diegoh/tesis/Finales/N-Body/BH_CUDA/build/_deps/glm-subbuild/glm-populate-prefix/tmp"
  "/home/diegoh/tesis/Finales/N-Body/BH_CUDA/build/_deps/glm-subbuild/glm-populate-prefix/src/glm-populate-stamp"
  "/home/diegoh/tesis/Finales/N-Body/BH_CUDA/build/_deps/glm-subbuild/glm-populate-prefix/src"
  "/home/diegoh/tesis/Finales/N-Body/BH_CUDA/build/_deps/glm-subbuild/glm-populate-prefix/src/glm-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/diegoh/tesis/Finales/N-Body/BH_CUDA/build/_deps/glm-subbuild/glm-populate-prefix/src/glm-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/diegoh/tesis/Finales/N-Body/BH_CUDA/build/_deps/glm-subbuild/glm-populate-prefix/src/glm-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
