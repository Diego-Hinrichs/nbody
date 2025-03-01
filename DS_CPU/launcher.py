#!/usr/bin/env python3

import subprocess

# Lista de valores de x
valores_x = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000, 1024000, 2048000, 4096000, 8192000]

for x in valores_x:
    # Construimos el parámetro y el nombre de salida
    parametro = f"{x}"
    salida = f"bh_{x}.csv"
    
    # Armamos el comando con redirección de salida
    comando = f"./build/main {parametro} 1 1 1000 > tests/{salida}"
    
    # Ejecutamos el comando
    subprocess.run(comando, shell=True, check=True)
