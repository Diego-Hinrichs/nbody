import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import sys

def plot_speedup(csv_file):
    # Cargar el archivo CSV
    print(f"Cargando datos desde: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Filtrar solo los registros de tipo "final" para obtener tiempos totales
    df_final = df[df['type'] == 'final']
    
    # Mapeo de algoritmos para las etiquetas
    algorithm_names = {
        0: "CPU Direct Sum",
        1: "CPU SFC Direct Sum",
        2: "GPU Direct Sum",
        3: "GPU SFC Direct Sum"
    }
    
    # Crear una figura para el gráfico
    plt.figure(figsize=(12, 8))
    
    # Grupos de tamaños de cuerpos
    body_sizes = sorted(df_final['bodies'].unique())
    
    # Para cada tamaño de cuerpo, calcular el speedup
    speedup_data = []
    
    for bodies in body_sizes:
        # Filtrar por número de cuerpos
        df_bodies = df_final[df_final['bodies'] == bodies]
        
        # Agrupar por algoritmo y calcular el tiempo promedio
        avg_times = df_bodies.groupby('algorithm')['time_ms'].mean().reset_index()
        
        # Obtener el tiempo del método base (direct sum, algoritmo 0)
        base_time = avg_times[avg_times['algorithm'] == 0]['time_ms'].values
        
        if len(base_time) == 0:
            print(f"Advertencia: No se encontró tiempo base para {bodies} cuerpos")
            continue
            
        base_time = base_time[0]
        
        # Calcular speedup para cada algoritmo
        for idx, row in avg_times.iterrows():
            algorithm = row['algorithm']
            time_ms = row['time_ms']
            speedup = base_time / time_ms
            
            speedup_data.append({
                'bodies': bodies,
                'algorithm': algorithm,
                'speedup': speedup
            })
    
    # Convertir los datos de speedup a un DataFrame
    speedup_df = pd.DataFrame(speedup_data)
    
    # Crear un gráfico de barras agrupadas
    width = 0.2
    x = np.arange(len(body_sizes))
    
    for alg in sorted(speedup_df['algorithm'].unique()):
        if alg == 0:  # Omitir el algoritmo base (speedup = 1)
            continue
            
        alg_data = speedup_df[speedup_df['algorithm'] == alg]
        alg_speedups = []
        
        for bodies in body_sizes:
            speedup = alg_data[alg_data['bodies'] == bodies]['speedup'].values
            if len(speedup) > 0:
                alg_speedups.append(speedup[0])
            else:
                alg_speedups.append(0)
        
        offset = (alg - 0.5) * width
        plt.bar(x + offset, alg_speedups, width, label=algorithm_names.get(alg, f"Algorithm {alg}"))
    
    # Añadir línea del método base (speedup = 1)
    plt.axhline(y=1, color='r', linestyle='-', label=algorithm_names.get(0, "CPU Direct Sum"))
    
    plt.xlabel('Número de cuerpos')
    plt.ylabel('Speedup (veces más rápido que CPU Direct Sum)')
    plt.title('Speedup de diferentes métodos de simulación N-body en relación a CPU Direct Sum')
    plt.xticks(x, body_sizes)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Guardar la figura
    output_file = os.path.splitext(csv_file)[0] + '_speedup.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Gráfica guardada como: {output_file}")
    
    # Mostrar la gráfica
    plt.show()

def main():
    # Comprobar si se proporcionó un archivo
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        if not os.path.exists(csv_file):
            print(f"Error: El archivo {csv_file} no existe")
            return
    else:
        # Buscar el archivo CSV más reciente
        csv_files = glob.glob("nbody_results_*.csv")
        if not csv_files:
            print("Error: No se encontraron archivos CSV")
            return
        
        # Obtener el más reciente por fecha de modificación
        csv_file = max(csv_files, key=os.path.getmtime)
    
    plot_speedup(csv_file)

if __name__ == "__main__":
    main() 