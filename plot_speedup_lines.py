import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import sys

def plot_speedup_lines(csv_file):
    # Cargar el archivo CSV
    print(f"Cargando datos desde: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Filtrar solo los registros de tipo "final" para obtener tiempos totales
    df_final = df[df['type'] == 'final']
    
    # Mapeo de algoritmos para las etiquetas
    algorithm_names = {
        0: "CPU DIRECT SUM",
        1: "CPU SFC DIRECT SUM",
        2: "GPU DIRECT SUM",
        3: "GPU SFC DIRECT SUM"
    }
    
                    # 0: "CPU_DIRECT_SUM",
        # 1: "CPU_SFC_DIRECT_SUM",
        # 2: "GPU_DIRECT_SUM",
        # 3: "GPU_SFC_DIRECT_SUM"
    # Definir colores para cada algoritmo
    colors = {
        0: 'red',
        1: 'blue',
        2: 'green',
        3: 'purple'
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
                'speedup': speedup,
                'time_ms': time_ms
            })
    
    # Convertir los datos de speedup a un DataFrame
    speedup_df = pd.DataFrame(speedup_data)
    
    # Para cada algoritmo, crear una línea en el gráfico
    for alg in sorted(speedup_df['algorithm'].unique()):
        if alg == 0:  # Omitir el algoritmo base (speedup = 1)
            continue
            
        alg_data = speedup_df[speedup_df['algorithm'] == alg]
        
        # Ordenar por número de cuerpos
        alg_data = alg_data.sort_values(by='bodies')
        
        plt.plot(
            alg_data['bodies'], 
            alg_data['speedup'], 
            'o-', 
            label=algorithm_names.get(alg, f"Algorithm {alg}"),
            color=colors.get(alg, 'black'),
            linewidth=2,
            markersize=8
        )
    
    # Añadir línea del método base (speedup = 1)
    plt.axhline(y=1, color='red', linestyle='--', label=algorithm_names.get(0, "CPU Direct Sum"))
    
    # Configurar gráfico
    plt.xlabel('Número de cuerpos', fontsize=12)
    plt.ylabel('Speedup (veces más rápido que CPU Direct Sum)', fontsize=12)
    plt.title('Tendencia de Speedup vs. Número de cuerpos', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Usar escala logarítmica para el eje X si hay varios órdenes de magnitud
    if max(body_sizes) / min(body_sizes) > 100:
        plt.xscale('log')
        plt.xticks(body_sizes, labels=[str(x) for x in body_sizes])
    
    # Añadir anotaciones con los valores de speedup
    for alg in sorted(speedup_df['algorithm'].unique()):
        if alg == 0:
            continue
            
        alg_data = speedup_df[speedup_df['algorithm'] == alg]
        alg_data = alg_data.sort_values(by='bodies')
        
        for idx, row in alg_data.iterrows():
            plt.annotate(
                f"{row['speedup']:.2f}x",
                (row['bodies'], row['speedup']),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center'
            )
    
    # Guardar la figura
    output_file = os.path.splitext(csv_file)[0] + '_speedup_lines.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Gráfica guardada como: {output_file}")
    
    # Mostrar la gráfica
    plt.show()
    
    # Crear tabla de resumen
    print("\nTabla de resumen de speedup:")
    pivot_table = speedup_df.pivot(index='bodies', columns='algorithm', values='speedup')
    
    # Renombrar columnas
    pivot_table.columns = [algorithm_names.get(alg, f"Algorithm {alg}") for alg in pivot_table.columns]
    
    # Imprimir tabla
    print(pivot_table)
    
    # Guardar tabla en CSV
    table_file = os.path.splitext(csv_file)[0] + '_speedup_table.csv'
    pivot_table.to_csv(table_file)
    print(f"Tabla guardada como: {table_file}")

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
    
    plot_speedup_lines(csv_file)

if __name__ == "__main__":
    main() 