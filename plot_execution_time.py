import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import sys

def plot_execution_time(csv_file):
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
    
    # Definir colores para cada algoritmo
    colors = {
        0: 'red',
        1: 'blue',
        2: 'green',
        3: 'purple'
    }
    
    # Crear una figura para el gráfico
    plt.figure(figsize=(14, 10))
    
    # Grupos de tamaños de cuerpos
    body_sizes = sorted(df_final['bodies'].unique())
    
    # Para cada algoritmo, crear una línea en el gráfico
    for alg in sorted(df_final['algorithm'].unique()):
        alg_data = df_final[df_final['algorithm'] == alg]
        
        # Agrupar por número de cuerpos y calcular el tiempo promedio
        avg_times = alg_data.groupby('bodies')['time_ms'].mean().reset_index()
        
        # Ordenar por número de cuerpos
        avg_times = avg_times.sort_values(by='bodies')
        
        plt.plot(
            avg_times['bodies'], 
            avg_times['time_ms'], 
            'o-', 
            label=algorithm_names.get(alg, f"Algorithm {alg}"),
            color=colors.get(alg, 'black'),
            linewidth=2,
            markersize=8
        )
    
    # Configurar gráfico
    plt.xlabel('Número de cuerpos', fontsize=12)
    plt.ylabel('Tiempo de ejecución (ms)', fontsize=12)
    plt.title('Tiempo de ejecución vs. Número de cuerpos', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Usar escala logarítmica para ambos ejes si hay varios órdenes de magnitud
    if max(body_sizes) / min(body_sizes) > 100:
        plt.xscale('log')
        plt.xticks(body_sizes, labels=[str(x) for x in body_sizes])
    
    plt.yscale('log')  # Escala logarítmica para el tiempo
    
    # Añadir anotaciones con los valores de tiempo
    for alg in sorted(df_final['algorithm'].unique()):
        alg_data = df_final[df_final['algorithm'] == alg]
        avg_times = alg_data.groupby('bodies')['time_ms'].mean().reset_index()
        avg_times = avg_times.sort_values(by='bodies')
        
        for idx, row in avg_times.iterrows():
            plt.annotate(
                f"{row['time_ms']:.1f} ms",
                (row['bodies'], row['time_ms']),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center'
            )
    
    # Guardar la figura
    output_file = os.path.splitext(csv_file)[0] + '_execution_time.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Gráfica guardada como: {output_file}")
    
    # Mostrar la gráfica
    plt.show()
    
    # Crear y mostrar tabla de comparación de tiempos
    print("\nTabla de tiempos de ejecución (ms):")
    time_table = df_final.pivot_table(index='bodies', columns='algorithm', values='time_ms', aggfunc='mean')
    
    # Renombrar columnas
    time_table.columns = [algorithm_names.get(alg, f"Algorithm {alg}") for alg in time_table.columns]
    
    # Imprimir tabla
    print(time_table)
    
    # Guardar tabla en CSV
    table_file = os.path.splitext(csv_file)[0] + '_execution_time_table.csv'
    time_table.to_csv(table_file)
    print(f"Tabla guardada como: {table_file}")
    
    # También graficar en formato de barras
    plt.figure(figsize=(14, 8))
    
    # Crear una lista para cada tamaño de cuerpos
    x = np.arange(len(body_sizes))
    width = 0.2  # Ancho de las barras
    
    # Por cada algoritmo, crear un conjunto de barras
    for i, alg in enumerate(sorted(df_final['algorithm'].unique())):
        alg_data = df_final[df_final['algorithm'] == alg]
        times = []
        
        for bodies in body_sizes:
            alg_body_data = alg_data[alg_data['bodies'] == bodies]
            if len(alg_body_data) > 0:
                times.append(alg_body_data['time_ms'].mean())
            else:
                times.append(0)
        
        plt.bar(
            x + i*width - width*len(df_final['algorithm'].unique())/2 + width/2, 
            times, 
            width, 
            label=algorithm_names.get(alg, f"Algorithm {alg}"),
            color=colors.get(alg, 'black')
        )
    
    plt.xlabel('Número de cuerpos', fontsize=12)
    plt.ylabel('Tiempo de ejecución (ms)', fontsize=12)
    plt.title('Comparación de tiempos de ejecución por algoritmo', fontsize=14)
    plt.xticks(x, body_sizes)
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Guardar la figura de barras
    output_file_bars = os.path.splitext(csv_file)[0] + '_execution_time_bars.png'
    plt.savefig(output_file_bars, dpi=300, bbox_inches='tight')
    print(f"Gráfica de barras guardada como: {output_file_bars}")
    
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
    
    plot_execution_time(csv_file)

if __name__ == "__main__":
    main() 