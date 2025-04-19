# Análisis de Speedup para Simulación N-Body

Este conjunto de scripts permite analizar los resultados de las simulaciones N-body y visualizar los speedups de los diferentes métodos en relación al método base (Direct Sum).

## Requisitos

Asegúrate de tener instaladas las siguientes bibliotecas de Python:

```bash
pip install pandas matplotlib numpy
```

## Archivos

- `analyze_results.py`: Script principal que ejecuta todos los análisis
- `plot_speedup.py`: Genera un gráfico de barras de speedup
- `plot_speedup_lines.py`: Genera un gráfico de líneas de speedup y tabla CSV
- `plot_execution_time.py`: Genera gráficos de tiempos de ejecución y tabla CSV

## Uso

### Análisis completo (recomendado)

Para ejecutar todos los análisis de una vez:

```bash
python analyze_results.py
```

Por defecto, esto analizará el archivo CSV más reciente en el directorio actual.

Para especificar un archivo CSV particular:

```bash
python analyze_results.py --csv ruta/al/archivo.csv
```

### Scripts individuales

También puedes ejecutar cada script individualmente:

```bash
python plot_speedup.py [ruta/al/archivo.csv]
python plot_speedup_lines.py [ruta/al/archivo.csv]
python plot_execution_time.py [ruta/al/archivo.csv]
```

Si no se proporciona un archivo CSV, los scripts usarán el más reciente del directorio.

## Resultados

Los scripts generarán varios archivos:

- `*_speedup.png`: Gráfico de barras de speedup
- `*_speedup_lines.png`: Gráfico de líneas de speedup
- `*_speedup_table.csv`: Tabla de datos de speedup
- `*_execution_time.png`: Gráfico de líneas de tiempos de ejecución
- `*_execution_time_bars.png`: Gráfico de barras de tiempos de ejecución
- `*_execution_time_table.csv`: Tabla de datos de tiempos de ejecución

## Interpretación de resultados

- **Speedup**: Muestra cuántas veces más rápido es un método en comparación con el método base (CPU Direct Sum). Un valor mayor que 1 indica que el método es más rápido.
  
- **Tiempo de ejecución**: Muestra el tiempo absoluto de ejecución en milisegundos para cada método y número de cuerpos.

## Notas

- Los scripts filtran los registros de tipo "final" en el CSV para obtener los tiempos totales.
- Para el benchmark, asegúrate de que todos los métodos se ejecuten con el mismo número de cuerpos para obtener comparaciones válidas.
- El speedup se calcula como: tiempo_base / tiempo_método 