import os
import sys
import glob
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Analizar resultados de la simulación N-Body")
    parser.add_argument('--csv', type=str, help='Archivo CSV específico para analizar')
    parser.add_argument('--latest', action='store_true', help='Usar el archivo CSV más reciente')
    args = parser.parse_args()

    csv_file = None

    # Determinar qué archivo CSV usar
    if args.csv:
        if os.path.exists(args.csv):
            csv_file = args.csv
        else:
            print(f"Error: El archivo {args.csv} no existe")
            return
    elif args.latest or not csv_file:
        # Buscar el archivo CSV más reciente
        csv_files = glob.glob("nbody_results_*.csv")
        if not csv_files:
            print("Error: No se encontraron archivos CSV")
            return
        
        # Obtener el más reciente por fecha de modificación
        csv_file = max(csv_files, key=os.path.getmtime)
    
    if not csv_file:
        print("Error: No se encontró ningún archivo CSV para analizar")
        return
    
    print(f"Analizando archivo: {csv_file}")
    
    # Ejecutar todos los scripts de análisis
    scripts = [
        "plot_speedup.py",
        "plot_speedup_lines.py",
        "plot_execution_time.py"
    ]
    
    for script in scripts:
        if os.path.exists(script):
            print(f"\nEjecutando {script}...")
            try:
                subprocess.run([sys.executable, script, csv_file], check=True)
                print(f"Ejecución de {script} completada.")
            except subprocess.CalledProcessError as e:
                print(f"Error al ejecutar {script}: {e}")
        else:
            print(f"Advertencia: El script {script} no existe")
    
    print("\nAnálisis completo.")
    print(f"Archivo analizado: {csv_file}")
    
    # Mostrar los archivos generados
    base_name = os.path.splitext(csv_file)[0]
    generated_files = glob.glob(f"{base_name}_*.png") + glob.glob(f"{base_name}_*.csv")
    
    if generated_files:
        print("\nArchivos generados:")
        for file in sorted(generated_files):
            print(f"- {file}")
    
    print("\nPara ver los resultados del análisis, abra los archivos PNG generados.")

if __name__ == "__main__":
    main() 