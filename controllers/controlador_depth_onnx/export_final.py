import torch
import os
import sys

# Importamos la clase directamente del archivo local
from depth_anything.dpt import DepthAnything

def exportar_onnx_manual():
    print("-> 1. Configurando modelo Depth Anything (Versión Small/Vits)...")
    
    # --- CONFIGURACIÓN MANUAL PARA 'VITS' ---
    # Estos son los números mágicos para la versión pequeña.
    # Si no los ponemos, intenta cargar la versión gigante y fallará al cargar los pesos.
    config = {
        'encoder': 'vits',
        'features': 64,
        'out_channels': [48, 96, 192, 384],
        'localhub': True  # Importante: Usa los archivos locales que ya tienes
    }
    
    # Instanciamos el modelo pasando el diccionario
    try:
        model = DepthAnything(config)
        print("-> Modelo instanciado correctamente en memoria.")
    except Exception as e:
        print(f"ERROR instanciando modelo: {e}")
        print("Asegúrate de ejecutar esto desde la carpeta ~/TFG/Depth-Anything")
        return

    # --- CARGAR PESOS ---
    # Los pesos están en la carpeta de al lado, vamos a buscarlos
    weights_path = "../depth_and_distance_measure/depth_anything_vits14.pth"
    
    print(f"-> 2. Cargando pesos desde: {weights_path}")
    if not os.path.exists(weights_path):
        print("ERROR: No encuentro el archivo .pth. ¿Lo descargaste con wget antes?")
        return
        
    state_dict = torch.load(weights_path, map_location='cpu')
    
    # A veces los pesos vienen con prefijos extra, limpiamos si es necesario
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    
    try:
        model.load_state_dict(state_dict)
        print("-> ¡Pesos cargados! El modelo está listo.")
    except Exception as e:
        print(f"ERROR de coincidencia de capas: {e}")
        return

    model.eval()
    
    # --- EXPORTAR A ONNX ---
    # Resolución optimizada para Jetson
    target_size = 364 #debe ser multiplo de 14 
    dummy_input = torch.randn(1, 3, target_size, target_size)
    output_file = f"depth_anything_vits14_{target_size}_jetson.onnx"
    
    print(f"-> 3. Generando ONNX ({target_size}x{target_size})...")
    torch.onnx.export(
        model, 
        dummy_input, 
        output_file, 
        opset_version=11, #el del pc se uso esto en 18
        input_names=['input'],
        output_names=['depth'],
        dynamic_axes={'input': {0: 'batch_size'}, 'depth': {0: 'batch_size'}},
        do_constant_folding=True
    )
    
    print("-" * 30)
    print(f"-> ¡ÉXITO! Archivo creado en: {os.getcwd()}/{output_file}")
    print("-> Ahora mueve este archivo a tu PC/Jetson para las pruebas.")
    print("-" * 30)

if __name__ == "__main__":
    exportar_onnx_manual()
