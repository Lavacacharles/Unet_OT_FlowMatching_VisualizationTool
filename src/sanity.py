import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def audit_generative_data(base_path):
    base_path = "./projections"
    epochs = ['epoch_250', 'epoch_500', 'epoch_750', 'epoch_1000', 'epoch_1250', 'epoch_1500']
    files = ['labels.pth', 'mid_hiddens.pth', 'attn_maps.pth', 'generation_progress.pth']

    report = defaultdict(dict)
    global_samples_count = None

    # epoch = 'epoch_250'
    for epoch in epochs:    
        print('het')
        print(f"\\n{'='*20}\\nAuditando {epoch}\\n{'='*20}")
        epoch_path = os.path.join(base_path, epoch)

        if not os.path.exists(epoch_path):
            print(f"[ERROR] Directorio no encontrado: {epoch_path}")
            continue

        data = {}
        # 1. Carga y Estructura
        for file in files:
            file_path = os.path.join(epoch_path, file)
            if os.path.exists(file_path):
                tensor = torch.load(file_path, map_location='cpu')
                tensor = torch.tensor(tensor)
                data[file.split('.')[0]] = tensor
                print(f"[{file}] Shape: {tensor.shape}, Dtype: {tensor.dtype}, "
                        f"Device: {tensor.device}, Memoria: {tensor.element_size() * tensor.nelement() / 1e6:.2f} MB")
            else:
                print(f"[ERROR] Archivo faltante: {file_path}")

        # 2 y 3. Consistencia y Correspondencia
        if len(data) == 4:
            n_samples = data['labels'].shape[0]
            if global_samples_count is None:
                global_samples_count = n_samples
            consistency_check = all(data[k].shape[0] == n_samples for k in data)
            if not consistency_check:
                print("[ERROR] Inconsistencia en el número de muestras entre tensores.")
            elif n_samples != global_samples_count:
                print(f"[ERROR] Inconsistencia entre épocas. Esperado {global_samples_count}, encontrado {n_samples}.")
            else:
                print(f"[OK] Correspondencia verificada para {n_samples} muestras.")
            # 4. Balance de Clases
            labels_np = data['labels'].numpy()
            classes, counts = np.unique(labels_np, return_counts=True)
            print("\\n--- Balance de Clases ---")
            for c, count in zip(classes, counts):
                print(f"Clase {c}: {count} muestras ({count/n_samples*100:.2f}%)")
            # 5, 6, 7 y 9. Estadísticas e Integridad
            for key, tensor in data.items():
                has_nan = torch.isnan(tensor).any().item()
                has_inf = torch.isinf(tensor).any().item()
                print(f"\\n--- Análisis de {key} ---")
                print(f"NaN: {has_nan}, Inf: {has_inf}")
                if key != 'labels' and tensor.is_floating_point():
                    print(f"Min: {tensor.min().item():.4f}, Max: {tensor.max().item():.4f}, "
                            f"Mean: {tensor.mean().item():.4f}, Std: {tensor.std().item():.4f}")
                if key == 'mid_hiddens':
                    print(f"Dimensiones de embedding: {tensor.shape[1:]}")
                if key == 'attn_maps':
                    print(f"Resolución de atención: {tensor.shape[2:]}")
                    if tensor.max() > 1.0 or tensor.min() < 0.0:
                        print("[AVISO] Los mapas de atención requieren normalización (Min-Max) para visualización.")
                if key == 'generation_progress':
                    print(f"Pasos de integración: {tensor.shape[1]}, Resolución de imagen: {tensor.shape[-2:]}")
            # 8. Probability Path (Exportar ejemplo visual)
            if 'generation_progress' in data:
                print("\\nGenerando visualización de prueba del Probability Path (Muestra 0)...")
                sample_path = data['generation_progress'][0] # Shape esperada: [Pasos, C, H, W]
                num_steps = sample_path.shape[0]
                fig, axes = plt.subplots(1, min(10, num_steps), figsize=(20, 2))
                step_indices = np.linspace(0, num_steps - 1, min(10, num_steps), dtype=int)
                for idx, ax in zip(step_indices, axes):
                    img = sample_path[idx].permute(1, 2, 0).numpy()
                    img = (img - img.min()) / (img.max() - img.min() + 1e-8) # Normalización básica para plot
                    ax.imshow(img)
                    ax.set_title(f"Step {idx}")
                    ax.axis('off')
                plt.savefig(f"prob_path_test_{epoch}.png")
                plt.close()

# Ejecución (ajusta la ruta a tu directorio de datos)
audit_generative_data("./projections")