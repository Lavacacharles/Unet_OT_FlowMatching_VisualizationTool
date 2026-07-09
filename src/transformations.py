import os
import torch
import json
import numpy as np
import matplotlib.pyplot as plt

def build_flat_dataset(base_path, out_dir):
    epochs = [250, 500, 750, 1000, 1250, 1500]
    os.makedirs(out_dir, exist_ok=True)
    
    records = []
    
    for ep in epochs:
        ep_str = f"epoch_{ep}"
        ep_path = os.path.join(base_path, ep_str)
        
        labels = torch.tensor(torch.load(os.path.join(ep_path, 'labels.pth'), map_location='cpu'))
        mid_hiddens = torch.tensor(torch.load(os.path.join(ep_path, 'mid_hiddens.pth'), map_location='cpu'))
        attn_maps = torch.tensor(torch.load(os.path.join(ep_path, 'attn_maps.pth'), map_location='cpu'))
        gen_prog = torch.tensor(torch.load(os.path.join(ep_path, 'generation_progress.pth'), map_location='cpu'))
        
        num_samples = labels.shape[0]
        num_steps = mid_hiddens.shape[0]
        
        for sample_id in range(num_samples):
            label = labels[sample_id].item()
            
            for step in range(num_steps):
                # Extraer coordenadas
                x, y = mid_hiddens[step, sample_id].tolist()
                
                # Rutas para guardar imágenes
                img_dir = os.path.join(out_dir, "images", ep_str)
                attn_dir = os.path.join(out_dir, "attention", ep_str)
                os.makedirs(img_dir, exist_ok=True)
                os.makedirs(attn_dir, exist_ok=True)
                
                img_path = os.path.join(img_dir, f"sample_{sample_id}_step_{step}.png")
                attn_path = os.path.join(attn_dir, f"sample_{sample_id}_step_{step}.png")
                
                # Guardar imagen (Probability Path)
                img_tensor = gen_prog[sample_id, step].permute(1, 2, 0).numpy()
                img_norm = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min() + 1e-8)
                plt.imsave(img_path, img_norm)
                
                # Guardar mapa de atención
                attn_tensor = attn_maps[sample_id, step].numpy()
                plt.imsave(attn_path, attn_tensor, cmap='viridis')
                
                # Crear registro plano
                records.append({
                    "epoch": ep,
                    "generation_step": step,
                    "sample_id": sample_id,
                    "label": label,
                    "x": x,
                    "y": y,
                    "image_path": img_path,
                    "attention_path": attn_path
                })
                
    with open(os.path.join(out_dir, 'dataset.json'), 'w') as f:
        json.dump(records, f)

# Ejecutar el preprocesador
build_flat_dataset("./projections", "viztool/assets")