import nibabel as nib
import os
import numpy as np
import matplotlib.pyplot as plt

paciente_original = "Total de Pacientes"
mascara = "Mascaras"

for pac, masc in zip(os.listdir(paciente_original), os.listdir(mascara)):
    caminho_pac_orig = os.path.join(paciente_original, pac)
    caminho_pac_masc = os.path.join(mascara, masc)
    
    patient = nib.load(caminho_pac_orig).get_fdata()
    mask = nib.load(caminho_pac_masc).get_fdata()
    
    # Plotting the patient and the mask side by side 
    slice_idx = patient.shape[2] // 2 
    fig, axes = plt.subplots(1, 2, figsize=(12, 6)) 
    
    axes[0].imshow(patient[:, :, slice_idx], cmap="gray") 
    axes[0].set_title("Imagem do Paciente") 
    axes[0].axis("off") 
    
    axes[1].imshow(mask[:, :, slice_idx], cmap="gray") 
    axes[1].set_title("Máscara Resultante") 
    axes[1].axis("off") 
    
    plt.show(block=False)
    plt.pause(1)
    plt.close()
