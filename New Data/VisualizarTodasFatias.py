#Codigo para plotar todas fatias individualmente (pressione CTRL C repetidamente para parar)
import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

base_dir = "Contralateral"
for subject in os.listdir(base_dir):
    subject_path = os.path.join(base_dir, subject)
    if os.path.isdir(subject_path):
        left_side = "left"
        right_side = "right"
        side_path_left = os.path.join(subject_path, left_side)
        side_path_right = os.path.join(subject_path, right_side)
        if os.path.isdir(side_path_left):
            cont = 1
            for file_left, file_right in zip(os.listdir(side_path_left),os.listdir(side_path_right)):
                if ((cont+2)%4 != 0):
                    file_path_left = f"{side_path_left}/{file_left}"
                    file_path_right = f"{side_path_right}/{file_right}"
                    
                    img_left = nib.load(file_path_left)
                    data_left = img_left.get_fdata()
                    img_right = nib.load(file_path_right)
                    data_right = img_right.get_fdata()
                    
                    # Rotacionar a fatia em -90 graus
                    rotated_slice_left = np.rot90(data_left, k=-1)
                    rotated_slice_right = np.rot90(data_right, k=-1)

                    # Configurar a figura e os subplots
                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

                    # Metade esquerda
                    axes[0].imshow(rotated_slice_left, cmap='gray')
                    axes[0].set_title("Lado Esquerdo")
                    axes[0].axis('off')

                    # Metade direita
                    axes[1].imshow(rotated_slice_right, cmap='gray')
                    axes[1].set_title("Lado Direito")
                    axes[1].axis('off')

                    view = cont//4
                    view_old = view
                    if (cont%4==0):
                        view -= 1
                    plt.suptitle(f"Fatias {view + 1} - Rotacionada e Dividida", fontsize=16)
                    plt.show(block=False)
                    plt.pause(1)  # Pausa por 1 segundo para cada fatia
                    plt.close()
                    view = view_old
                    
                cont+=1