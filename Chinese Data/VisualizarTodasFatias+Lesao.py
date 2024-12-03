import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

# Função para carregar e plotar fatias com lesão
def plot_lesions_combined(lesion_mask_path_left, lesion_mask_path_right, side_path_left, side_path_right):
    # Loop pelos arquivos nas pastas
    cont = 1
    for file_left, file_right, file_lesion_left, file_lesion_right in zip(os.listdir(side_path_left), os.listdir(side_path_right), os.listdir(lesion_mask_path_left), os.listdir(lesion_mask_path_right)):
        path_file_left = os.path.join(side_path_left, file_left)
        path_file_right = os.path.join(side_path_right, file_right)
        path_file_left_lesion = os.path.join(lesion_mask_path_left, file_lesion_left)
        path_file_right_lesion = os.path.join(lesion_mask_path_right, file_lesion_right)
        for slice_left, slice_right, slice_lesion_left, slice_lesion_right in zip(os.listdir(path_file_left), os.listdir(path_file_right), os.listdir(path_file_left_lesion), os.listdir(path_file_right_lesion)):
            file_path_left = f"{path_file_left}/{slice_left}"
            file_path_right = f"{path_file_right}/{slice_right}"
            file_path_left_lesion = f"{path_file_left_lesion}/{slice_lesion_left}"
            file_path_right_lesion = f"{path_file_right_lesion}/{slice_lesion_right}"

            # Carregar dados dos arquivos .nii.gz do lado esquerdo e direito
            img_left = nib.load(file_path_left)
            data_left = img_left.get_fdata()
            img_right = nib.load(file_path_right)
            data_right = img_right.get_fdata()
            lesion_left = nib.load(file_path_left_lesion)
            lesion_data_left = lesion_left.get_fdata()
            lesion_right = nib.load(file_path_right_lesion)
            lesion_data_right = lesion_right.get_fdata()
            
            # Rotacionar as fatias
            rotated_slice_left = np.rot90(data_left, k=-1)
            rotated_slice_right = np.rot90(data_right, k=-1)
            rotated_lesion_slice_left = np.rot90(lesion_data_left, k=-1)
            rotated_lesion_slice_right = np.rot90(lesion_data_right, k=-1)

            # Configurar a figura e os subplots
            fig, axes = plt.subplots(1, 4, figsize=(16, 5))

            # Plotar fatia esquerda com sobreposição de lesão, se houver
            axes[0].imshow(rotated_slice_left, cmap='gray', origin='lower')
            axes[0].set_title(f"Lado Esquerdo\n{slice_left}")
            axes[0].axis('off')

            # Plotar fatia direita
            axes[1].imshow(rotated_slice_right, cmap='gray', origin='lower')
            axes[1].set_title(f"Lado Direito\n{slice_right}")
            axes[1].axis('off')

            # Lesões lado esquerdo
            axes[2].imshow(rotated_lesion_slice_left, cmap="gray", alpha=0.5, origin="lower")
            axes[2].set_title(f"{'Lesion' if np.any(rotated_lesion_slice_left == 1) else 'No Lesion'}\n{slice_lesion_left}")
            axes[2].axis('off')

            # Lesões lado direito
            axes[3].imshow(rotated_lesion_slice_right, cmap="gray", alpha=0.5, origin="lower")
            axes[3].set_title(f"{'Lesion' if np.any(rotated_lesion_slice_right == 1) else 'No Lesion'}\n{slice_lesion_right}")
            axes[3].axis('off')
            
            view = cont // 4
            if (cont % 4 == 0):
                view -= 1
            plt.suptitle(f"Fatias {view + 1} - Rotacionada e Dividida", fontsize=16)
            plt.show(block=False)
            plt.pause(1)
            plt.close()

            cont += 1


# Diretório base
base_dir = "Contralateral"
for subject in os.listdir(base_dir):
    subject_path = os.path.join(base_dir, subject)
    if os.path.isdir(subject_path):
        left_side = "left"
        right_side = "right"
        lesion_left_side = "lesion_left"
        lesion_right_side = "lesion_right"
        
        side_path_left = os.path.join(subject_path, left_side)
        side_path_right = os.path.join(subject_path, right_side)
        side_path_lesion_left = os.path.join(subject_path, lesion_left_side)
        side_path_lesion_right = os.path.join(subject_path, lesion_right_side)
        
        if os.path.isdir(side_path_left) and os.path.isdir(side_path_right) and os.path.isdir(side_path_lesion_left) and os.path.isdir(side_path_lesion_right):
            # Executa a função de plotagem combinada
            plot_lesions_combined(side_path_lesion_left, side_path_lesion_right, side_path_left, side_path_right)