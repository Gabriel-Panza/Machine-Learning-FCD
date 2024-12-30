import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

def calculate_label(subimage, threshold=0.009):
    """
    Determina o label da subimagem com base no percentual de fundo não-preto.
    :param subimage: Array da subimagem.
    :param threshold: Percentual mínimo de fundo não-preto para considerar como label 1.
    :return: int indicando o label.
    """
    # Total de pixels na subimagem
    total_pixels = subimage.size
    # Número de pixels não-preto
    non_zero_pixels = np.count_nonzero(subimage)
    # Proporção de pixels não-preto
    non_black_ratio = non_zero_pixels / total_pixels if total_pixels > 0 else 0
    
    # Verifica se há lesão e se o fundo não-preto é maior que o limiar
    if np.any(subimage == 1) and non_black_ratio >= threshold:
        return 1
    else:
        return 0

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
            
            # Configurar a figura e os subplots
            _, axes = plt.subplots(1, 4, figsize=(16, 5))

            axes[0].imshow(data_left, cmap='gray', origin='lower')
            axes[0].set_title(f"Lado Esquerdo")
            axes[0].axis('off')

            axes[1].imshow(data_right, cmap='gray', origin='lower')
            axes[1].set_title("Lado Direito Flippado")
            axes[1].axis('off')

            axes[2].imshow(lesion_data_left, cmap="gray", alpha=0.5, origin="lower")
            axes[2].set_title(f"{'Lesion' if calculate_label(lesion_data_left) else 'No Lesion'}")
            axes[2].axis('off')
        
            axes[3].imshow(lesion_data_right, cmap="gray", alpha=0.5, origin="lower")
            axes[3].set_title(f"{'Lesion' if calculate_label(lesion_data_right) else 'No Lesion'}")
            axes[3].axis('off')
            
            view = cont // 8
            if (cont % 8 == 0):
                view -= 1
            plt.suptitle(f"Fatias {view + 1} - Rotacionada e Dividida", fontsize=16)
            plt.show(block=False)
            plt.pause(3)
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