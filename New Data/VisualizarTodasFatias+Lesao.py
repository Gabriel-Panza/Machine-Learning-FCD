import os
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import nibabel as nib

# Função para carregar e plotar fatias com lesão
def plot_lesions_combined(image_path, lesion_mask_path, side_path_left, side_path_right):
    # Carregar a imagem NRRD
    img = sitk.ReadImage(image_path)
    data = sitk.GetArrayFromImage(img)

    # Carregar a máscara de lesão NRRD
    lesion_mask = sitk.ReadImage(lesion_mask_path)
    lesion_data = sitk.GetArrayFromImage(lesion_mask)

    # Loop pelos arquivos nas pastas
    cont = 1
    for file_left, file_right in zip(os.listdir(side_path_left), os.listdir(side_path_right)):
        if ((cont + 2) % 4 != 0):
            file_path_left = f"{side_path_left}/{file_left}"
            file_path_right = f"{side_path_right}/{file_right}"

            # Carregar dados dos arquivos .nii.gz do lado esquerdo e direito
            img_left = nib.load(file_path_left)
            data_left = img_left.get_fdata()
            img_right = nib.load(file_path_right)
            data_right = img_right.get_fdata()

            lesion_slice = lesion_data[cont, :, :]

            # Verificar se a fatia contém lesão
            has_lesion = np.any(lesion_slice == 1)

            # Rotacionar as fatias
            rotated_slice_left = np.rot90(data_left, k=-1)
            rotated_slice_right = np.rot90(data_right, k=-1)

            # Configurar a figura e os subplots
            fig, axes = plt.subplots(1, 3, figsize=(12, 5))

            # Plotar fatia esquerda com sobreposição de lesão, se houver
            axes[0].imshow(rotated_slice_left, cmap='gray', origin='lower')
            axes[0].set_title(f"Lado Esquerdo")
            axes[0].axis('off')

            # Plotar fatia direita
            axes[1].imshow(rotated_slice_right, cmap='gray', origin='lower')
            axes[1].set_title("Lado Direito")
            axes[1].axis('off')

            axes[2].imshow(np.ma.masked_where(lesion_slice == 0, lesion_slice), cmap="gray", alpha=0.5, origin="lower")
            axes[2].set_title(f"{'Lesion' if has_lesion else 'No Lesion'}")
            axes[2].axis('off')
        
            view = cont // 4
            view_old = view
            if (cont % 4 == 0):
                view -= 1
            plt.suptitle(f"Fatias {view + 1} - Rotacionada e Dividida", fontsize=16)
            plt.show(block=False)
            plt.pause(1)
            plt.close()
            view = view_old

        cont += 1

# Diretório base
base_dir = "Contralateral"
for subject in os.listdir(base_dir):
    subject_path = os.path.join(base_dir, subject)
    if os.path.isdir(subject_path):
        left_side = "left"
        right_side = "right"
        side_path_left = os.path.join(subject_path, left_side)
        side_path_right = os.path.join(subject_path, right_side)
        if os.path.isdir(side_path_left) and os.path.isdir(side_path_right):
            # Caminhos das imagens de entrada NRRD e máscaras de lesão
            image_path = f"Patients_Displasya/{subject}/ses-01/anat/{subject} Displasia.seg.nrrd"
            lesion_mask_path = f"Patients_Displasya/{subject}/ses-01/anat/{subject} Label FLAIR Questionvel.seg.nrrd"

            # Executa a função de plotagem combinada
            plot_lesions_combined(image_path, lesion_mask_path, side_path_left, side_path_right)