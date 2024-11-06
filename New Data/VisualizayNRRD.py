import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os

# Função para carregar e plotar fatias com lesão
def plot_lesions_nrrd(image_path, lesion_mask_path, output_dir):
    # Carregar a imagem NRRD
    img = sitk.ReadImage(image_path)
    data = sitk.GetArrayFromImage(img)

    # Carregar a máscara de lesão NRRD
    lesion_mask = sitk.ReadImage(lesion_mask_path)
    lesion_data = sitk.GetArrayFromImage(lesion_mask)

    # Criar o diretório de saída
    os.makedirs(output_dir, exist_ok=True)

    # Loop para plotar cada fatia axial
    for slice_idx in range(data.shape[0]):  # data.shape[0] representa o número de fatias
        # Obter a fatia atual e a correspondente máscara de lesão
        slice_data = data[slice_idx, :, :]
        lesion_slice = lesion_data[slice_idx, :, :]

        # Verificar se a fatia contém lesão
        has_lesion = np.any(lesion_slice == 1)

        # Plotar a fatia e sobrepor a máscara de lesão
        plt.figure(figsize=(6, 6))
        plt.imshow(slice_data, cmap="gray", origin="lower")
        plt.imshow(np.ma.masked_where(lesion_slice == 0, lesion_slice), cmap="hot", alpha=0.5, origin="lower")
        plt.title(f"Slice {slice_idx} - {'Lesion' if has_lesion else 'No Lesion'}")

        # Salvar a imagem plotada
        output_path = os.path.join(output_dir, f"slice_{slice_idx}.png")
        plt.savefig(output_path)
        plt.close()

        print(f"Saved {output_path} - {'Lesion' if has_lesion else 'No Lesion'}")

# Exemplo de uso
image_path = "Patients_Displasya/sub-00H10/ses-01/anat/sub-00H10 Displasia.seg.nrrd"  
lesion_mask_path = "Patients_Displasya/sub-00H10/ses-01/anat/sub-00H10 Label FLAIR Questionvel.seg.nrrd"
output_dir = "output_slices_nrrd"

plot_lesions_nrrd(image_path, lesion_mask_path, output_dir)