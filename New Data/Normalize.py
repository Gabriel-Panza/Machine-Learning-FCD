import nibabel as nib
import numpy as np
import os

# Normalization -> valores de voxels entre 0 e 1
def normalize_image_min(image_data): 
    min_val = np.min(image_data)
    max_val = np.max(image_data)
    normalized_data = (image_data - min_val) / (max_val - min_val)
    return normalized_data

def get_info_data(data):
    # Calcular percentis
    percentil_inferior = np.percentile(data, 1)
    percentil_superior = np.percentile(data, 99)
    print(f"min: {np.min(data)}")
    print(f"1º percentil: {percentil_inferior}")
    print(f"99º percentil: {percentil_superior}")
    print(f"max: {np.max(data)}")

    # Estatísticas antes
    print(f"Média: {np.mean(data)}, Desvio padrão: {np.std(data)}")

folder = 'Patients_Displasya'
output_folder = f"{folder}_Normalized"
os.makedirs(output_folder, exist_ok=True)

for item in os.listdir(folder):
    if item.endswith(('.nii.gz', '.nii')):
        file = os.path.join(folder, item)
        output_path= os.path.join(output_folder, item)

        img = nib.load(file)
        img_data = img.get_fdata()

        print("\n\nBEFORE NORMALIZATION")
        get_info_data(img_data)

        new_img_data = normalize_image_min(img_data)

        print("\nAFTER NORMALIZATION")
        get_info_data(new_img_data)

        new_img = nib.Nifti1Image(new_img_data, img.affine, img.header)

        # Salvar a nova imagem no caminho de saída
        nib.save(new_img, output_path)