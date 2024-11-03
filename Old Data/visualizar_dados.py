import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

def visualize_nii_gz(file_path, patient, patch):
    # Carregar a imagem NIfTI
    nifti_img = nib.load(file_path)
    
    # Obter os dados da imagem como um array numpy
    data = nifti_img.get_fdata()
    np.set_printoptions(threshold=np.inf)
    print(data)
    
    # Exibir a imagem com o colormap viridis
    plt.figure(figsize=(6, 6))
    plt.imshow(data, cmap='gray')
    plt.title(f'Paciente {patient} | Patch {patch}')
    plt.show()

def find_image_paths(patch, type, path, file):
    if (file == "flair"):
        possible_patterns = [
            f'recortes_flair_50x50/sub-00H10_{patch}_{type}_flair.nii.gz',
            f'recortes_flair_50x50/sub-02A13_{patch}_{type}_flair.nii.gz',
            f'recortes_flair_50x50/sub-03C08_{patch}_{type}_flair.nii.gz',
            f'recortes_flair_50x50/sub-06C09_{patch}_{type}_flair.nii.gz',
            f'recortes_flair_50x50/sub-14F04_{patch}_{type}_flair.nii.gz',
            f'recortes_flair_50x50/sub-16E03_{patch}_{type}_flair.nii.gz',
            f'recortes_flair_50x50/sub-16G09_{patch}_{type}_flair.nii.gz',
            f'recortes_flair_50x50/sub-16I12_{patch}_{type}_flair.nii.gz',
            f'recortes_flair_50x50/sub-19F09_{patch}_{type}_flair.nii.gz',
            f'recortes_flair_50x50/sub-22F14_{patch}_{type}_flair.nii.gz',
            f'recortes_flair_50x50/sub-25B08_{patch}_{type}_flair.nii.gz',
            f'recortes_flair_50x50/sub-26B09_{patch}_{type}_flair.nii.gz',
            f'recortes_flair_50x50/sub-29D03_{patch}_{type}_flair.nii.gz',
            f'recortes_flair_50x50/sub-31F07_{patch}_{type}_flair.nii.gz',
            f'recortes_flair_50x50/sub-35E12_{patch}_{type}_flair.nii.gz',
            f'recortes_flair_50x50/sub-36K02_{patch}_{type}_flair.nii.gz',
            f'recortes_flair_50x50/sub-41D08_{patch}_{type}_flair.nii.gz',
            f'recortes_flair_50x50/sub-42B05_{patch}_{type}_flair.nii.gz',
            f'recortes_flair_50x50/sub-42K06_{patch}_{type}_flair.nii.gz',
            f'recortes_flair_50x50/sub-44H05_{patch}_{type}_flair.nii.gz',
            f'recortes_flair_50x50/sub-51C05_{patch}_{type}_flair.nii.gz',
            f'recortes_flair_50x50/sub-56E13_{patch}_{type}_flair.nii.gz',
            f'recortes_flair_50x50/sub-57D04_{patch}_{type}_flair.nii.gz',
            f'recortes_flair_50x50/sub-59E09_{patch}_{type}_flair.nii.gz',
            f'recortes_flair_50x50/sub-59G00_{patch}_{type}_flair.nii.gz',
            f'recortes_flair_50x50/sub-60G06_{patch}_{type}_flair.nii.gz',
            f'recortes_flair_50x50/sub-60G13_{patch}_{type}_flair.nii.gz',
            f'recortes_flair_50x50/sub-60K04_{patch}_{type}_flair.nii.gz',
            f'recortes_flair_50x50/sub-71C07_{patch}_{type}_flair.nii.gz',
            f'recortes_flair_50x50/sub-72I02_{patch}_{type}_flair.nii.gz',
            f'recortes_flair_50x50/sub-72K02_{patch}_{type}_flair.nii.gz',
            f'recortes_flair_50x50/sub-76E02_{patch}_{type}_flair.nii.gz',
            f'recortes_flair_50x50/sub-76J09_{patch}_{type}_flair.nii.gz',
            f'recortes_flair_50x50/sub-79H07_{patch}_{type}_flair.nii.gz',
            f'recortes_flair_50x50/sub-83K08_{patch}_{type}_flair.nii.gz',
            f'recortes_flair_50x50/sub-85I05_{patch}_{type}_flair.nii.gz',
            f'recortes_flair_50x50/sub-86B13_{patch}_{type}_flair.nii.gz'
        ]
    elif (file == "T132"):
        possible_patterns = [
            f'recortes_T1_32x32/sub-00H10_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_32x32/sub-02A13_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_32x32/sub-03C08_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_32x32/sub-06C09_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_32x32/sub-14F04_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_32x32/sub-16E03_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_32x32/sub-16G09_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_32x32/sub-16I12_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_32x32/sub-19F09_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_32x32/sub-22F14_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_32x32/sub-25B08_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_32x32/sub-26B09_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_32x32/sub-29D03_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_32x32/sub-31F07_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_32x32/sub-35E12_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_32x32/sub-36K02_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_32x32/sub-41D08_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_32x32/sub-42B05_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_32x32/sub-42K06_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_32x32/sub-44H05_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_32x32/sub-51C05_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_32x32/sub-56E13_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_32x32/sub-57D04_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_32x32/sub-59E09_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_32x32/sub-59G00_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_32x32/sub-60G06_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_32x32/sub-60G13_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_32x32/sub-60K04_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_32x32/sub-71C07_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_32x32/sub-72I02_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_32x32/sub-72K02_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_32x32/sub-76E02_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_32x32/sub-76J09_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_32x32/sub-79H07_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_32x32/sub-83K08_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_32x32/sub-85I05_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_32x32/sub-86B13_{patch}_{type}_T1.nii.gz'
        ]
    elif (file == "T150"):
        possible_patterns = [
            f'recortes_T1_50x50/sub-00H10_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_50x50/sub-02A13_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_50x50/sub-03C08_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_50x50/sub-06C09_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_50x50/sub-14F04_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_50x50/sub-16E03_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_50x50/sub-16G09_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_50x50/sub-16I12_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_50x50/sub-19F09_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_50x50/sub-22F14_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_50x50/sub-25B08_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_50x50/sub-26B09_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_50x50/sub-29D03_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_50x50/sub-31F07_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_50x50/sub-35E12_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_50x50/sub-36K02_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_50x50/sub-41D08_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_50x50/sub-42B05_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_50x50/sub-42K06_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_50x50/sub-44H05_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_50x50/sub-51C05_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_50x50/sub-56E13_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_50x50/sub-57D04_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_50x50/sub-59E09_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_50x50/sub-59G00_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_50x50/sub-60G06_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_50x50/sub-60G13_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_50x50/sub-60K04_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_50x50/sub-71C07_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_50x50/sub-72I02_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_50x50/sub-72K02_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_50x50/sub-76E02_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_50x50/sub-76J09_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_50x50/sub-79H07_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_50x50/sub-83K08_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_50x50/sub-85I05_{patch}_{type}_T1.nii.gz',
            f'recortes_T1_50x50/sub-86B13_{patch}_{type}_T1.nii.gz'
        ]
        
    patient = 0
    for pattern in possible_patterns:
        patient+=1
        if os.path.exists(pattern):
            path = pattern
            break

    return path, patient

# Input de qual patch do paciente deve ser analisado
file = "T150"
if (file == "T150"):
    patch = int(input("Digite o numero do recorte que deseja analisar: De 1 a 66552 \n"))
    while (patch<1 or patch>66552):
        patch = int(input("Digite um numero de recorte valido: De 1 a 66552 \n"))
elif (file == "T132"):
    patch = int(input("Digite o numero do recorte que deseja analisar: De 1 a 97572 \n"))
    while (patch<1 or patch>97572):
        patch = int(input("Digite um numero de recorte valido: De 1 a 97572 \n"))
elif (file == "flair"):
    patch = int(input("Digite o numero do recorte que deseja analisar: De 1 a 35196 \n"))
    while (patch<1 or patch>35196):
        patch = int(input("Digite um numero de recorte valido: De 1 a 35196 \n"))
else:
    exit(1)
    
# Caminho para o arquivo .nii.gz gerado
path_crop, patient_crop = find_image_paths(patch,"recorte", path = None, file=file)
path_mask, patient_mask = find_image_paths(patch,"mascara", path = None, file=file)

path = path_crop
patient = patient_crop

# Visualizar o patch no primeiro canal
visualize_nii_gz(f"{path}", patient, patch)