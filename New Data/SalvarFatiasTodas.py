import nibabel as nib
import numpy as np
import os
import nrrd

def calculate_label(subimage, threshold=0.05):
    """
    Determina o label da subimagem com base no percentual de fundo não-preto.
    :param subimage: Array da subimagem.
    :param threshold: Percentual mínimo de fundo não-preto para considerar como label 1.
    :return: String indicando o label.
    """
    # Total de pixels na subimagem
    total_pixels = subimage.size
    # Número de pixels não-preto
    non_zero_pixels = np.count_nonzero(subimage)
    # Proporção de pixels não-preto
    non_black_ratio = non_zero_pixels / total_pixels if total_pixels > 0 else 0
    
    # Verifica se há lesão e se o fundo não-preto é maior que o limiar
    if np.any(subimage == 1) and non_black_ratio >= threshold:
        return "label1"
    else:
        return "label0"

imagens = "Patients_Displasya"
mascara = "Mascaras"

excluded_patients = ["sub-54K08", "sub-87G01", "sub-89A03", "sub-90K10"]

for img, mask in zip([f for f in os.listdir(imagens) if f.endswith(('.nii', '.nii.gz'))], [f for f in os.listdir(mascara) if f.endswith(('.nrrd', '.nii', '.nii.gz'))]):    
    if img.split('_')[0] in excluded_patients:
        continue
    data = nib.load(os.path.join(imagens, img)).get_fdata()
    lesion_data, _ = nrrd.read(os.path.join(mascara, mask))
    
    # Rotacionar as imagens e as máscaras 180 graus
    data = np.rot90(data, k=3)  # Rotaciona a imagem 90 graus
    lesion_data = np.rot90(lesion_data, k=3)  # Rotaciona a máscara 90 graus
    
    # Verifica o formato das imagens
    print(data[2].shape)
    print(lesion_data[2].shape)

    if (lesion_data.shape[2]>data.shape[2]):
        continue

    # Contador para acompanhar quantas fatias foram processadas
    processed_slices = 0

    # Diretório de saída para salvar as fatias
    output_dir = f"Fatias/{img.split('_')[0]}"
    output_dir_lesion = f"Mask_Fatias/{mask.split(' ')[0]}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_lesion, exist_ok=True)
    
    # Loop para cada fatia axial
    for slice_idx in range(lesion_data.shape[2]):
        # Pega a lesão da fatia axial atual
        lesion_slice_data = lesion_data[16:233-17, 18:197-19, slice_idx]
        lesion_slice_data = np.where(lesion_slice_data>0.9, 1, 0)

        output_dir_lesion_slice = os.path.join(output_dir_lesion, f"Slice{slice_idx}.nii.gz")
        
        # Pega a fatia axial atual
        slice_data = data[16:233-17, 18:197-19, slice_idx]            

        # Total de pixels na subimagem
        total_pixels = slice_data.size
        # Número de pixels não-preto
        non_zero_pixels = np.count_nonzero(slice_data)
        # Proporção de pixels não-preto
        non_black_ratio = non_zero_pixels / total_pixels if total_pixels > 0 else 0
        
        if non_black_ratio >= 0.08:
            output_dir_slice = os.path.join(output_dir, f"Slice{slice_idx}.nii.gz")
            
            processed_slices += 1
            
            # Converter o array numpy para um objeto NIfTI
            subimage_nii = nib.Nifti1Image(lesion_slice_data, affine=np.eye(4))
            
            # Salvar o arquivo NIfTI
            if (lesion_slice_data.size>0 and lesion_slice_data is not None):
                nib.save(subimage_nii, output_dir_lesion_slice)

            # Converter o array numpy para um objeto NIfTI
            subimage_nii = nib.Nifti1Image(slice_data, affine=np.eye(4))
            
            # Salvar o arquivo NIfTI
            if (slice_data.size>0 and slice_data is not None):
                nib.save(subimage_nii, output_dir_slice)
    
    print(f"Total de fatias processadas do paciente {img.split('_')[0]}: {processed_slices}")