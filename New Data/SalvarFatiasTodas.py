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

# Verifica se um pedaço é o único com pixels `1` e ajusta se necessário
def adjust_unique_lesion_pieces_with_neighbors(subimages, current_index, total_slices, lesion_data):
    """
    Ajusta os pedaços da máscara para garantir que nenhum pedaço seja o único com pixels `1`
    ao analisar os slices anteriores e posteriores.
    :param subimages: Lista de subimagens (máscaras divididas em 8 pedaços) do slice atual.
    :param current_index: Índice do slice atual.
    :param total_slices: Número total de slices disponíveis.
    :param lesion_data: Dados completos da máscara de lesão.
    :return: Lista de subimagens ajustadas.
    """
    # Contar o número de pedaços com pixels `1` no slice atual
    has_lesion_current = [np.any(piece == 1) for piece, _ in subimages]

    # Verificar slices anteriores e posteriores
    has_lesion_before = 0
    has_lesion_after = 0
    if current_index > 0:  # Slice anterior
        has_lesion_before = [np.any(lesion_data[:, :, current_index - 1] == 1)]
    if current_index < total_slices - 1:  # Slice posterior
        has_lesion_after = [np.any(lesion_data[:, :, current_index + 1] == 1)]

    # Atualizar peças únicas com pixels `1` no slice atual
    total_lesion_pieces_current = sum(has_lesion_current) + has_lesion_before + has_lesion_after

    if sum(has_lesion_current) == 1 and total_lesion_pieces_current <= 1: # Se essa condição for satisfeita então o recorte atual com label 1 é um recorte isolado (sem lesão em cima e em baixo dele)
        unique_piece_index = has_lesion_current.index(True)
        subimages[unique_piece_index] = (np.zeros_like(subimages[unique_piece_index][0]), subimages[unique_piece_index][1]) # Zero a mascara lesionada isolada

    return subimages

imagens = "Patients_Displasya"
mascara = "Mascaras"

excluded_patients = ["sub-54K08", "sub-87G01", "sub-89A03", "sub-90K10"]

for img, mask in zip([f for f in os.listdir(imagens) if f.endswith(('.nii', '.nii.gz'))], [f for f in os.listdir(mascara) if f.endswith(('.nrrd', '.nii', '.nii.gz'))]):    
    #print(f"{img}, {mask}")
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
        lesion_slice_data = lesion_data[:, :, slice_idx]
        lesion_slice_data = np.where(lesion_slice_data>0.9, 1, 0)

        output_dir_lesion_slice = os.path.join(output_dir_lesion, f"Slice_{slice_idx:03}.nii.gz")
        
        # Pega a fatia axial atual
        slice_data = data[:, :, slice_idx]            

        output_dir_slice = os.path.join(output_dir, f"Slice{slice_idx:03}.nii.gz")
        
        processed_slices += 1

        # Converter o array numpy para um objeto NIfTI
        subimage_nii = nib.Nifti1Image(lesion_slice_data, affine=np.eye(4), dtype=np.int64)
        
        # Salvar o arquivo NIfTI
        if (lesion_slice_data.size>0 and lesion_slice_data is not None):
            nib.save(subimage_nii, output_dir_lesion_slice)

        # Converter o array numpy para um objeto NIfTI
        subimage_nii = nib.Nifti1Image(slice_data, affine=np.eye(4))
        
        # Salvar o arquivo NIfTI
        if (slice_data.size>0 and slice_data is not None):
            nib.save(subimage_nii, output_dir_slice)
    
    print(f"Total de fatias processadas do paciente {img.split('_')[0]}: {processed_slices}")