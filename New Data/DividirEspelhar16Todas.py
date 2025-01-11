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

def divide_16_pieces(rotated_slice):
    # Dividir a fatia ajustada em esquerda e direita
    midpoint = rotated_slice.shape[1] // 2
    left_half = rotated_slice[:, 10:midpoint]
    right_half = rotated_slice[:, midpoint:midpoint*2 - 10]

    # Inverter horizontalmente o lado direito
    right_half_flipped = np.fliplr(right_half)

    # Dividir as metades esquerda e direita horizontalmente em duas partes
    horizontal_mid_left = left_half.shape[0] // 2
    horizontal_mid_right = right_half_flipped.shape[0] // 2

    top_left = left_half[24:horizontal_mid_left, :]
    top_right = left_half[horizontal_mid_left:horizontal_mid_left*2 -24, :]
    bottom_left = right_half_flipped[24:horizontal_mid_right, :]
    bottom_right = right_half_flipped[horizontal_mid_right:horizontal_mid_right*2 -24, :]

    # Dividir cada quadrante em 4 subquadrantes (totalizando 16 divisões)
    def split_quadrant(quadrant):
        vertical_mid = quadrant.shape[0] // 2
        horizontal_mid = quadrant.shape[1] // 2
        
        top_left = quadrant[:vertical_mid+4, :horizontal_mid+6]
        top_right = quadrant[:vertical_mid+4, horizontal_mid-6:]
        bottom_left = quadrant[vertical_mid-4:, :horizontal_mid+6]
        bottom_right = quadrant[vertical_mid-4:, horizontal_mid-6:]
        
        return top_left, top_right, bottom_left, bottom_right
    
    left_left_top_left, left_left_top_right, left_left_bottom_left, left_left_bottom_right = split_quadrant(top_left)
    left_right_top_left, left_right_top_right, left_right_bottom_left, left_right_bottom_right = split_quadrant(top_right)
    right_left_top_left, right_left_top_right, right_left_bottom_left, right_left_bottom_right = split_quadrant(bottom_left)
    right_right_top_left, right_right_top_right, right_right_bottom_left, right_right_bottom_right = split_quadrant(bottom_right)

    return (left_left_top_left, left_left_top_right, left_left_bottom_left, left_left_bottom_right,
            left_right_top_left, left_right_top_right, left_right_bottom_left, left_right_bottom_right,
            right_left_top_left, right_left_top_right, right_left_bottom_left, right_left_bottom_right,
            right_right_top_left, right_right_top_right, right_right_bottom_left, right_right_bottom_right)

       
imagens = "Patients_Displasya"
mascara = "Mascaras"

total_label1 = []
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
    output_dir_left = os.path.join(f"Contralateral/{img.split('_')[0]}", "left")
    output_dir_right = os.path.join(f"Contralateral/{img.split('_')[0]}", "right")
    output_dir_lesion_left = os.path.join(f"Contralateral/{mask.split(' ')[0]}", "lesion_left")
    output_dir_lesion_right = os.path.join(f"Contralateral/{mask.split(' ')[0]}", "lesion_right")
    os.makedirs(output_dir_left, exist_ok=True)
    os.makedirs(output_dir_right, exist_ok=True)
    os.makedirs(output_dir_lesion_left, exist_ok=True)
    os.makedirs(output_dir_lesion_right, exist_ok=True)
    

    count_label1 = 0
    # Loop para cada fatia axial
    for slice_idx in range(lesion_data.shape[2]):
        # Pega a lesão da fatia axial atual
        lesion_slice_data = lesion_data[:, :, slice_idx]
        lesion_slice_data = np.where(lesion_slice_data>0.9, 1, 0)
        
        # Pega a fatia axial atual
        slice_data = data[:, :, slice_idx]            

        output_dir_left_slice = os.path.join(output_dir_left, f"Slice{slice_idx}/")
        output_dir_right_slice = os.path.join(output_dir_right, f"Slice{slice_idx}/")

        os.makedirs(output_dir_left_slice, exist_ok=True)
        os.makedirs(output_dir_right_slice, exist_ok=True)
        
        processed_slices += 1

        output_dir_lesion_left_slice = os.path.join(output_dir_lesion_left, f"Slice{slice_idx}")
        output_dir_lesion_right_slice = os.path.join(output_dir_lesion_right, f"Slice{slice_idx}")
        
        os.makedirs(output_dir_lesion_left_slice, exist_ok=True)
        os.makedirs(output_dir_lesion_right_slice, exist_ok=True)
        
        left_top_top_left, left_top_top_right, left_top_bottom_left, left_top_bottom_right, left_bottom_top_left, left_bottom_top_right, left_bottom_bottom_left, left_bottom_bottom_right, right_top_top_left, right_top_top_right, right_top_bottom_left, right_top_bottom_right, right_bottom_top_left, right_bottom_top_right, right_bottom_bottom_left, right_bottom_bottom_right = divide_16_pieces(lesion_slice_data)
        
        count_label1_anterior = count_label1
        for lesion_part in [left_top_top_left, left_top_top_right, left_top_bottom_left, left_top_bottom_right, left_bottom_top_left, left_bottom_top_right, left_bottom_bottom_left, left_bottom_bottom_right, right_top_top_left, right_top_top_right, right_top_bottom_left, right_top_bottom_right, right_bottom_top_left, right_bottom_top_right, right_bottom_bottom_left, right_bottom_bottom_right]:
            print(lesion_part.shape)
            if calculate_label(lesion_part) == "label1":
                count_label1 += 1
        count_label1_posterior = count_label1
        
        # Lista com todas as subimagens e identificações
        subimages_lesion = [
            (left_top_top_left, f"left_1_lesion_{calculate_label(left_top_top_left)}"),
            (right_top_top_right, f"right_1_lesion_{calculate_label(right_top_top_right)}"),
            (left_top_top_right, f"left_2_lesion_{calculate_label(left_top_top_right)}"),
            (right_top_top_left, f"right_2_lesion_{calculate_label(right_top_top_left)}"),
            (left_top_bottom_left, f"left_3_lesion_{calculate_label(left_top_bottom_left)}"),
            (right_top_bottom_right, f"right_3_lesion_{calculate_label(right_top_bottom_right)}"),
            (left_top_bottom_right, f"left_4_lesion_{calculate_label(left_top_bottom_right)}"),
            (right_top_bottom_left, f"right_4_lesion_{calculate_label(right_top_bottom_left)}"),
            (left_bottom_top_left, f"left_5_lesion_{calculate_label(left_bottom_top_left)}"),
            (right_bottom_top_right, f"right_5_lesion_{calculate_label(right_bottom_top_right)}"),
            (left_bottom_top_right, f"left_6_lesion_{calculate_label(left_bottom_top_right)}"),
            (right_bottom_top_left, f"right_6_lesion_{calculate_label(right_bottom_top_left)}"),
            (left_bottom_bottom_left, f"left_7_lesion_{calculate_label(left_bottom_bottom_left)}"),
            (right_bottom_bottom_right, f"right_7_lesion_{calculate_label(right_bottom_bottom_right)}"),
            (left_bottom_bottom_right, f"left_8_lesion_{calculate_label(left_bottom_bottom_right)}"),
            (right_bottom_bottom_left, f"right_8_lesion_{calculate_label(right_bottom_bottom_left)}")
        ]
        
        # Salvar cada subimagem como um arquivo NIfTI separado
        for subimage, position in subimages_lesion:
            # Definir o diretório de saída com base na posição
            if position.startswith("left"):
                output_path_lesion = os.path.join(output_dir_lesion_left_slice, f"{position}.nii.gz")
            else:
                output_path_lesion = os.path.join(output_dir_lesion_right_slice, f"{position}.nii.gz")

            # Converter o array numpy para um objeto NIfTI
            subimage_nii = nib.Nifti1Image(subimage, affine=np.eye(4))
            
            # Salvar o arquivo NIfTI
            if (subimage.size>0 and subimage is not None):
                nib.save(subimage_nii, output_path_lesion)

        left_top_top_left, left_top_top_right, left_top_bottom_left, left_top_bottom_right, left_bottom_top_left, left_bottom_top_right, left_bottom_bottom_left, left_bottom_bottom_right, right_top_top_left, right_top_top_right, right_top_bottom_left, right_top_bottom_right, right_bottom_top_left, right_bottom_top_right, right_bottom_bottom_left, right_bottom_bottom_right = divide_16_pieces(slice_data)

        for elem in [left_top_top_left, left_top_top_right, left_top_bottom_left, left_top_bottom_right, left_bottom_top_left, left_bottom_top_right, left_bottom_bottom_left, left_bottom_bottom_right, right_top_top_left, right_top_top_right, right_top_bottom_left, right_top_bottom_right, right_bottom_top_left, right_bottom_top_right, right_bottom_bottom_left, right_bottom_bottom_right]:
            print(elem.shape)
        
        # Lista com todas as subimagens e identificações
        subimages = [
            (left_top_top_left, f"left_1"),
            (right_top_top_right, f"right_1"),
            (left_top_top_right, f"left_2"),
            (right_top_top_left, f"right_2"),
            (left_top_bottom_left, f"left_3"),
            (right_top_bottom_right, f"right_3"),
            (left_top_bottom_right, f"left_4_"),
            (right_top_bottom_left, f"right_4"),
            (left_bottom_top_left, f"left_5"),
            (right_bottom_top_right, f"right_5"),
            (left_bottom_top_right, f"left_6"),
            (right_bottom_top_left, f"right_6"),
            (left_bottom_bottom_left, f"left_7"),
            (right_bottom_bottom_right, f"right_7"),
            (left_bottom_bottom_right, f"left_8"),
            (right_bottom_bottom_left, f"right_8")
        ]

        # Salvar cada subimagem como um arquivo NIfTI separado
        for subimage, position in subimages:            
            # Definir o diretório de saída com base na posição
            if position.startswith("left"):
                output_path = os.path.join(output_dir_left_slice, f"{position}.nii.gz")
            else:
                output_path = os.path.join(output_dir_right_slice, f"{position}.nii.gz")

            # Converter o array numpy para um objeto NIfTI
            subimage_nii = nib.Nifti1Image(subimage, affine=np.eye(4))
            
            # Salvar o arquivo NIfTI
            if (subimage.size>0 and subimage is not None):
                nib.save(subimage_nii, output_path)
        
        
    total_label1.append(f"{img.split('_')[0]}-{count_label1}")
    print(f"Total de fatias processadas do paciente {img.split('_')[0]}: {processed_slices}")
    
    
total_imagens_displasicas = 0
for i in range (len(total_label1)):
    print(f'{total_label1[i]}\n')
for i in range(len(total_label1)):
    partes = total_label1[i].split('-')
    numero = partes[2]
    total_imagens_displasicas += int(numero)
print(f"Total de subimagens com label 1: {total_imagens_displasicas}")
