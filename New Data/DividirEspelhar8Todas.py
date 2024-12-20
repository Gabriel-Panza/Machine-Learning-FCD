import nibabel as nib
import numpy as np
import os
import nrrd
import matplotlib.pyplot as plt

def calculate_label(subimage, threshold=0.009):
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

def divide_8_pieces_lesion(rotated_lesion_slice):
    # Dividir a fatia rotacionada em esquerda e direita
    midpoint_lesion = rotated_lesion_slice.shape[1] // 2
    left_half_lesion = rotated_lesion_slice[:, :midpoint_lesion+2]
    right_half_lesion = rotated_lesion_slice[:, midpoint_lesion-2:]

    # Inverter horizontalmente o lado direito
    right_half_lesion_flipped = np.fliplr(right_half_lesion)

    # Dividir as metades esquerda e direita horizontalmente em duas partes
    horizontal_mid_left_lesion = (left_half_lesion.shape[0]) // 2
    horizontal_mid_right_lesion = (right_half_lesion_flipped.shape[0]) // 2
    
    left_left_lesion = left_half_lesion[17:horizontal_mid_left_lesion-17, :]
    left_right_lesion = left_half_lesion[horizontal_mid_left_lesion+17:2*horizontal_mid_left_lesion-17, :]
    right_left_lesion = right_half_lesion_flipped[17:horizontal_mid_right_lesion-17, :]
    right_right_lesion = right_half_lesion_flipped[horizontal_mid_right_lesion+17:2*horizontal_mid_right_lesion-17, :]

    # Dividir cada quadrante em 2 subquadrantes (totalizando 8 divisões)
    left_left_top_lesion = left_left_lesion[:, :left_left_lesion.shape[1] // 2 +10]
    left_right_top_lesion = left_left_lesion[:, left_left_lesion.shape[1] // 2 -10:]
    left_left_bottom_lesion = left_right_lesion[:, :left_right_lesion.shape[1] // 2 +10]
    left_right_bottom_lesion = left_right_lesion[:, left_right_lesion.shape[1] // 2 -10:]
    right_left_top_lesion = right_left_lesion[:, :right_left_lesion.shape[1] // 2 +10]
    right_right_top_lesion = right_left_lesion[:, right_left_lesion.shape[1] // 2 -9:]
    right_left_bottom_lesion = right_right_lesion[:, :right_right_lesion.shape[1] // 2 +10]
    right_right_bottom_lesion = right_right_lesion[:, right_right_lesion.shape[1] // 2 -9:]
    
    #print(left_left_top_lesion.shape, left_right_top_lesion.shape, left_left_bottom_lesion.shape, left_right_bottom_lesion.shape, right_left_top_lesion.shape, right_right_top_lesion.shape, right_left_bottom_lesion.shape, right_right_bottom_lesion.shape)
    
    return left_left_top_lesion, left_right_top_lesion, left_left_bottom_lesion, left_right_bottom_lesion, right_left_top_lesion, right_right_top_lesion, right_left_bottom_lesion, right_right_bottom_lesion

def divide_8_pieces(rotated_slice):
    # Dividir a fatia rotacionada em esquerda e direita
    midpoint = rotated_slice.shape[1] // 2
    left_half = rotated_slice[:, 2:midpoint]
    right_half = rotated_slice[:, midpoint:(2*midpoint) - 3]

    # Inverter horizontalmente o lado direito
    right_half_flipped = np.fliplr(right_half)

    # Dividir as metades esquerda e direita horizontalmente em duas partes
    horizontal_mid_left = (left_half.shape[0]) // 2
    horizontal_mid_right = (right_half_flipped.shape[0]) // 2
    
    left_left = left_half[17:horizontal_mid_left-17, :]
    left_right = left_half[horizontal_mid_left+17:2*horizontal_mid_left-17, :]
    right_left = right_half_flipped[17:horizontal_mid_right-17, :]
    right_right = right_half_flipped[horizontal_mid_right+17:2*horizontal_mid_right-17, :]

    # Dividir cada quadrante em 2 subquadrantes (totalizando 8 divisões)
    left_left_top = left_left[:, :(left_left.shape[1] // 2)+12]
    left_right_top = left_left[:, (left_left.shape[1] // 2)-12:]
    left_left_bottom = left_right[:, :(left_right.shape[1] // 2)+12]
    left_right_bottom = left_right[:, (left_right.shape[1] // 2)-12:]
    right_left_top = right_left[:, :(right_left.shape[1] // 2)+13]
    right_right_top = right_left[:, (right_left.shape[1] // 2)-12:]
    right_left_bottom = right_right[:, :(right_right.shape[1] // 2)+13]
    right_right_bottom = right_right[:, (right_right.shape[1] // 2)-12:]
    
    slices = [left_left_top, right_left_top, left_right_top, right_right_top, left_left_bottom, right_left_bottom, left_right_bottom, right_right_bottom]
    
    #print(left_left_top.shape, left_right_top.shape, left_left_bottom.shape, left_right_bottom.shape, right_left_top.shape, right_right_top.shape, right_left_bottom.shape, right_right_bottom.shape)
    
    return left_left_top, left_right_top, left_left_bottom, left_right_bottom, right_left_top, right_right_top, right_left_bottom, right_right_bottom
      
imagens = "Patients_Displasya"
mascara = "Mascaras"

total_label1 = []

for img, mask in zip([f for f in os.listdir(imagens) if f.endswith(('.nii', '.nii.gz'))], [f for f in os.listdir(mascara) if f.endswith(('.nrrd', '.nii', '.nii.gz'))]):    
    data = nib.load(os.path.join(imagens, img)).get_fdata()
    lesion_data, _ = nrrd.read(os.path.join(mascara, mask))
    data = np.transpose(data, (2, 0, 1))
    lesion_data = np.transpose(lesion_data, (2, 0, 1))
    print(data.shape)
    print(lesion_data.shape)
    if (lesion_data.shape[2]>data.shape[2]):
        continue
    # Definir o limite para considerar os pixels não pretos
    non_black_threshold = 0.1 / 255 
    
    # Definir a porcentagem mínima de pixels não pretos para exibir a imagem
    min_percentage_non_black = 0.2

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
        # Pega a lesão da fatia inteira
        lesion_slice_data = lesion_data[:, :, slice_idx]
        
        # Contar o número total de pixels
        total_pixels_lesion = lesion_slice_data.size
        
        # Contar o número de pixels que são considerados não pretos
        non_black_pixels_lesion = np.sum(lesion_slice_data > non_black_threshold)

        # Calcular a porcentagem de pixels não pretos
        percentage_non_black_lesion = non_black_pixels_lesion / total_pixels_lesion
        
        # Selecionar a fatia axial atual
        slice_data = data[:, :, slice_idx]
                
        # Contar o número total de pixels
        total_pixels = slice_data.size
        
        # Contar o número de pixels que são considerados não pretos
        non_black_pixels = np.sum(slice_data > non_black_threshold)

        # Calcular a porcentagem de pixels não pretos
        percentage_non_black = non_black_pixels / total_pixels

        # Se a porcentagem de pixels não pretos for maior que 20%, processar a fatia
        if percentage_non_black > min_percentage_non_black:
            output_dir_left_slice = os.path.join(output_dir_left, f"Slice{slice_idx}/")
            output_dir_right_slice = os.path.join(output_dir_right, f"Slice{slice_idx}/")
    
            os.makedirs(output_dir_left_slice, exist_ok=True)
            os.makedirs(output_dir_right_slice, exist_ok=True)
            
            processed_slices += 1

            output_dir_lesion_left_slice = os.path.join(output_dir_lesion_left, f"Slice{slice_idx}")
            output_dir_lesion_right_slice = os.path.join(output_dir_lesion_right, f"Slice{slice_idx}")
            
            os.makedirs(output_dir_lesion_left_slice, exist_ok=True)
            os.makedirs(output_dir_lesion_right_slice, exist_ok=True)
            
            left_top_left_lesion, left_top_right_lesion, left_bottom_left_lesion, left_bottom_right_lesion, right_top_left_lesion, right_top_right_lesion, right_bottom_left_lesion, right_bottom_right_lesion = divide_8_pieces_lesion(lesion_slice_data)

            # print(left_top_left_lesion.shape, left_top_right_lesion.shape, left_bottom_left_lesion.shape, left_bottom_right_lesion.shape, right_top_left_lesion.shape, right_top_right_lesion.shape, right_bottom_left_lesion.shape, right_bottom_right_lesion.shape)
            
            count_label1_anterior = count_label1
            if calculate_label(left_top_left_lesion) == "label1":
                count_label1 +=1

            if calculate_label(left_top_right_lesion) == "label1":
                count_label1 +=1

            if calculate_label(left_bottom_left_lesion) == "label1":
                count_label1 +=1

            if calculate_label(left_bottom_right_lesion) == "label1":
                count_label1 +=1

            if calculate_label(right_top_left_lesion) == "label1":
                count_label1 +=1

            if calculate_label(right_top_right_lesion) == "label1":
                count_label1 +=1

            if calculate_label(right_bottom_left_lesion) == "label1":
                count_label1 +=1

            if calculate_label(right_bottom_right_lesion) == "label1":
                count_label1 +=1
            count_label1_posterior = count_label1
            
            print(f"Total de subimagens com label 1: {count_label1}")
            
            # Lista com todas as subimagens e identificações
            subimages = [
                (left_top_left_lesion, f"left_top_left_lesion_{calculate_label(left_top_left_lesion)}"),
                (left_top_right_lesion, f"left_top_right_lesion_{calculate_label(left_top_right_lesion)}"),
                (left_bottom_left_lesion, f"left_bottom_left_lesion_{calculate_label(left_bottom_left_lesion)}"),
                (left_bottom_right_lesion, f"left_bottom_right_lesion_{calculate_label(left_bottom_right_lesion)}"),
                (right_bottom_left_lesion, f"right_bottom_left_lesion_{calculate_label(right_bottom_left_lesion)}"),
                (right_bottom_right_lesion, f"right_bottom_right_lesion_{calculate_label(right_bottom_right_lesion)}"),
                (right_top_left_lesion, f"right_top_left_lesion_{calculate_label(right_top_left_lesion)}"),
                (right_top_right_lesion, f"right_top_right_lesion_{calculate_label(right_top_right_lesion)}"),
            ]

            # Ajustar os pedaços da máscara para remover pedaços únicos com pixels `1`
            if (count_label1_anterior != count_label1_posterior):
                subimages = adjust_unique_lesion_pieces_with_neighbors(subimages, slice_idx, lesion_data.shape[2], lesion_data)
            
            # Salvar cada subimagem como um arquivo NIfTI separado
            for subimage, position in subimages:
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

            left_top_left, left_top_right, left_bottom_left, left_bottom_right, right_top_left, right_top_right, right_bottom_left, right_bottom_right = divide_8_pieces(slice_data)
    
            # print(left_top_left.shape, left_top_right.shape, left_bottom_left.shape, left_bottom_right.shape, right_top_left.shape, right_top_right.shape, right_bottom_left.shape, right_bottom_right.shape)
            
            # Lista com todas as subimagens e identificações
            subimages = [
                (left_top_left, f"left_top_left"),
                (left_top_right, f"left_top_right"),
                (left_bottom_left, f"left_bottom_left"),
                (left_bottom_right, f"left_bottom_right"),
                (right_bottom_left, f"right_bottom_left"),
                (right_bottom_right, f"right_bottom_right"),
                (right_top_left, f"right_top_left"),
                (right_top_right, f"right_top_right"),
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
