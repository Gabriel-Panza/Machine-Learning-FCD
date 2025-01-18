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

def divide_16_pieces(slice, slice_mask):
    midpoint = slice.shape[1] // 2
    left_half = slice[:, :midpoint]
    right_half = slice[:, midpoint:midpoint*2]

    midpoint_lesion = slice_mask.shape[1] // 2
    left_half_lesion = slice_mask[:, :midpoint_lesion]
    right_half_lesion = slice_mask[:, midpoint_lesion:midpoint_lesion*2]
    
    # Inverter horizontalmente o lado direito
    right_half_flipped = np.fliplr(right_half)
    right_half_lesion_flipped = np.fliplr(right_half_lesion)

    # Dividir as metades esquerda e direita horizontalmente em duas partes
    horizontal_mid_left = left_half.shape[0] // 2
    horizontal_mid_right = right_half_flipped.shape[0] // 2

    horizontal_mid_left_lesion = left_half_lesion.shape[0] // 2
    horizontal_mid_right_lesion = right_half_lesion_flipped.shape[0] // 2
    
    top_left = left_half[:horizontal_mid_left, :]
    top_right = left_half[horizontal_mid_left:horizontal_mid_left*2, :]
    bottom_left = right_half_flipped[:horizontal_mid_right, :]
    bottom_right = right_half_flipped[horizontal_mid_right:horizontal_mid_right*2, :]

    lesion_top_left = left_half_lesion[:horizontal_mid_left_lesion, :]
    lesion_top_right = left_half_lesion[horizontal_mid_left_lesion:horizontal_mid_left_lesion*2, :]
    lesion_bottom_left = right_half_lesion_flipped[:horizontal_mid_right_lesion, :]
    lesion_bottom_right = right_half_lesion_flipped[horizontal_mid_right_lesion:horizontal_mid_right_lesion*2, :]
    
    # Coordenadas iniciais de cada quadrante
    coordinates = {
        "top_left": (0, 0),
        "top_right": (0, midpoint),
        "bottom_left": (horizontal_mid_left, 0),
        "bottom_right": (horizontal_mid_left, midpoint)
    }
    
    # Dividir cada quadrante em 4 subquadrantes (totalizando 16 divisões) + as coordenadas
    def split_quadrant(quadrant, start_x, start_y):
        vertical_mid = quadrant.shape[0] // 2
        horizontal_mid = quadrant.shape[1] // 2
        
        top_left = quadrant[:vertical_mid, :horizontal_mid]
        top_right = quadrant[:vertical_mid, horizontal_mid:]
        bottom_left = quadrant[vertical_mid:, :horizontal_mid]
        bottom_right = quadrant[vertical_mid:, horizontal_mid:]
        
        # Armazenar as coordenadas de cada subquadrante
        sub_coordinates = [
            (start_x, start_y),  # top_left
            (start_x, start_y+horizontal_mid),  # top_right
            (start_x+vertical_mid, start_y),  # bottom_left
            (start_x+vertical_mid, start_y+horizontal_mid)  # bottom_right
        ]
        
        return top_left, top_right, bottom_left, bottom_right, sub_coordinates
    
    # Dividir cada quadrante em 4 subquadrantes (totalizando 16 divisões)
    def split_quadrant_lesion(quadrant):
        vertical_mid = quadrant.shape[0] // 2
        horizontal_mid = quadrant.shape[1] // 2
        
        top_left = quadrant[:vertical_mid, :horizontal_mid]
        top_right = quadrant[:vertical_mid, horizontal_mid:]
        bottom_left = quadrant[vertical_mid:, :horizontal_mid]
        bottom_right = quadrant[vertical_mid:, horizontal_mid:]
        
        return top_left, top_right, bottom_left, bottom_right
    
    top_left_top_left, top_left_top_right, top_left_bottom_left, top_left_bottom_right, top_left_coords = split_quadrant(top_left, *coordinates["top_left"])
    top_right_top_left, top_right_top_right, top_right_bottom_left, top_right_bottom_right, top_right_coords = split_quadrant(top_right, *coordinates["top_right"])
    bottom_left_top_left, bottom_left_top_right, bottom_left_bottom_left, bottom_left_bottom_right, bottom_left_coords = split_quadrant(bottom_left, *coordinates["bottom_left"])
    bottom_right_top_left, bottom_right_top_right, bottom_right_bottom_left, bottom_right_bottom_right, bottom_right_coords = split_quadrant(bottom_right, *coordinates["bottom_right"])

    lesion_top_left_top_left, lesion_top_left_top_right, lesion_top_left_bottom_left, lesion_top_left_bottom_right = split_quadrant_lesion(lesion_top_left)
    lesion_top_right_top_left, lesion_top_right_top_right, lesion_top_right_bottom_left, lesion_top_right_bottom_right = split_quadrant_lesion(lesion_top_right)
    lesion_bottom_left_top_left, lesion_bottom_left_top_right, lesion_bottom_left_bottom_left, lesion_bottom_left_bottom_right = split_quadrant_lesion(lesion_bottom_left)
    lesion_bottom_right_top_left, lesion_bottom_right_top_right, lesion_bottom_right_bottom_left, lesion_bottom_right_bottom_right = split_quadrant_lesion(lesion_bottom_right)
    
    # Unindo todas as coordenadas
    all_coordinates = top_left_coords + bottom_left_coords + top_right_coords + bottom_right_coords

    quadrants_around = [top_left_top_left, top_left_top_right, top_left_bottom_left, top_left_bottom_right, bottom_left_top_left, bottom_left_top_right, bottom_left_bottom_left, bottom_left_bottom_right, top_right_top_left, top_right_top_right, top_right_bottom_left, top_right_bottom_right, bottom_right_top_left, bottom_right_top_right, bottom_right_bottom_left, bottom_right_bottom_right]
    quadrants_around_lesion = [lesion_top_left_top_left, lesion_top_left_top_right, lesion_top_left_bottom_left, lesion_top_left_bottom_right, lesion_bottom_left_top_left, lesion_bottom_left_top_right, lesion_bottom_left_bottom_left, lesion_bottom_left_bottom_right, lesion_top_right_top_left, lesion_top_right_top_right, lesion_top_right_bottom_left, lesion_top_right_bottom_right, lesion_bottom_right_top_left, lesion_bottom_right_top_right, lesion_bottom_right_bottom_left, lesion_bottom_right_bottom_right]
    for idx in range(len(quadrants_around)):
        if (not np.any(quadrants_around[idx]>0)):
            if idx > 7:
                quadrants_around[idx-1] = []
                quadrants_around_lesion[idx-1] = []
            else:
                quadrants_around[idx] = []
                quadrants_around_lesion[idx] = []    
    for idx in range(8):
        if len(quadrants_around[idx]) != len(quadrants_around[idx+8]):
            quadrants_around[idx] = []
            quadrants_around[idx+8] = []
            quadrants_around_lesion[idx] = []
            quadrants_around_lesion[idx+8] = []
    for idx in range(len(quadrants_around)-1):
        if (len(quadrants_around[idx]) == 0):
            if idx > 7:
                all_coordinates[idx+1] = (-1,-1)
            else:
                all_coordinates[idx] = (-1,-1)

        # elif np.any(recorte):  # Garantir que recorte não está vazio
        #     while not (np.any(recorte[0, :]) and np.any(recorte[-1, :]) and np.any(recorte[:, 0]) and np.any(recorte[:, -1])):
        #         if not np.any(recorte[0, :]):
        #             recorte = np.roll(recorte, shift=1, axis=0) # Incremento vertical
        #             quadrants_around_lesion[idx] = np.roll(quadrants_around_lesion[idx], shift=1, axis=0) 
        #             all_coordinates[idx] = (all_coordinates[idx][0] + 1, all_coordinates[idx][1])  
        #         if not np.any(recorte[:, 0]):
        #             recorte = np.roll(recorte, shift=1, axis=1) # Incremento horizontal
        #             quadrants_around_lesion[idx] = np.roll(quadrants_around_lesion[idx], shift=1, axis=1)
        #             all_coordinates[idx] = (all_coordinates[idx][0], all_coordinates[idx][1] + 1)  
        #         elif not np.any(recorte[:, -1]):
        #             recorte = np.roll(recorte, shift=-1, axis=1) # Decremento horizontal
        #             quadrants_around_lesion[idx] = np.roll(quadrants_around_lesion[idx], shift=-1, axis=1)
        #             all_coordinates[idx] = (all_coordinates[idx][0], all_coordinates[idx][1] - 1)  
        #         interactions+=1
        #         if interactions >= max_iterations:
        #             break
            
    return (quadrants_around[0], quadrants_around[1], quadrants_around[2], quadrants_around[3], 
            quadrants_around[4], quadrants_around[5], quadrants_around[6], quadrants_around[7], 
            quadrants_around[8], quadrants_around[9], quadrants_around[10], quadrants_around[11],
            quadrants_around[12], quadrants_around[13], quadrants_around[14], quadrants_around[15], 
            quadrants_around_lesion[0], quadrants_around_lesion[1], quadrants_around_lesion[2], quadrants_around_lesion[3],
            quadrants_around_lesion[4], quadrants_around_lesion[5], quadrants_around_lesion[6], quadrants_around_lesion[7],
            quadrants_around_lesion[8], quadrants_around_lesion[9], quadrants_around_lesion[10], quadrants_around_lesion[11],
            quadrants_around_lesion[12], quadrants_around_lesion[13], quadrants_around_lesion[14], quadrants_around_lesion[15], 
            all_coordinates)


def save_coordinates_to_txt(coordinates, filename):
    # Converte a lista de coordenadas em um array numpy e salva como txt
    np.savetxt(filename, coordinates, fmt='%d', delimiter=',', header="x,y", comments='')


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

    # Contador para acompanhar quantas fatias foram processadas
    processed_slices = 0

    # Diretório de saída para salvar as fatias
    output_dir_left = os.path.join(f"Novo_Contralateral/{img.split('_')[0]}", "left")
    output_dir_right = os.path.join(f"Novo_Contralateral/{img.split('_')[0]}", "right")
    output_dir_lesion_left = os.path.join(f"Novo_Contralateral/{mask.split(' ')[0]}", "lesion_left")
    output_dir_lesion_right = os.path.join(f"Novo_Contralateral/{mask.split(' ')[0]}", "lesion_right")
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
        
        left_top_top_left, left_top_top_right, left_top_bottom_left, left_top_bottom_right, left_bottom_top_left, left_bottom_top_right, left_bottom_bottom_left, left_bottom_bottom_right, right_top_top_left, right_top_top_right, right_top_bottom_left, right_top_bottom_right, right_bottom_top_left, right_bottom_top_right, right_bottom_bottom_left, right_bottom_bottom_right, lesion_top_left_top_left, lesion_top_left_top_right, lesion_top_left_bottom_left, lesion_top_left_bottom_right, lesion_top_right_top_left, lesion_top_right_top_right, lesion_top_right_bottom_left, lesion_top_right_bottom_right, lesion_bottom_left_top_left, lesion_bottom_left_top_right, lesion_bottom_left_bottom_left, lesion_bottom_left_bottom_right, lesion_bottom_right_top_left, lesion_bottom_right_top_right, lesion_bottom_right_bottom_left, lesion_bottom_right_bottom_right, coordinates = divide_16_pieces(slice_data, lesion_slice_data)

        for elem in [left_top_top_left, left_top_top_right, left_top_bottom_left, left_top_bottom_right, left_bottom_top_left, left_bottom_top_right, left_bottom_bottom_left, left_bottom_bottom_right, right_top_top_left, right_top_top_right, right_top_bottom_left, right_top_bottom_right, right_bottom_top_left, right_bottom_top_right, right_bottom_bottom_left, right_bottom_bottom_right]:
            if len(elem)>0:
                print(elem.shape)
            else:
                print(len(elem))
        
        # Lista com todas as subimagens e identificações
        subimages = [
            (left_top_top_left, f"left_1"),
            (right_top_top_left, f"right_1"),
            (left_top_top_right, f"left_2"),
            (right_top_top_right, f"right_2"),
            (left_top_bottom_left, f"left_3"),
            (right_top_bottom_left, f"right_3"),
            (left_top_bottom_right, f"left_4_"),
            (right_top_bottom_right, f"right_4"),
            (left_bottom_top_left, f"left_5"),
            (right_bottom_top_left, f"right_5"),
            (left_bottom_top_right, f"left_6"),
            (right_bottom_top_right, f"right_6"),
            (left_bottom_bottom_left, f"left_7"),
            (right_bottom_bottom_left, f"right_7"),
            (left_bottom_bottom_right, f"left_8"),
            (right_bottom_bottom_right, f"right_8")
        ]

        # Salvar cada subimagem como um arquivo NIfTI separado
        for subimage, position in subimages:            
            # Definir o diretório de saída com base na posição
            if position.startswith("left"):
                output_path = os.path.join(output_dir_left_slice, f"{position}.nii.gz")
            else:
                output_path = os.path.join(output_dir_right_slice, f"{position}.nii.gz")

            # Salvar o arquivo NIfTI
            if (len(subimage) > 0):
                # Converter o array numpy para um objeto NIfTI
                subimage_nii = nib.Nifti1Image(subimage, affine=np.eye(4))
                nib.save(subimage_nii, output_path)
                
        for lesion_part in [lesion_top_left_top_left, lesion_top_left_top_right, lesion_top_left_bottom_left, lesion_top_left_bottom_right, lesion_top_right_top_left, lesion_top_right_top_right, lesion_top_right_bottom_left, lesion_top_right_bottom_right, lesion_bottom_left_top_left, lesion_bottom_left_top_right, lesion_bottom_left_bottom_left, lesion_bottom_left_bottom_right, lesion_bottom_right_top_left, lesion_bottom_right_top_right, lesion_bottom_right_bottom_left, lesion_bottom_right_bottom_right]:
            if len(lesion_part)>0:
                print(lesion_part.shape)
                if calculate_label(lesion_part) == "label1":
                    count_label1 += 1
            else:
                print(len(lesion_part))
        
        # Lista com todas as subimagens e identificações
        subimages_lesion = [
            (lesion_top_left_top_left, f"left_1_lesion"),
            (lesion_top_right_top_left, f"right_1_lesion"),
            (lesion_top_left_top_right, f"left_2_lesion"),
            (lesion_top_right_top_right, f"right_2_lesion"),
            (lesion_top_left_bottom_left, f"left_3_lesion"),
            (lesion_top_right_bottom_left, f"right_3_lesion"),
            (lesion_top_left_bottom_right, f"left_4_lesion"),
            (lesion_top_right_bottom_right, f"right_4_lesion"),
            (lesion_bottom_left_top_left, f"left_5_lesion"),
            (lesion_bottom_right_top_left, f"right_5_lesion"),
            (lesion_bottom_left_top_right, f"left_6_lesion"),
            (lesion_bottom_right_top_right, f"right_6_lesion"),
            (lesion_bottom_left_bottom_left, f"left_7_lesion"),
            (lesion_bottom_right_bottom_left, f"right_7_lesion"),
            (lesion_bottom_left_bottom_right, f"left_8_lesion"),
            (lesion_bottom_right_bottom_right, f"right_8_lesion")
        ]
        
        # Salvar cada subimagem como um arquivo NIfTI separado
        for subimage, position in subimages_lesion:
            # Definir o diretório de saída com base na posição
            if position.startswith("left"):
                output_path_lesion = os.path.join(output_dir_lesion_left_slice, f"{position}.nii.gz")
            else:
                output_path_lesion = os.path.join(output_dir_lesion_right_slice, f"{position}.nii.gz")

            # Salvar o arquivo NIfTI
            if (len(subimage) > 0):
                # Converter o array numpy para um objeto NIfTI
                subimage_nii = nib.Nifti1Image(subimage, affine=np.eye(4))
                nib.save(subimage_nii, output_path_lesion)
        
        path_coordenadas = f"Coordenadas_grid"
        os.makedirs(path_coordenadas, exist_ok=True)
        path_coordenadas = f"Coordenadas_grid/{img.split('_')[0]}"
        os.makedirs(path_coordenadas, exist_ok=True)
        path_coordenadas = f"Coordenadas_grid/{img.split('_')[0]}/Slice_{slice_idx}.txt"
        save_coordinates_to_txt(coordinates, path_coordenadas)
        
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
