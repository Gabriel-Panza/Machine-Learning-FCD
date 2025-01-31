import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages
from time import time

def calculate_label(subimage, threshold=0.01):
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
        return 1
    else:
        return 0
    
# Função para carregar as coordenadas dos arquivos txt
def load_one_coordinate(coordinates_path):
    with open(coordinates_path, 'r') as f:
        coords = [tuple(map(int, line.strip().split(','))) for line in f.readlines()]
                
    return coords

# Função para carregar a imagem e a máscara
def load_full_image_and_mask(image_path, mask_path):
    images = {}
    masks = {}
    patient_ids = []
    
    for patient_id, mask_id in tqdm(zip(os.listdir(image_path), os.listdir(mask_path)), desc="Carregamento de arquivos NIfTI..."):
        patient_path = os.path.join(image_path, patient_id)
        mask_patient_path = os.path.join(mask_path, mask_id)
        patient_ids.append(patient_id)
        images[patient_id] = []
        masks[patient_id] = []

        for patch_id, mask_patch_id in zip(os.listdir(patient_path), os.listdir(mask_patient_path)):
            img = nib.load(os.path.join(patient_path, patch_id)).get_fdata()
            mask = nib.load(os.path.join(mask_patient_path, mask_patch_id)).get_fdata()
            images[patient_id].append(img)
            masks[patient_id].append(mask)

    return images, masks

# Função para desenhar a imagem com o grid em PDF e destacar áreas com displasia e seu espelho
def plot_images_with_grid_to_pdf_adjusted(fatias_dir, masks_dir, grid_dir, pdf_filename):
    start_time = time()
    patients = os.listdir(fatias_dir)

    with PdfPages(pdf_filename) as pdf:
        for patient_id in patients:
            patient_start_time = time()
            print(f"Paciente: {patient_id}")

            grid_vec = []
            imgs_vec = []
            mask_vec = []

            fatia_path = f"{fatias_dir}/{patient_id}"  # Caminho das fatias do paciente

            mascara_path = f"{masks_dir}/{patient_id}"  # Caminho das máscaras do paciente

            grid_path = f"{grid_dir}/{patient_id}"  # Caminho dos grids do paciente
            grids_names = os.listdir(grid_path)

            # Removendo extensão dos arquivos para comparações
            grids_names = [g.split(".")[0] for g in grids_names]

            for item in grids_names:
                img_data_path = f"{fatia_path}/{item}.nii.gz"
                mask_data_path = f"{mascara_path}/{item}.nii.gz"
                grid_data_path = f"{grid_path}/{item}.txt"

                coordinates = load_one_coordinate(grid_data_path)
                img_data = nib.load(img_data_path).get_fdata()
                mask_data = nib.load(mask_data_path).get_fdata()

                grid_vec.append(coordinates)
                imgs_vec.append(img_data)
                mask_vec.append(mask_data)

            for img_data, mask_data, coordinates in zip(imgs_vec, mask_vec, grid_vec):

                height, width = mask_data.shape  # Dimensões da imagem

                plt.figure(figsize=(8, 8))

                # Mostrar a imagem original
                plt.subplot(1, 2, 1)
                plt.imshow(img_data, cmap='gray')
                plt.title(f"{patient_id} - Slice")

                # Criar lista para coordenadas destacadas
                highlighted_grids = []

                # Desenhar os grids na imagem original
                for (y1, y2, x1, x2) in coordinates:
                    if y1 == -1 and y2 == -1 and x1 == -1 and x2 == -1:
                        continue

                    # Verifica se o grid contém pixels da máscara (valor 1)
                    grid_mask = mask_data[y1:y2, x1:x2]
                    has_displasia = calculate_label(grid_mask)

                    if has_displasia:
                        color = 'blue'
                        linewidth = 1 
                        highlighted_grids.append((y1, y2, x1, x2))  # Salva para pintar o espelho
                    else:
                        color = 'red'
                        linewidth = 1

                    plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color=color, linewidth=linewidth)

                # Desenhar grids espelhados
                for (y1, y2, x1, x2) in highlighted_grids:
                    x1_mirror = width - x2
                    x2_mirror = width - x1

                    plt.plot([x1_mirror, x2_mirror, x2_mirror, x1_mirror, x1_mirror], 
                             [y1, y1, y2, y2, y1], 
                             color='blue', linewidth=1)

                # Mostrar a máscara correspondente
                plt.subplot(1, 2, 2)
                plt.imshow(mask_data, cmap='gray')
                plt.title(f"{patient_id} - Slice_Mask")

                # Desenhar os grids na máscara
                for (y1, y2, x1, x2) in coordinates:
                    if y1 == -1 and y2 == -1 and x1 == -1 and x2 == -1:
                        continue

                    grid_mask = mask_data[y1:y2, x1:x2]
                    has_displasia = calculate_label(grid_mask)

                    if has_displasia:
                        color = 'blue'
                        linewidth = 1
                    else:
                        color = 'red'
                        linewidth = 1

                    plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color=color, linewidth=linewidth)

                # Desenhar grids espelhados na máscara
                for (y1, y2, x1, x2) in highlighted_grids:
                    x1_mirror = width - x2
                    x2_mirror = width - x1

                    plt.plot([x1_mirror, x2_mirror, x2_mirror, x1_mirror, x1_mirror], 
                             [y1, y1, y2, y2, y1], 
                             color='blue', linewidth=1)

                pdf.savefig()
                plt.close()

            # Tempo gasto por paciente
            patient_end_time = time()
            patient_duration = patient_end_time - patient_start_time
            print(f"Tempo para o paciente {patient_id}: {patient_duration:.2f} segundos\n\n")

    end_time = time()
    total_duration = end_time - start_time
    print(f"Tempo total de execução: {total_duration:.2f} segundos")

# Caminhos das imagens e máscaras
image_path = "Fatias"                        # gerado no SalvarFatiasTodas.py
mask_path = "Mask_Fatias"                    # gerado no SalvarFatiasTodas.py
coordinates_path = "Coordenadas_grid"        # gerado no moving_grid.ipynb
pdf_filename = "Pdf/Pacientes_com_Grid.pdf"  # local de salvamento do pdf
os.makedirs("Pdf", exist_ok=True)

plot_images_with_grid_to_pdf_adjusted(image_path, mask_path, coordinates_path, pdf_filename)