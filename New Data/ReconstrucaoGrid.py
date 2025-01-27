import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages
from time import time

# Função para carregar as coordenadas dos arquivos txt
def load_coordinates(coordinates_path):
    coordinates = {}

    for patient_id in os.listdir(coordinates_path):
        patient_path = os.path.join(coordinates_path, patient_id)
        coordinates[patient_id] = []

        for slice_file in sorted(os.listdir(patient_path)):
            print(f"SLICE: {slice_file}")
            slice_path = os.path.join(patient_path, slice_file)
            with open(slice_path, 'r') as f:
                coords = [tuple(map(int, line.strip().split(','))) for line in f.readlines()]
                coordinates[patient_id].append(coords)
                
    return coordinates

# Função para carregar as coordenadas dos arquivos txt
def load_one_coordinate(coordinates_path):
    coordinates = []

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
            #print(f"patch: {patch_id} e mask: {mask_patch_id}") corrigir ordem 79, 8, 80
            images[patient_id].append(img)
            masks[patient_id].append(mask)

    return images, masks

# Função para desenhar a imagem com o grid em PDF
def plot_images_with_grid_to_pdf(images, masks, coordinates, pdf_filename):
    with PdfPages(pdf_filename) as pdf:
        for patient_id in images.keys():
            index = 0
            for img, mask in zip(images[patient_id], masks[patient_id]):
                # Verificar se existem coordenadas para a fatia atual
                if patient_id not in coordinates or index >= len(coordinates[patient_id]):
                    print(f"Aviso: Coordenadas ausentes para o paciente {patient_id}, fatia {index}")
                    index+=1
                    continue
                
                # Total de pixels na subimagem
                total_pixels = img.size
                # Número de pixels não-preto
                non_zero_pixels = np.count_nonzero(img)
                # Proporção de pixels não-preto
                non_black_ratio = non_zero_pixels / total_pixels if total_pixels > 0 else 0
        
                if non_black_ratio >= 0.04:
                    plt.figure(figsize=(8, 8))

                    # Mostrar a imagem
                    plt.subplot(1, 2, 1)
                    plt.imshow(img, cmap='gray')
                    plt.title(f"{patient_id} - Slice")
                    
                    # Desenhar os grids usando as coordenadas ajustadas (y1, y2, x1, x2)
                    for (y1, y2, x1, x2) in coordinates[patient_id][index]:
                        if y1 == -1 and y2 == -1 and x1 == -1 and x2 == -1:
                            continue
                        plt.plot(
                            [x1, x2, x2, x1, x1],  # Coordenadas horizontais
                            [y1, y1, y2, y2, y1],  # Coordenadas verticais
                            'r'
                        )

                    # Mostrar a máscara
                    plt.subplot(1, 2, 2)
                    plt.imshow(mask, cmap='gray')
                    plt.title(f"{patient_id} - Slice_Mask")
                    
                    # Desenhar os grids na máscara
                    for (y1, y2, x1, x2) in coordinates[patient_id][index]:
                        if y1 == -1 and y2 == -1 and x1 == -1 and x2 == -1:
                            continue
                        plt.plot(
                            [x1, x2, x2, x1, x1],  # Coordenadas horizontais
                            [y1, y1, y2, y2, y1],  # Coordenadas verticais
                            'r'
                        )

                    pdf.savefig()
                    plt.close()
                index+=1
            print(f"Paciente {patient_id} gerado com sucesso!")
        print(f"As imagens foram salvas no arquivo PDF {pdf_filename} com sucesso.")


# Função para desenhar a imagem com o grid em PDF
def plot_images_with_grid_to_pdf_adjusted(fatias_dir, masks_dir, grid_dir, pdf_filename):
    start_time = time()
    patients = os.listdir(fatias_dir)

    with PdfPages(pdf_filename) as pdf:
        for patient_id in patients:
            patient_start_time = time()
            print(f"paciente: {patient_id}\n\n")

            fatia_path = f"{fatias_dir}/{patient_id}" #caminho da pasta de fatias para um paciente
            fatias_names = os.listdir(fatia_path)

            mascara_path = f"{masks_dir}/{patient_id}" #caminho da pasta de mascaras para um paciente

            grid_path =  f"{grid_dir}/{patient_id}" #caminho da pasta de grids para um paciente
            grids_names = os.listdir(grid_path)

            for idx in range(len(fatias_names)):
                fatias_names[idx] = fatias_names[idx].split(".")[0] #tira o .nii.gz

            for idx in range(len(grids_names)):
                grids_names[idx] = grids_names[idx].split(".")[0] #tira o .txt

            for slice_img in fatias_names:
                if slice_img not in grids_names: # verifica se existe grid para esta imagem
                    continue
                else:
                    # caminho dos dados para esta fatia
                    img_data_path = f"{fatia_path}/{slice_img}.nii.gz"
                    mask_data_path = f"{mascara_path}/{slice_img}.nii.gz"
                    grid_data_path = f"{grid_path}/{slice_img}.txt"

                    # carrega dados para esta fatia
                    coordinates = load_one_coordinate(grid_data_path)
                    img_data = nib.load(img_data_path).get_fdata()
                    mask_data = nib.load(mask_data_path).get_fdata()

                    plt.figure(figsize=(8, 8))

                    # Mostrar a imagem
                    plt.subplot(1, 2, 1)
                    plt.imshow(img_data, cmap='gray')
                    plt.title(f"{patient_id} - Slice")
                    
                    # Desenhar os grids usando as coordenadas ajustadas (y1, y2, x1, x2)
                    for (y1, y2, x1, x2) in coordinates:
                        if y1 == -1 and y2 == -1 and x1 == -1 and x2 == -1:
                            continue
                        plt.plot(
                            [x1, x2, x2, x1, x1],  # Coordenadas horizontais
                            [y1, y1, y2, y2, y1],  # Coordenadas verticais
                            'r'
                        )

                    # Mostrar a máscara
                    plt.subplot(1, 2, 2)
                    plt.imshow(mask_data, cmap='gray')
                    plt.title(f"{patient_id} - Slice_Mask")
                    
                    # Desenhar os grids na máscara
                    for (y1, y2, x1, x2) in coordinates:
                        if y1 == -1 and y2 == -1 and x1 == -1 and x2 == -1:
                            continue
                        plt.plot(
                            [x1, x2, x2, x1, x1],  # Coordenadas horizontais
                            [y1, y1, y2, y2, y1],  # Coordenadas verticais
                            'r'
                        )

                    pdf.savefig()
                    plt.close()

            # Calcula o tempo gasto por paciente
            patient_end_time = time()
            patient_duration = patient_end_time - patient_start_time
            print(f"Tempo para o paciente {patient_id}: {patient_duration:.2f} segundos\n")

    end_time = time()
    total_duration = end_time - start_time
    print(f"Tempo total de execução: {total_duration:.2f} segundos")

# Caminhos das imagens e máscaras
image_path = "Fatias" # gerado no SalvarFatiasTodas.py
mask_path = "Mask_Fatias" # gerado no SalvarFatiasTodas.py
coordinates_path = "Coordenadas_grid" # gerado no moving_grid.ipynb
pdf_filename = "Pdf/Pacientes_com_Grid.pdf"
os.makedirs("Pdf", exist_ok=True)

# Carregar e plotar as imagens
#images, masks = load_full_image_and_mask(image_path, mask_path)

#coordinates = load_coordinates(coordinates_path)

plot_images_with_grid_to_pdf_adjusted(image_path, mask_path, coordinates_path, pdf_filename)