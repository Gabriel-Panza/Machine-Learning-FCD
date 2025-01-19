import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages

# Função para carregar as coordenadas dos arquivos txt
def load_coordinates(coordinates_path):
    coordinates = {}
    for patient_id in os.listdir(coordinates_path):
        patient_path = os.path.join(coordinates_path, patient_id)
        coordinates[patient_id] = []
        for slice_file in sorted(os.listdir(patient_path)):
            slice_path = os.path.join(patient_path, slice_file)
            with open(slice_path, 'r') as f:
                coords = [tuple(map(int, line.strip().split(','))) for line in f.readlines()]
                coordinates[patient_id].append(coords)
    return coordinates

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

# Função para desenhar a imagem com o grid em PDF
def plot_images_with_grid_to_pdf(images, masks, coordinates, pdf_filename):
    with PdfPages(pdf_filename) as pdf:
        for patient_id in images.keys():
            index = 0
            for img, mask in zip(images[patient_id], masks[patient_id]):
                # # Verificar se existem coordenadas para a fatia atual
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

# Caminhos das imagens e máscaras
image_path = "Fatias"
mask_path = "Mask_Fatias"
coordinates_path = "Coordenadas_grid"
pdf_filename = "Pdf/Pacientes_com_Grid.pdf"

# Carregar e plotar as imagens
images, masks = load_full_image_and_mask(image_path, mask_path)
coordinates = load_coordinates(coordinates_path)
plot_images_with_grid_to_pdf(images, masks, coordinates, pdf_filename)