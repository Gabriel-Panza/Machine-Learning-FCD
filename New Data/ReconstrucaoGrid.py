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
            lines = 0
            with open(slice_path, 'r') as f:
                next(f)  # Ignora a primeira linha
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

# Função para calcular o label
def calculate_label(image, threshold=0.003125):
    total_pixels = image.size
    non_zero_pixels = np.count_nonzero(image)
    non_black_ratio = non_zero_pixels / total_pixels if total_pixels > 0 else 0
    return 1 if np.any(image == 1) and non_black_ratio >= threshold else 0

# Função para desenhar a imagem com o grid em PDF
def plot_images_with_grid_to_pdf(images, masks, coordinates, pdf_filename):
    with PdfPages(pdf_filename) as pdf:
        for patient_id in images.keys():
            for index, (img, mask) in enumerate(zip(images[patient_id], masks[patient_id])):
                if calculate_label(mask):
                    plt.figure(figsize=(8, 8))

                    # Mostrar a imagem
                    plt.subplot(1, 2, 1)
                    plt.imshow(np.flipud(img), cmap='gray')
                    plt.title(f"{patient_id} - Slice")
                    for (x_start, y_start) in coordinates[patient_id][index]:
                        # if x_start == 0 and y_start == 0:
                        #     continue
                        plt.plot([y_start, y_start, y_start+49,  y_start+49, y_start], [x_start, x_start+58,  x_start+58, x_start, x_start], 'r')

                    # Mostrar a máscara
                    plt.subplot(1, 2, 2)
                    plt.imshow(np.flipud(mask), cmap='gray')
                    plt.title(f"{patient_id} - Slice_Mask")
                    for (x_start, y_start) in coordinates[patient_id][index]:
                        # if x_start == 0 and y_start == 0:
                        #     continue
                        plt.plot([y_start, y_start, y_start+49,  y_start+49, y_start], [x_start, x_start+58,  x_start+58, x_start, x_start], 'r')

                    pdf.savefig()
                    plt.close()
        print(f"As imagens foram salvas no arquivo PDF {pdf_filename} com sucesso.")

# Caminhos das imagens e máscaras
image_path = "Fatias"
mask_path = "Mask_Fatias"
coordinates_path = "Coordenadas_grid"
pdf_filename = "Pdf/Pacientes_Reconstruidos_Grid.pdf"

# Carregar e plotar as imagens
images, masks = load_full_image_and_mask(image_path, mask_path)
coordinates = load_coordinates(coordinates_path)
plot_images_with_grid_to_pdf(images, masks, coordinates, pdf_filename)