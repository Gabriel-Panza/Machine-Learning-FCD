import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages

# Coordenadas do grid baseado nas divisões anteriores
coordinates = [
    (26, 75, 36, 78), (47, 96, 36, 78),
    (26, 75, 76, 118), (47, 96, 76, 118),
    (26, 75, 116, 158), (47, 96, 116, 158),
    (26, 75, 156, 198), (47, 96, 156, 198),
    (96, 145, 36, 78), (120, 169, 36, 78),
    (96, 145, 76, 118), (120, 169, 76, 118),
    (96, 145, 116, 158), (120, 169, 116, 158),
    (96, 145, 156, 198), (120, 169, 156, 198)
]

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
                if (np.any(mask) == 1):
                    plt.figure(figsize=(8, 8))

                    # Mostrar a imagem
                    plt.subplot(1, 2, 1)
                    plt.imshow(np.flipud(img), cmap='gray')
                    plt.title(f"{patient_id} - Slice_{index}")
                    for (x_start, x_end, y_start, y_end) in coordinates:
                        plt.plot([x_start, x_end, x_end, x_start, x_start], [y_start, y_start, y_end, y_end, y_start], 'r')

                    # Mostrar a máscara
                    plt.subplot(1, 2, 2)
                    plt.imshow(np.flipud(mask), cmap='gray')
                    plt.title(f"{patient_id} - Slice_Mask_{index}")
                    for (x_start, x_end, y_start, y_end) in coordinates:
                        plt.plot([x_start, x_end, x_end, x_start, x_start], [y_start, y_start, y_end, y_end, y_start], 'r')

                    pdf.savefig()
                    plt.close()
                index+=1
        print(f"As imagens foram salvas no arquivo PDF {pdf_filename} com sucesso.")

# Caminhos das imagens e máscaras
image_path = "Fatias"
mask_path = "Mask_Fatias"
pdf_filename = "Pacientes_Reconstruidos_Grid.pdf"

# Carregar e plotar as imagens
images, masks = load_full_image_and_mask(image_path, mask_path)
plot_images_with_grid_to_pdf(images, masks, coordinates, pdf_filename)