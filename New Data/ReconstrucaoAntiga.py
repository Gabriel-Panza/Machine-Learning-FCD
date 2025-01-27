import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

def calculate_label(image, threshold=0.05):
    """
    Determina o label da subimagem com base no percentual de fundo não-preto.
    :param subimage: Array da subimagem.
    :param threshold: Percentual mínimo de fundo não-preto para considerar como label 1.
    :return: String indicando o label.
    """
    # Total de pixels na subimagem
    total_pixels = image.size
    # Número de pixels não-preto
    non_zero_pixels = np.count_nonzero(image)
    # Proporção de pixels não-preto
    non_black_ratio = non_zero_pixels / total_pixels if total_pixels > 0 else 0
    
    # Verifica se há lesão e se o fundo não-preto é maior que o limiar
    if np.any(image == 1) and non_black_ratio >= threshold:
        return 1
    else:
        return 0

# Função que carrega os dados com pares de imagens
def load_data_with_pairs(folder):
    if not os.path.exists(folder):
        print(f"A pasta {folder} não existe.")
        return {}, {}, {}, {}, {}
    
    images_left = {}
    images_right = {}
    mask_left = {}
    mask_right = {}
    labels_left = {}
    labels_right = {}
    patient_ids = []

    # Itera sobre os pacientes no diretório
    for patient_id in tqdm(os.listdir(folder), desc="Carregamento de arquivos NIfTI..."):
        patient_path = os.path.join(folder, patient_id)

        areas_image = ["left", "right"]
        areas_mask = ["lesion_left", "lesion_right"]
        path_left = os.path.join(patient_path, areas_image[0])
        path_right = os.path.join(patient_path, areas_image[1])
        lesion_path_left = os.path.join(patient_path, areas_mask[0])
        lesion_path_right = os.path.join(patient_path, areas_mask[1])

        if patient_id not in images_left:
            images_left[patient_id] = []
        if patient_id not in images_right:
            images_right[patient_id] = []
        if patient_id not in mask_left:
            mask_left[patient_id] = []
        if patient_id not in mask_right:
            mask_right[patient_id] = []
        if patient_id not in labels_left:
            labels_left[patient_id] = []
        if patient_id not in labels_right:
            labels_right[patient_id] = []
                    
        # Carrega as imagens e máscaras do lado esquerdo
        for patch_id, mask_id in zip(os.listdir(path_left), os.listdir(lesion_path_left)):
            img_path= os.path.join(path_left, patch_id)
            mask_path = os.path.join(lesion_path_left, mask_id)
            for img_path_left, mask_path_left in zip(os.listdir(img_path), os.listdir(mask_path)):
                image_data_left = nib.load(os.path.join(img_path, img_path_left)).get_fdata()
                mask_data_left = nib.load(os.path.join(mask_path, mask_path_left)).get_fdata()
                if (len(image_data_left) > 0 and image_data_left is not []) or (len(mask_data_left) > 0 and mask_data_left is not []):
                    images_left[patient_id].append(image_data_left)
                    mask_left[patient_id].append(mask_data_left)

                    labels_left[patient_id].append(calculate_label(mask_data_left))

        # Carrega as imagens e máscaras do lado direito
        for patch_id, mask_id in zip(os.listdir(path_right), os.listdir(lesion_path_right)):
            img_path = os.path.join(path_right, patch_id)
            mask_path = os.path.join(lesion_path_right, mask_id)
            for img_path_right, mask_path_right in zip(os.listdir(img_path), os.listdir(mask_path)):
                image_data_right = nib.load(os.path.join(img_path, img_path_right)).get_fdata()
                mask_data_right = nib.load(os.path.join(mask_path, mask_path_right)).get_fdata()
                if (len(image_data_right) > 0 and image_data_right is not []) or (len(mask_data_right) > 0 and mask_data_right is not []):
                    images_right[patient_id].append(image_data_right)
                    mask_right[patient_id].append(mask_data_right)
                    
                    labels_right[patient_id].append(calculate_label(mask_data_right))
        patient_ids.append(patient_id)

    # Estruturas para armazenar os pares de labels
    labels_pair = {}
    for patient_id,_ in zip(labels_left.keys(), labels_right.keys()):
        labels_pair[patient_id] = []
        for label_left, label_right in zip(labels_left[patient_id], labels_right[patient_id]): 
            if label_left == 0 and label_right == 0:
                labels_pair[patient_id].append(0)
            else:
                labels_pair[patient_id].append(1)

    print(f"Total de pacientes: {len(patient_ids)}")
    for patient_id, labels in labels_pair.items():
        print(f"Paciente {patient_id}: Total de pares de recortes: {len(labels)}")

    return images_left, images_right, labels_pair, mask_left, mask_right, patient_ids

def build_image(img, mask):
    # Tamanho dos patches
    patch_size = 96

    # Inicializar a matriz para a imagem e a máscara reconstruídas
    imagem_reconstruida = np.zeros((4 * patch_size, 2 * patch_size//2), dtype=np.uint8)
    mascara_reconstruida = np.zeros((4 * patch_size, 2 * patch_size//2), dtype=np.uint8)

    # Ordem correta para reconstrução
    correct_order = [
        0, 1, 8, 9,    # Linha 1
        2, 3, 10, 11,  # Linha 2
        4, 5, 12, 13,  # Linha 3
        6, 7, 14, 15   # Linha 4
    ]

    # Loop para reconstruir as linhas e colunas
    for i in range(4):  # Linha
        for j in range(2):  # Coluna
            idx = correct_order[i * 2 + j]
            x_start = i * patch_size 
            y_start = j * patch_size//2
            
            imagem_reconstruida[x_start:x_start + patch_size, y_start:y_start + patch_size//2] = img[idx]
            mascara_reconstruida[x_start:x_start + patch_size, y_start:y_start + patch_size//2] = mask[idx]
            
    return imagem_reconstruida, mascara_reconstruida

def plot_patient_slices(pdf_filename, patients, images_left, images_right, mask_left, mask_right):
    with PdfPages(pdf_filename) as pdf:
            for patient in patients:
                cont = 0            
                
                classificacao = []
                vetor_left_img = []
                vetor_left_mask = []
                vetor_right_img = []
                vetor_right_mask = []
                tmp_vetor_left_img = []
                tmp_vetor_left_mask = []
                tmp_vetor_right_img = []
                tmp_vetor_right_mask = []
                
                for img_left, img_right, msk_left, msk_right in zip(images_left[patient], images_right[patient], mask_left[patient], mask_right[patient]):
                    tmp_vetor_left_img.append(img_left)
                    tmp_vetor_left_mask.append(msk_left)
                    tmp_vetor_right_img.append(np.fliplr(img_right))
                    tmp_vetor_right_mask.append(np.fliplr(msk_right))
                
                    cont+=1
                    if cont%8 ==0:
                        vetor_left_img.append(tmp_vetor_left_img)
                        vetor_left_mask.append(tmp_vetor_left_mask)
                        vetor_right_img.append(tmp_vetor_right_img)
                        vetor_right_mask.append(tmp_vetor_right_mask)
                        tmp_vetor_left_img = []
                        tmp_vetor_left_mask = []
                        tmp_vetor_right_img = []
                        tmp_vetor_right_mask = []
                
                cont = 0
                for i in range(len(vetor_left_img)):
                    imagem_reconstruida, mascara_reconstruida = build_image(vetor_left_img[i]+vetor_right_img[i], vetor_left_mask[i]+vetor_right_mask[i])
                    if (np.any(mascara_reconstruida) == 1):
                        # Configurar a figura
                        fig, axs = plt.subplots(2, 1, figsize=(4, 4))

                        axs[0].imshow(np.flipud(imagem_reconstruida), cmap='gray')
                        axs[0].set_title(f'{patient}')
                        axs[0].axis('off')
                        axs[1].imshow(np.flipud(mascara_reconstruida), cmap='gray')
                        axs[1].axis('off')
                        
                        # Adicionar ao PDF
                        pdf.savefig(fig)
                        plt.close(fig)
                
            print(f"As imagens foram salvas no arquivo PDF {pdf_filename} com sucesso.")

folder = "Contralateral"
pdf_filename="Pacientes_Reconstruidos.pdf"
images_left_by_patient, images_right_by_patient, labels_pair_by_patient, mask_left_by_patient, mask_right_by_patient, patient_ids = load_data_with_pairs(folder)
plot_patient_slices(
    pdf_filename=pdf_filename,
    patients=patient_ids,
    images_left=images_left_by_patient,
    images_right=images_right_by_patient,
    mask_left=mask_left_by_patient,
    mask_right=mask_right_by_patient,
)