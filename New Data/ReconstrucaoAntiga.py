import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
from skimage import measure

def calculate_label(image, threshold=0.01):
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

def load_patient_data(folder, patient_id):
    """
    Carrega os dados de um único paciente (imagens, máscaras e labels) de um diretório.

    Args:
        folder (str): Caminho da pasta contendo os dados dos pacientes.
        patient_id (str): ID do paciente a ser carregado.

    Returns:
        dict: Dados do paciente, incluindo imagens, máscaras e labels para os lados esquerdo e direito.
              Retorna None se o paciente não for encontrado.
    """
    patient_path = os.path.join(folder, patient_id)
    if not os.path.exists(patient_path):
        print(f"Paciente {patient_id} não encontrado na pasta {folder}.")
        return None

    # Inicializa estruturas para armazenar os dados do paciente
    patient_data = {
        "images_left": [],
        "images_right": [],
        "mask_left": [],
        "mask_right": [],
        "labels_left": [],
        "labels_right": [],
    }

    areas_image = ["left", "right"]
    areas_mask = ["lesion_left", "lesion_right"]
    path_left = os.path.join(patient_path, areas_image[0])
    path_right = os.path.join(patient_path, areas_image[1])
    lesion_path_left = os.path.join(patient_path, areas_mask[0])
    lesion_path_right = os.path.join(patient_path, areas_mask[1])

    # Verifica se os diretórios existem
    if not os.path.exists(path_left) or not os.path.exists(path_right) or \
       not os.path.exists(lesion_path_left) or not os.path.exists(lesion_path_right):
        print(f"Estrutura de diretórios inválida para o paciente {patient_id}.")
        return None

    # Carrega as imagens e máscaras do lado esquerdo e direito
    for patch_id_left, mask_id_left, patch_id_right, mask_id_right in zip(
        os.listdir(path_left), os.listdir(lesion_path_left),
        os.listdir(path_right), os.listdir(lesion_path_right)
    ):
        img_path_left = os.path.join(path_left, patch_id_left)
        mask_path_left = os.path.join(lesion_path_left, mask_id_left)
        img_path_right = os.path.join(path_right, patch_id_right)
        mask_path_right = os.path.join(lesion_path_right, mask_id_right)

        for img_left, msk_left, img_right, msk_right in zip(
            os.listdir(img_path_left), os.listdir(mask_path_left),
            os.listdir(img_path_right), os.listdir(mask_path_right)
        ):
            # Carrega os dados do lado esquerdo
            data_left = nib.load(os.path.join(img_path_left, img_left)).get_fdata()
            data_msk_left = nib.load(os.path.join(mask_path_left, msk_left)).get_fdata()
            if len(data_left) > 0 or len(data_msk_left) > 0:
                patient_data["images_left"].append(data_left)
                patient_data["mask_left"].append(data_msk_left)
                patient_data["labels_left"].append(calculate_label(data_msk_left))

            # Carrega os dados do lado direito
            data_right = nib.load(os.path.join(img_path_right, img_right)).get_fdata()
            data_msk_right = nib.load(os.path.join(mask_path_right, msk_right)).get_fdata()
            if len(data_right) > 0 or len(data_msk_right) > 0:
                patient_data["images_right"].append(data_right)
                patient_data["mask_right"].append(data_msk_right)
                patient_data["labels_right"].append(calculate_label(data_msk_right))

    # Gera os pares de labels
    labels_pair = []
    for label_left, label_right in zip(patient_data["labels_left"], patient_data["labels_right"]):
        if label_left == 0 and label_right == 0:
            labels_pair.append(0)
        else:
            labels_pair.append(1)
    patient_data["labels_pair"] = labels_pair

    print(f"Paciente {patient_id} carregado com sucesso.")
    print(f"Total de recortes: {len(labels_pair)}")
    return patient_data

def highlight_lesions(image, mask):
    """
    Destaca as áreas de lesão na imagem, desenhando contornos azuis ao redor das máscaras.
    Para áreas sem lesão, desenha contornos vermelhos.
    """
    highlighted_image = np.copy(image)
    
    # Encontra os contornos das máscaras
    contours = measure.find_contours(mask, 0.5)
    
    # Define a cor com base na presença de lesão
    if calculate_label(mask):  # Verifica se é lesão
        color = np.array([0, 0, 255])  # Azul para lesões (em RGB)
    else:
        color = np.array([255, 0, 0])  # Vermelho para áreas sem lesão (em RGB)
    
    # Desenha os contornos na imagem
    for contour in contours:
        for i in range(len(contour) - 1):
            y1, x1 = map(int, contour[i])
            y2, x2 = map(int, contour[i + 1])
            # Desenha uma linha entre os pontos do contorno
            highlighted_image[y1:y2, x1:x2] = color
    
    return highlighted_image

def build_image(img, mask):
    # Juntar lado esquerdo (4 pedaços)
    left_side = np.vstack([img[0], img[1], img[2], img[3]])
    left_side_grid = np.vstack([highlight_lesions(img[0], mask[0]), highlight_lesions(img[1], mask[1]), highlight_lesions(img[2], mask[2]), highlight_lesions(img[3], mask[3])])
    # Juntar lado direito (4 pedaços)
    right_side = np.vstack([img[4], img[5], img[6], img[7]])
    right_side_grid = np.vstack([highlight_lesions(img[4], mask[4]), highlight_lesions(img[5], mask[5]), highlight_lesions(img[6], mask[6]), highlight_lesions(img[7], mask[7])])[:, ::-1]
    
    # Juntar lado esquerdo (4 pedaços)
    left_side_mask_grid = np.vstack([highlight_lesions(mask[0], mask[0]), highlight_lesions(mask[1], mask[1]), highlight_lesions(mask[2], mask[2]), highlight_lesions(mask[3], mask[3])])
    left_side_mask = np.vstack([mask[0], mask[1], mask[2], mask[3]])
    # Juntar lado direito (4 pedaços)
    right_side_mask_grid = np.vstack([highlight_lesions(mask[4], mask[4]), highlight_lesions(mask[5], mask[5]), highlight_lesions(mask[6], mask[6]), highlight_lesions(mask[7], mask[7])])[:, ::-1]
    right_side_mask = np.vstack([mask[4], mask[5], mask[6], mask[7]])
    
    return np.hstack([left_side, right_side]) , np.hstack([left_side_mask, right_side_mask]), np.hstack([left_side_grid, right_side_grid]) , np.hstack([left_side_mask_grid, right_side_mask_grid])

def plot_patient_slices(pdf_filename, folder):
    """
    Gera um PDF com as fatias reconstruídas de cada paciente, destacando as lesões.
    """
    with PdfPages(pdf_filename) as pdf:
        for patient_id in tqdm(os.listdir(folder), desc="Processando pacientes..."):
            patient_data = load_patient_data(folder, patient_id)
            if patient_data is None:
                continue

            images_left = patient_data["images_left"]
            images_right = patient_data["images_right"]
            mask_left = patient_data["mask_left"]
            mask_right = patient_data["mask_right"]

            cont = 0            
            vetor_left_img = []
            vetor_left_mask = []
            vetor_right_img = []
            vetor_right_mask = []
            tmp_vetor_left_img = []
            tmp_vetor_left_mask = []
            tmp_vetor_right_img = []
            tmp_vetor_right_mask = []
            
            for img_left, img_right, msk_left, msk_right in zip(images_left, images_right, mask_left, mask_right):
                tmp_vetor_left_img.append(img_left)
                tmp_vetor_left_mask.append(msk_left)
                tmp_vetor_right_img.append(img_right)
                tmp_vetor_right_mask.append(msk_right)
            
                cont += 1
                if cont % 8 == 0:
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
                imagem_reconstruida, mascara_reconstruida, imagem_grid, mascara_grid = build_image(vetor_left_img[i] + vetor_right_img[i], vetor_left_mask[i] + vetor_right_mask[i])
                
                # Configurar a figura
                fig, axs = plt.subplots(2, 1, figsize=(8, 8))

                axs[0].imshow(imagem_reconstruida, cmap='gray')
                axs[0].imshow(imagem_grid)
                axs[0].set_title(f'{patient_id} - Imagem Reconstruída')
                axs[0].axis('off')

                axs[1].imshow(mascara_reconstruida, cmap='gray')
                axs[1].imshow(mascara_grid)
                axs[1].set_title(f'{patient_id} - Lesões Destacadas')
                axs[1].axis('off')
                                
                # Adicionar ao PDF
                pdf.savefig(fig)
                plt.close(fig)
        
    print(f"As imagens foram salvas no arquivo PDF {pdf_filename} com sucesso.")

# Caminho da pasta contendo os dados dos pacientes
folder = "Contralateral"
pdf_filename = "Pdf/Pacientes_Reconstruidos.pdf"

# Gera o PDF com as fatias reconstruídas
plot_patient_slices(pdf_filename, folder)