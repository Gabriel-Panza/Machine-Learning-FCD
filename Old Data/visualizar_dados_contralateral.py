import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

def save_nii_gz(file_path, patient, patch, type, dupla_index):
    # Verificar se o arquivo existe
    if not os.path.exists(file_path):
        print(f"Arquivo não encontrado: {file_path}")
        return

    try:
        # Carregar a imagem NIfTI
        nifti_img = nib.load(file_path)
    except Exception as e:
        print(f"Erro ao carregar o arquivo NIfTI: {file_path}. Erro: {e}")
        return
    
    # Obter os dados da imagem como um array numpy
    data = nifti_img.get_fdata()
    np.set_printoptions(threshold=np.inf)
    
    if type == 1:
        data = np.flipud(data)
    
    # Criar a pasta de saída, se não existir
    output_dir = f"Images/Paciente_{patient}/par_{dupla_index}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Exibir a imagem com o colormap gray
    plt.figure(figsize=(6, 6))
    plt.imshow(data, cmap='gray')
    plt.title(f'Paciente {patient} | Patch {patch}')
    
    # Salvar o plot como PNG na pasta criada
    file_name = f"Paciente_{patient}_Patch_{patch}.png"
    save_path = os.path.join(output_dir, file_name)
    plt.savefig(save_path, format='png')

    plt.close()

def find_image_paths(patch, type, path):
    possible_patterns = [
        f'recortes_T1_50x50/sub-02A13_{patch}_{type}_T1.nii.gz',
        f'recortes_T1_50x50/sub-03C08_{patch}_{type}_T1.nii.gz',
        f'recortes_T1_50x50/sub-16G09_{patch}_{type}_T1.nii.gz',
        f'recortes_T1_50x50/sub-25B08_{patch}_{type}_T1.nii.gz',
        f'recortes_T1_50x50/sub-41D08_{patch}_{type}_T1.nii.gz',
        f'recortes_T1_50x50/sub-44H05_{patch}_{type}_T1.nii.gz',
    ]
        
    patient = 0
    for pattern in possible_patterns:
        patient += 1
        if os.path.exists(pattern):
            path = pattern
            break

    return path, patient

df_pairs = pd.read_csv('Base_Informações/Base_informações_com_previsao_50x50_T1.csv')
dupla = 0
oposto = []

# Iterar sobre cada linha do DataFrame df_pairs
for index, row in tqdm(df_pairs.iterrows(), desc="Carregamento de arquivos com contra-lateral..."):
    # Extrair o valor de 'patch_identificacao' e 'patch_oposto'
    patch = row['patch_identificacao']
    patch_oposto = row['patch_oposto']

    # Identificar o paciente pelo nome
    patient_name = row['Nome_P']
    
    if patient_name in ["sub-02A13", "sub-03C08"]:
        continue
    
    oposto.append(patch_oposto)
    if patch in oposto:
        continue
    
    dupla+=1
    # Caminho para o arquivo .nii.gz gerado
    path_crop, patient_crop = find_image_paths(patch, "recorte", path=None)
    path_crop_opposite, patient_crop_opposite = find_image_paths(patch_oposto, "recorte", path=None)

    if path_crop is None:
        print(f"Paciente {patient_name} não tem contra-lateral!")
        continue

    # Verificar se o caminho do patch oposto foi encontrado
    if path_crop_opposite is None:
        print(f"Patch oposto para o paciente {patient_name} não encontrado!")
        continue
    
    # Visualizar o patch original
    save_nii_gz(f"{path_crop}", patient_name, patch, 0, dupla)

    # Visualizar o patch oposto
    save_nii_gz(f"{path_crop_opposite}", patient_name, patch_oposto, 1, dupla)

print("Processo de visualização completo para todos os pacientes.")