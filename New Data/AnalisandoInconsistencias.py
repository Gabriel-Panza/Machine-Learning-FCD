import os
import shutil
import numpy as np
import nibabel as nb

base_path = "Novo_Contralateral"
modalities = ["Contralateral_T1", "Contralateral_Flair", "Contralateral_T2"]
sides = ["left", "lesion_left", "right", "lesion_right"]

# Função auxiliar para verificar se há lesão (pixel == 1) em algum arquivo dentro de um Slice
def has_lesion(slice_path):
    if not os.path.isdir(slice_path):
        return False

    for file in os.listdir(slice_path):
        file_path = os.path.join(slice_path, file)
        if os.path.isfile(file_path):
            try:
                # Abrir imagem e verificar se existe pixel com valor 1
                img = nb.load(file_path).get_fdata()
                arr = np.array(img)
                if np.any(arr == 1):
                    return True
            except Exception as e:
                print(f"Erro ao abrir {file_path}: {e}")
    return False

# Primeira etapa: Coletar todos os IDs de pacientes únicos
all_patient_ids = set()
print("Procurando por todos os IDs de pacientes...")
for modality_scan in modalities:
    mod_scan_path = os.path.join(base_path, modality_scan)
    if os.path.isdir(mod_scan_path):
        try:
            for patient_id_scan in os.listdir(mod_scan_path):
                if os.path.isdir(os.path.join(mod_scan_path, patient_id_scan)):
                    all_patient_ids.add(patient_id_scan)
        except OSError as e:
            print(f"  Erro ao escanear modalidade {modality_scan}: {e}")
all_patient_ids = sorted(list(all_patient_ids))
print(f"Encontrados {len(all_patient_ids)} ID(s) de paciente(s) único(s): {all_patient_ids}\n")

# Segunda etapa: remove slices inconsistentes entre modalidades
for patient_id in all_patient_ids:
    print(f"Paciente: {patient_id}")
    for side_of_body in ["left", "lesion_left", "right", "lesion_right"]:
        slice_counts = {}

        for modality in modalities:
            path_modality_patient_side = os.path.join(base_path, modality, patient_id, side_of_body)
            
            if not os.path.isdir(path_modality_patient_side):
                continue

            for slice_name in os.listdir(path_modality_patient_side):
                current_slice_dir_path = os.path.join(path_modality_patient_side, slice_name)
                
                if not os.path.isdir(current_slice_dir_path):
                    continue

                try:
                    file_count = len(os.listdir(current_slice_dir_path))
                except OSError as e:
                    print(f"    Erro ao listar arquivos em {current_slice_dir_path}: {e}. Pulando este slice.")
                    continue
                    
                if slice_name not in slice_counts:
                    slice_counts[slice_name] = []
                
                # Store details: modality, the side being processed, path to slice files, and file count
                slice_counts[slice_name].append((modality, side_of_body, current_slice_dir_path, file_count))

        # Check for inconsistencies and take action
        if not slice_counts:
            print(f"    Nenhum slice encontrado ou processado para o lado '{side_of_body}'.")
            continue

        for slice_name, data_for_slice in slice_counts.items():
            counts = [item[3] for item in data_for_slice]
            
            if len(counts) < 2:
                continue

            if len(set(counts)) > 1: # Alguma inconsistência existe
                min_val = min(counts)
                max_val = max(counts)
                unique_counts_set = set(counts)

                print(f"    ⚠ Inconsistência de lesão detectada para slice '{slice_name}' (patient: {patient_id}). Contagens: {counts}.")
        break
    print("\n------------------------------\n")
print("Processamento concluído.")