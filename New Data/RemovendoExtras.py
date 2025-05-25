import os
import shutil

# Pastas principais
base_path = "Novo_Contralateral"
modalities = ["Contralateral_T1", "Contralateral_Flair", "Contralateral_T2"]

# Coleta os slices válidos por paciente em Contralateral_T2
valid_slices_per_patient = {}

path_T2 = os.path.join(base_path, "Contralateral_T2")
for patient_id in os.listdir(path_T2):
    path_left = os.path.join(path_T2, patient_id, "left")
    if not os.path.isdir(path_left):
        continue

    slices = [f for f in os.listdir(path_left) if f.startswith("Slice_")]
    valid_slices_per_patient[patient_id] = set(slices)

# Agora remove slices não existentes em T2 nas outras modalidades
for modality in ["Contralateral_T1", "Contralateral_Flair"]:
    path_mod = os.path.join(base_path, modality)
    for patient_id in os.listdir(path_mod):
        if patient_id not in valid_slices_per_patient:
            continue

        valid_slices = valid_slices_per_patient[patient_id]

        for side in ["left", "lesion_left", "right", "lesion_right"]:
            path_side = os.path.join(path_mod, patient_id, side)
            if not os.path.isdir(path_side):
                continue

            for slice_file in os.listdir(path_side):
                if slice_file not in valid_slices:
                    slice_path = os.path.join(path_side, slice_file)
                    print(f"Removendo {slice_path}")
                    shutil.rmtree(slice_path)  # Remove pasta Slice_XXX inteira