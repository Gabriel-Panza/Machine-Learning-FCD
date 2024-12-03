import nibabel as nib
import os
import numpy as np

paciente_com_mascara = "Pacientes com Mascara"
saida_mascara = "Mascaras"

for pac_masc in os.listdir(paciente_com_mascara):
    patient_mask = nib.load(os.path.join(paciente_com_mascara, pac_masc)).get_fdata()
    patient_mask = np.where(patient_mask < 0.75, 0, 1) 
    
    nova_imagem = nib.Nifti1Image(patient_mask, affine=np.eye(4)) 
    
    pac_masc = pac_masc.split(".")
    caminho_saida = os.path.join(saida_mascara, f"{pac_masc[0]}_roi.nii.gz")
    nib.save(nova_imagem, caminho_saida)