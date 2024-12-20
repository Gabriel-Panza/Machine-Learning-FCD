import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
import nrrd
from matplotlib.backends.backend_pdf import PdfPages

def plot_slices_with_masks(data, mask, slice_idx, pdf):
    # Garantir que o índice do slice está dentro dos limites
    total_slices = data.shape[2]
    slice_previous = max(slice_idx - 1, 0)
    slice_current = slice_idx
    slice_next = min(slice_idx + 1, total_slices - 1)

    # Obter os slices
    data_previous = data[:, :, slice_previous]
    data_current = data[:, :, slice_current]
    data_next = data[:, :, slice_next]

    mask_previous = mask[:, :, slice_previous]
    mask_current = mask[:, :, slice_current]
    mask_next = mask[:, :, slice_next]

    # Criar o plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plots dos slices originais
    axes[0, 0].imshow(data_previous, cmap='gray')
    axes[0, 0].set_title(f"Slice Anterior ({slice_previous})")

    axes[0, 1].imshow(data_current, cmap='gray')
    axes[0, 1].set_title(f"Slice Atual ({slice_current})")

    axes[0, 2].imshow(data_next, cmap='gray')
    axes[0, 2].set_title(f"Slice Posterior ({slice_next})")

    # Plots das máscaras
    axes[1, 0].imshow(mask_previous, cmap='gray')
    axes[1, 0].set_title(f"Máscara Anterior ({slice_previous})")

    axes[1, 1].imshow(mask_current, cmap='gray')
    axes[1, 1].set_title(f"Máscara Atual ({slice_current})")

    axes[1, 2].imshow(mask_next, cmap='gray')
    axes[1, 2].set_title(f"Máscara Posterior ({slice_next})")

    # Ajustar layout
    for ax in axes.flat:
        ax.axis('off')
    plt.tight_layout()
    
    # Salvar o plot no diretório especificado
    pdf.savefig(fig)
    
    plt.close(fig)

# Diretórios de entrada e saída
imagens = "Patients_Displasya"
mascara = "Mascaras"
output_pdf_path = "Pdf/Plot_Slices.pdf"

# Criar o arquivo PDF para salvar os plots
with PdfPages(output_pdf_path) as pdf:
    # Iterar sobre as imagens e máscaras
    for img, mask in zip([f for f in os.listdir(imagens) if f.endswith(('.nii', '.nii.gz'))], [f for f in os.listdir(mascara) if f.endswith(('.nrrd', '.nii', '.nii.gz'))]):    
        data = nib.load(os.path.join(imagens, img)).get_fdata()
        lesion_data, _ = nrrd.read(os.path.join(mascara, mask))
        
        data = np.transpose(data, (2, 0, 1))
        lesion_data = np.transpose(lesion_data, (2, 0, 1))
        
        #print(data.shape)
        #print(lesion_data.shape)
        
        if (lesion_data.shape[2]>data.shape[2]):
            continue

        # Gerar plots para cada slice
        for slice_idx in range(30, lesion_data.shape[2],3):
            plot_slices_with_masks(data, lesion_data, slice_idx, pdf)

        print(f"Plots gerados para {img.split('_')[0]}")
print(f"PDF salvo em: {output_pdf_path}")