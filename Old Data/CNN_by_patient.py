import nibabel as nib
import numpy as np
import os
import random
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

# Função que carrega os dados
def load_data(folder, csv_file_path):
    # Verificar se a pasta existe
    if not os.path.exists(folder):
        print(f"A pasta {folder} não existe.")
        return {}, {}

    # Abre o arquivo .csv
    df = pd.read_csv(csv_file_path)

    # Inicializar dicionário para armazenar imagens e labels por grupo de paciente
    images_by_group = {}
    labels_by_group = {}
    
    # Iterar por cada linha do DataFrame
    current_patient_id = None

    for _, row in df.iterrows():
        # Obter o nome do paciente e o label daquela linha
        patient_id = row['Nome_P']
        label = int(row['label'])

        # Se o paciente mudou, cria-se uma nova lista dentro da matriz de labels
        if patient_id != current_patient_id:
            current_patient_id = patient_id
            labels_by_group[current_patient_id] = []

        # Adicionar o label na lista do paciente correspondente
        labels_by_group[current_patient_id].append(label)

    # Listar todos os arquivos do diretório
    files = os.listdir(folder)

    # Separar arquivos de máscara e recorte
    mask_files = sorted([f for f in files if 'mascara' in f])
    crop_files = sorted([f for f in files if 'recorte' in f])

    # Iterar sobre as máscaras e recortes correspondentes
    i=0
    for mask_file, crop_file in tqdm(zip(mask_files, crop_files), desc="Carregamento de arquivos NIfTI..."):        
        # Extrair o identificador do paciente a partir do nome do arquivo (assumindo que os IDs começam após 'sub-' e têm 6 caracteres)
        patient_id = crop_file.split('_')[0]

        # Se o grupo do paciente ainda não está no dicionário, inicializar
        if patient_id not in images_by_group:
            images_by_group[patient_id] = []

        # Carregar os arquivos NIfTI (.nii.gz)
        try:
            crop_img = nib.load(os.path.join(folder, crop_file)).get_fdata()
        except Exception as e:
            print(f"Erro ao carregar {crop_file}: {e}")
            continue

        # Expandir o patch_crop para ter um canal (necessário para CNN 2D)
        crop_img = np.expand_dims(crop_img, axis=-1)  # Agora (50, 50, 1)

        # Adicionar o patch de recorte e o label às listas do paciente
        images_by_group[patient_id].append(crop_img)

    # Verificar as formas finais
    print(f"Total de grupos de pacientes: {len(images_by_group)}")
    for patient_id, images in images_by_group.items():
        print(f"Paciente {patient_id}: Total de recortes: {len(images)}")
    
    return images_by_group, labels_by_group

# Função para concatenar as listas de pacientes em arrays únicos para treino e teste
def prepare_data_for_training_with_undersampling(images_by_group, labels_by_group, train_size=0.7, validation_size=0.1, test_size=0.2):
    # Obter o número mínimo de recortes entre os pacientes
    min_patches = min([len(images) for images in images_by_group.values()])
    print(f"Número mínimo de recortes por paciente: {min_patches}")

    # Inicializar listas para armazenar as imagens e labels balanceadas por paciente
    balanced_images_by_group = {}
    balanced_labels_by_group = {}

    # Para cada grupo de pacientes, balancear a quantidade de recortes para o número mínimo
    for patient_id in images_by_group:
        # Verificar se o paciente tem mais recortes do que o mínimo
        if len(images_by_group[patient_id]) > min_patches:
            # Fazer undersampling aleatório para trazer para o número mínimo
            indices = random.sample(range(len(images_by_group[patient_id])), min_patches)
            balanced_images_by_group[patient_id] = [images_by_group[patient_id][i] for i in indices]
            balanced_labels_by_group[patient_id] = [labels_by_group[patient_id][i] for i in indices]
        else:
            # Se o paciente já tem o número mínimo ou menos, manter os dados originais
            balanced_images_by_group[patient_id] = images_by_group[patient_id]
            balanced_labels_by_group[patient_id] = labels_by_group[patient_id]

    # Obter lista de IDs de pacientes (chaves do dicionário)
    patient_ids = list(balanced_images_by_group.keys())

    # Dividir os pacientes entre treino, validação e teste
    train_ids, test_val_ids = train_test_split(patient_ids, train_size=train_size, test_size=(validation_size + test_size))
    validation_ids, test_ids = train_test_split(test_val_ids, test_size=(test_size / (test_size + validation_size)))

    # Inicializar listas para armazenar as imagens e labels de treino, validação e teste
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []

    # Para cada grupo de treino, adicionar os dados ao conjunto de treino
    for patient_id in train_ids:
        for i in range(len(balanced_images_by_group[patient_id])):
            X_train.append(balanced_images_by_group[patient_id][i])
            y_train.append(balanced_labels_by_group[patient_id][i])

    # Para cada grupo de validação, adicionar os dados ao conjunto de validação
    for patient_id in validation_ids:
        for i in range(len(balanced_images_by_group[patient_id])):
            X_val.append(balanced_images_by_group[patient_id][i])
            y_val.append(balanced_labels_by_group[patient_id][i])

    # Para cada grupo de teste, adicionar os dados ao conjunto de teste
    for patient_id in test_ids:
        for i in range(len(balanced_images_by_group[patient_id])):
            X_test.append(balanced_images_by_group[patient_id][i])
            y_test.append(balanced_labels_by_group[patient_id][i])

    # Imprimir informações
    print(f"Total de grupos de pacientes: {len(balanced_images_by_group)}")
    for patient_id, images in balanced_images_by_group.items():
        print(f"Paciente {patient_id}: Total de recortes após balancear: {len(images)}")
        
    return np.array(X_train), np.array(X_val), np.array(X_test), np.array(y_train), np.array(y_val), np.array(y_test)

# Função para construir o modelo CNN 2D
def build_cnn_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, 3, data_format="channels_last", activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Conv2D(64, 3, data_format="channels_last", activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Conv2D(128, 3, data_format="channels_last", activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Função para plotar gráficos de loss e accuracy
def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

# Definir caminho e forma
input_folder = 'recortes_T1_50x50'
arquivo_csv = 'Base_Informações/Base_informaçΣes_50x50_T1.csv'

# Carregar os dados (utilizando a função load_data modificada)
X,y = load_data(input_folder, arquivo_csv)

# Separar os dados em treino, validação e teste
X_train, X_valid, X_test, y_train, y_valid, y_test = prepare_data_for_training_with_undersampling(X, y, train_size=0.7, validation_size=0.1, test_size=0.2)

# Construir e compilar o modelo CNN
cnn_model = build_cnn_model(X_train[0].shape)
cnn_model.summary()

# Treinamento do modelo
history = cnn_model.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size=64, epochs=50)

# Avaliar o modelo
loss, accuracy = cnn_model.evaluate(X_test, y_test)
print(f'Loss no teste: {loss:.4f}, Accuracy no teste: {accuracy:.4f}')

# Plotar o histórico do treinamento
plot_training_history(history)