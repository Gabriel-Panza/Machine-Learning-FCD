import nibabel as nib
import numpy as np
import os
import random
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, metrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
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
    
    # Definir IDs dos pacientes a serem ignorados
    ignored_patients = ['sub-00H10','sub-02A13', 'sub-03C08','sub-06C09']
    
    # Iterar por cada linha do DataFrame
    current_patient_id = None
    for _, row in df.iterrows():
        # Obter o nome do paciente e o label daquela linha
        patient_id = row['Nome_P']
        label = int(row['label'])

        if patient_id in ignored_patients:
            continue
        
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

        if patient_id in ignored_patients:
            continue
        
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

# Função para aumentar as imagens da classe minoritária
def augment_data(images, labels, augment_factor=2):
    # Criar um ImageDataGenerator com transformações aleatórias
    datagen = ImageDataGenerator(
        rotation_range=20,  # Rotação aleatória até 20 graus
        width_shift_range=0.2,  # Deslocamento horizontal aleatório até 20% da largura
        height_shift_range=0.2,  # Deslocamento vertical aleatório até 20% da altura
        horizontal_flip=True,  # Flip horizontal aleatório
        fill_mode='nearest'  # Preencher os pixels faltantes com o valor mais próximo
    )

    augmented_images = []
    augmented_labels = []

    for img, label in zip(images, labels):
        img = np.expand_dims(img, axis=0)  # Expande para o formato (1, altura, largura, canais)
        augment_iter = datagen.flow(img, batch_size=1)
        
        # Gerar mais imagens de augmentação
        for _ in range(augment_factor):
            augmented_img = next(augment_iter)[0]  # Extrai a imagem aumentada
            augmented_images.append(augmented_img)
            augmented_labels.append(label)

    return np.array(augmented_images), np.array(augmented_labels)

# Função auxiliar para concatenar imagens e labels por grupo de pacientes
def concat_data(balanced_images_by_patient, balanced_labels_by_patient, patients):
    images = []
    labels = []
    for patient_id in patients:
        images.extend(balanced_images_by_patient[patient_id])
        labels.extend(balanced_labels_by_patient[patient_id])
    return np.array(images), np.array(labels)

# Função para concatenar as listas de pacientes em arrays únicos para treino, validação e teste
def prepare_data_for_training_balanced(images_by_group, labels_by_group, train_size=0.7, validation_size=0.2, test_size=0.1, augment_factor=2):
    # Listas para armazenar as amostras balanceadas
    balanced_images_by_patient = {}
    balanced_labels_by_patient = {}

    # Para cada paciente, balancear o número de amostras da classe 0 e 1
    for patient_id in images_by_group:
        # Coletar imagens e labels por classe
        class_1_images = []
        class_0_images = []
        class_1_labels = []
        class_0_labels = []

        for i in range(len(labels_by_group[patient_id])):
            if labels_by_group[patient_id][i] == 1:
                class_1_images.append(images_by_group[patient_id][i])
                class_1_labels.append(labels_by_group[patient_id][i])
            else:
                class_0_images.append(images_by_group[patient_id][i])
                class_0_labels.append(labels_by_group[patient_id][i])
        
        # Quantidade de exemplos na classe minoritária (label 1) por paciente
        class_1_count = len(class_1_images)
        print(f"Total de labels 1: {class_1_count}")

        # Fazer undersampling da classe majoritária (label 0) para igualar ao número de exemplos da classe 1
        class_0_indices = random.sample(range(len(class_0_images)), min(class_1_count, len(class_0_images)))
        class_0_images_sampled = [class_0_images[i] for i in class_0_indices]
        class_0_labels_sampled = [class_0_labels[i] for i in class_0_indices]

        # Adicionar imagens da classe 1 (aumentadas) e da classe 0 (originais)
        balanced_images_by_patient[patient_id] = list(class_1_images) + class_0_images_sampled
        balanced_labels_by_patient[patient_id] = list(class_1_labels) + class_0_labels_sampled
    
    for patient_id, images in balanced_images_by_patient.items():
        print(f"Paciente {patient_id}: Total de recortes: {len(images)}")
        
    # Coletar todos os IDs de pacientes
    all_patient_ids = list(balanced_images_by_patient.keys())

    # Dividir os pacientes em treino, validação e teste
    train_patients, test_val_patients = train_test_split(all_patient_ids, train_size=train_size, random_state=42)
    val_patients, test_patients = train_test_split(test_val_patients, train_size=validation_size/(validation_size + test_size), random_state=42)

    # Separar os dados por conjunto
    X_train, y_train = concat_data(balanced_images_by_patient, balanced_labels_by_patient, train_patients)
    X_val, y_val = concat_data(balanced_images_by_patient, balanced_labels_by_patient, val_patients)
    X_test, y_test = concat_data(balanced_images_by_patient, balanced_labels_by_patient, test_patients)

    # Aplicar data augmentation em cada conjunto
    X_train, y_train = augment_data(X_train, y_train, augment_factor=augment_factor)
    X_val, y_val = augment_data(X_val, y_val, augment_factor=augment_factor)
    X_test, y_test = augment_data(X_test, y_test, augment_factor=augment_factor)
    
    # Contagem das labels no conjunto de treino, validação e teste
    print(f"Total de recortes no treino com label 1: {sum(y_train == 1)}")
    print(f"Total de recortes no treino com label 0: {sum(y_train == 0)}")
    print(f"Total de recortes na validação com label 1: {sum(y_val == 1)}")
    print(f"Total de recortes na validação com label 0: {sum(y_val == 0)}")
    print(f"Total de recortes no teste com label 1: {sum(y_test == 1)}")
    print(f"Total de recortes no teste com label 0: {sum(y_test == 0)}")

    return X_train, X_val, X_test, y_train, y_val, y_test

# Função para construir o modelo CNN 2D
def build_cnn_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3,3), data_format="channels_last", activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.5))
        
    model.add(layers.Conv2D(128, (3,3), data_format="channels_last", activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.5))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy', metrics.Precision(name="precision"), metrics.Recall(name="recall"), metrics.AUC(name="auc")])
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

# Função para plotar a matriz de confusão
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

# Definir caminho e forma
input_folder = 'recortes_T1_50x50'
arquivo_csv = 'Base_Informações/Base_informaçΣes_50x50_T1.csv'

# Carregar os dados (utilizando a função load_data modificada)
X,y = load_data(input_folder, arquivo_csv)

# Separar os dados em treino, validação e teste
X_train, X_valid, X_test, y_train, y_valid, y_test = prepare_data_for_training_balanced(X, y, train_size=0.7, validation_size=0.2, test_size=0.1)
X_train = X_train/255
X_valid = X_valid/255 
X_test = X_test/255

# Construir e compilar o modelo CNN
cnn_model = build_cnn_model(X_train[0].shape)
cnn_model.summary()

# Adicionar o callback EarlyStopping
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

# Adicionar peso nas classes minoritárias
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

# Treinamento do modelo
history = cnn_model.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size=32, epochs=200, callbacks=[early_stopping], class_weight=class_weights_dict)

# Avaliar o modelo no conjunto de teste
y_pred = (cnn_model.predict(X_test) > 0.5).astype(int)

# Gerar o relatório de classificação
print(classification_report(y_test, y_pred))

# Gerar a matriz de confusão
plot_confusion_matrix(y_test, y_pred)

# Plotar o histórico do treinamento
plot_training_history(history)