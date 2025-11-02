# Classificação de Displasia Cortical Focal com Redes Neurais
Este projeto foca no desenvolvimento e comparação de arquiteturas de aprendizado profundo para a classificação de Displasia Cortical Focal (FCD) a partir de imagens de ressonância magnética (MRI) do cérebro. O objetivo é criar um sistema automatizado que auxilie especialistas na identificação desta condição, que é uma das causas mais comuns de epilepsia refratária (farmacorresistente).

## Objetivo
Desenvolver um sistema automatizado que auxilie especialistas na identificação de displasia cortical focal, utilizando aprendizado profundo aplicado a imagens médicas (MRI).

## Estrutura do Projeto e Metodologias
O repositório está organizado em duas abordagens principais, refletindo a evolução da pesquisa:

### 1. `New_Methods` (Abordagem com Transformers)
Esta pasta contém as implementações mais recentes, que exploram o uso de arquiteturas baseadas em Transformers para processamento de imagens médicas.
* `pre_processing/`: Scripts e notebooks dedicados ao pré-processamento de dados específicos para os modelos Transformers.
* `Transformers2D.ipynb`: Notebook para o desenvolvimento e treinamento de modelos baseados em Vision Transformers (ViT) aplicados a cortes 2D das imagens de MRI.
* `Transformers3D.ipynb`: Notebook que explora o uso de Transformers para dados volumétricos (3D), processando múltiplos cortes ou o volume inteiro.
* `pre_process.py`: Script de pré-processamento principal para esta abordagem.

### 2. `Old_Methods` (Abordagem Siamesa e Contrastiva)
Esta pasta contém as arquiteturas "clássicas" que serviram de base para o projeto, focadas em Redes Neurais Siamesas (SNN) e Aprendizado Contrastivo. A lógica central aqui é comparar patches da lesão com seu correspondente contralateral (do outro lado do cérebro).
* `SNN.ipynb` / `SNN_Manual.ipynb`: Notebooks com a implementação da Rede Neural Siamesa. A CNN base extrai *embeddings* (características) de ambos os patches (lesão e contralateral), que são então subtraídos ou concatenados para uma classification final.
* `Contrastive_SNN.ipynb` / `Contrastive_SSCL.ipynb`: Implementações que utilizam *loss* (função de perda) contrastiva. O objetivo é "ensinar" o modelo a aproximar os *embeddings* de pares da mesma classe (ex: dois patches saudáveis) e afastar os de classes diferentes (ex: um patch saudável e um com lesão).
* `GridCreation.ipynb`, `PlotPairs.ipynb`, `SaveAllSlices.py`: Scripts utilitários para geração de dados, visualização de pares de imagens e salvamento de cortes para análise.

## Tecnologias e Bibliotecas
* **Core:** Python 3.9+
* **Deep Learning:** TensorFlow / Keras
* **Processamento de Imagens Médicas:** NiBabel (para leitura de arquivos NIfTI .nii.gz)
* **Computação Científica:** NumPy, SciPy
* **Manipulação de Imagens:** OpenCV
* **Avaliação e Utilitários:** Scikit-learn, Matplotlib

## Dataset
O projeto utiliza um conjunto de dados privado contendo imagens de RM ponderadas em T1 (T1-weighted) e suas respectivas máscaras de lesão (quando presentes). Para garantir a imparcialidade, os dados são divididos em conjuntos de treino, validação e teste por paciente, evitando que dados do mesmo paciente estejam em conjuntos diferentes.

## Pipeline de Trabalho
1.  **Pré-processamento:** Leitura dos arquivos NIfTI, extração de cortes (slices) e criação de *patches* (pequenos recortes) das regiões de interesse.
2.  **Geração de Labels:** Atribuição de rótulos binários (lesão/não-lesão) com base nas máscaras e critérios de intensidade.
3.  **Balanceamento de Dados:** Aplicação de técnicas de *data augmentation* (rotação, flips) e estratégias de *undersampling* para lidar com o desbalanceamento entre classes.
4.  **Treinamento:** Otimização do modelo com técnicas como `class_weight` (pesos de classe), `EarlyStopping` (parada antecipada), `ReduceLROnPlateau` (redução da taxa de aprendizado) e `ModelCheckpoint` (salvamento do melhor modelo).
5.  **Avaliação:** Análise de performance com métricas como Curva ROC/AUC, Acurácia, Precisão, Revocação, F1-Score e Matriz de Confusão.

## 📄 Licença
Este projeto é de uso acadêmico e está sujeito às normas de uso dos dados médicos. Para uso comercial, entre em contato com os autores.
