# Classificação de Displasia Cortical Focal com Redes Neurais
Este projeto utiliza redes neurais convolucionais siamesas (Siamese Neural Networks) para classificar se um paciente apresenta ou não Displasia Cortical Focal (FCD) a partir de imagens de ressonância magnética do cérebro.

## Objetivo
Desenvolver um sistema automatizado que auxilie especialistas na identificação de displasia cortical focal, uma das causas mais comuns de epilepsia farmacorresistente, utilizando aprendizado profundo aplicado a imagens médicas.

## Estrutura do Projeto
Preprocessamento de Imagens: Leitura de imagens em formato NIfTI (.nii.gz), extração de cortes e preparação de patches.

Geração de Labels: As labels binárias são atribuídas com base em critérios de intensidade e presença de lesões (máscaras).

Tratamento de dados: Suporte a estratégias simples de aumento de dados, como rotação e flips, e suporte para estratégias simples de undersampling, onde são escolhidos um subconjunto randomico e menor de imagens, para balanceamento e robustez.

Arquitetura Siamesa: CNNs gêmeas aplicadas em imagens do lado esquerdo e contralateral, com operações de subtração ou concatenação de embeddings.

Treinamento Supervisionado: Uso de class_weight, EarlyStopping, ReduceLROnPlateau, e ModelCheckpoint para otimização.

Avaliação do Modelo: Métricas como AUC, precisão, revocação, F1-score e matrizes de confusão são utilizadas.

## Tecnologias e Bibliotecas
Python 3.9+

TensorFlow / Keras

NumPy, SciPy, Matplotlib, OpenCV

NiBabel (leitura de arquivos NIfTI)

Scikit-learn (avaliação e divisão de dados)

## Dataset
O projeto trabalha com um conjunto de dados privado contendo imagens de RM ponderadas em T1, associadas a máscaras de lesões (quando presentes). As imagens são divididas em conjuntos de treino, validação e teste por pacientes.

## Como Funciona
Entrada: Dois patches 2D (lado esquerdo e lado contralateral) de uma mesma coordenada.

CNN Base: Aplica convoluções, pooling e flattening para gerar embeddings.

Comparação: As saídas da CNN são subtraídas ou concatenadas.

Decisão: O resultado é passado por camadas densas que classificam o par.

## Licença
Este projeto é de uso acadêmico e está sujeito às normas de uso dos dados médicos. Para uso comercial, entre em contato com os autores.
 
