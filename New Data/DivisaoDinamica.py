import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar a image
# m
image_path = '/home/joh/Documentos/GIT/SandBox/autoDetect/img4.jpeg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Detecção simples da área do cérebro (limiarização para segmentar)
_, brain_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)

# Encontrar contornos do cérebro
contours, _ = cv2.findContours(brain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Selecionar o maior contorno (assumindo que é o cérebro)
largest_contour = max(contours, key=cv2.contourArea)

# Obter bounding box do cérebro
x, y, w, h = cv2.boundingRect(largest_contour)

# Criar uma cópia da imagem original para desenhar
output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Desenhar a bounding box do cérebro
cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Desenhar as divisões dos quadrantes em vermelho
# Linha horizontal no meio
cv2.line(output_image, (x, y + h // 2), (x + w, y + h // 2), (0, 0, 255), 2)
cv2.line(output_image, (x, y + h // 4), (x + w, y + h // 4), (0, 0, 255), 2)
cv2.line(output_image, (x, y + 3*h // 4), (x + w, y + 3*h // 4), (0, 0, 255), 2)

cv2.line(output_image, (x + w // 2, y), (x + w // 2, y + h), (0, 0, 255), 2)
cv2.line(output_image, (x + w // 4, y), (x + w // 4, y + h), (0, 0, 255), 2)
cv2.line(output_image, (x + 3*w // 4, y), (x + 3*w // 4, y + h), (0, 0, 255), 2)

# Exibir a imagem
# plt.figure(figsize=(10, 10))
# plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.title('Bounding Box e Divisão de Quadrantes')
# plt.savefig('bounding_box_and_quadrants.png')

# Salvar a imagem com a bounding box e os quadrantes
cv2.imwrite('output_with_bounding_box_and_quadrants.png', output_image)
print("A imagem foi salva como 'output_with_bounding_box_and_quadrants.png'.")
