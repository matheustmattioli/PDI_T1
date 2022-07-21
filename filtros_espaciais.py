# Código para implementação do primeiro trabalho prático da disciplina
# Processamento Digital de Imagens 2022/1 oferta pelo DC-UFSCar
# Feito pelos alunos:
#   - Lucas Machado Cid          - RA: 769841
#   - Matheus Teixeira Mattioli  - RA: 769783
from unicodedata import name
import numpy as np 
import matplotlib.pyplot as plt

# Função para realizar filtragem dos pixels de uma imagem
# através da média geometrica dos pixels de uma vizinhança.
# O tamanho dessa vizinhança é definido por w
def geometricMean(img, w):
    # Dimensões de linhas e colunas da imagem e 
    # da vizinhança do w passado.
    num_rows, num_cols = img.shape
    num_rows_f, num_cols_f = w.shape

    # Pegamos metade das dimensões de w
    # para fazer o padding
    half_num_rows_f = num_rows_f//2       # O operador // retorna a parte inteira da divisão
    half_num_cols_f = num_cols_f//2

    # Cria imagem com uns ao redor da borda
    # Escolhemos um ao invés de zero, pois ele é o elemento neutro da multiplicação
    # se fosse zeros a imagem ficaria toda preta.
    img_padded = np.ones((num_rows+2*half_num_rows_f, num_cols+2*half_num_cols_f), dtype=img.dtype)
    for row in range(num_rows):
        for col in range(num_cols):   
            img_padded[row+half_num_rows_f, col+half_num_cols_f] = img[row, col]
    
    # Aplicação do filtro de média geométrica nos pixels da imagem
    img_filtered = np.zeros((num_rows, num_cols))
    for row in range(num_rows):
        for col in range(num_cols):
            prod_region = 1
            for s in range(num_rows_f):
                for t in range(num_cols_f):
                    prod_region *= (img_padded[row+s, col+t] * w[s, t])**(1/(num_rows_f*num_cols_f))
            img_filtered[row, col] = int(prod_region)
            
    return img_filtered

# Função para realizar filtragem dos pixels de uma imagem
# através da mediana dos pixels de uma vizinhança.
# O tamanho dessa vizinhança é definido por w
def median(img, w):
    # Dimensões de linhas e colunas da imagem e 
    # da vizinhança do w passado.
    num_rows, num_cols = img.shape
    num_rows_f, num_cols_f = w.shape  

    # Pegamos metade das dimensões de w
    # para fazer o padding
    half_num_rows_f = num_rows_f//2       # O operador // retorna a parte inteira da divisão
    half_num_cols_f = num_cols_f//2

    # Cria imagem com zeros ao redor da borda
    img_padded = np.ones((num_rows+2*half_num_rows_f, num_cols+2*half_num_cols_f), dtype=img.dtype)
    for row in range(num_rows):
        for col in range(num_cols):   
            img_padded[row+half_num_rows_f, col+half_num_cols_f] = img[row, col]
 
    # Aplicação do filtro de mediana nos pixels da imagem
    img_filtered = np.zeros((num_rows, num_cols))
    for row in range(num_rows):
        for col in range(num_cols):
            region = []     # Array que armazena o valor dos pixels da vizinhaça
            for s in range(num_rows_f):
                for t in range(num_cols_f):
                    region.append(img_padded[row+s, col+t])
            region.sort() # Ordenação em ordem crescente
            # Verificamos se o tamanho do filtro é par ou ímpar,
            # se par o meio é calculado através da média entre os valores centrais,
            # se ímpar escolhemos o valor central
            if num_cols_f*num_rows_f % 2 == 0: 
                a = int(region[num_rows_f*num_cols_f//2 - 1])
                b = int(region[num_rows_f*num_cols_f//2])
                
                img_filtered[row, col] = (a + b)//2
            else:
                img_filtered[row, col] = region[num_rows_f*num_cols_f//2]

            
    return img_filtered
    
# Função passada pelo Professor para retirar os canais de cores de uma imagem colorida,
# transformando em escala de cinza.
def rgb2gray(img):
    img_gray = 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]
    return img_gray

# Função para criar ruídos distintos em imagens
# os ruídos utilizados são o Gaussiano e o Salt and Peper (Impulsivo)
def noisy(noise_typ,image):
    if noise_typ == "gauss":
        row, col = image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, size=(row,col))
        gauss = gauss.reshape(row, col)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)

        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[coords] = 0

        return out


s = 3
w = np.ones([s, s])

imagesStrings = [
    {"name": "cameraman.tiff", "hasColor": False},
    {"name": "coruja.jpg", "hasColor": True},
    {"name": "paisagem.jpg", "hasColor": True}
]

noises = [
    "gauss",
    "s&p"
]

filters = [
    geometricMean,
    median
]

for image in imagesStrings:
    plt.figure(figsize=[15,15])
    img = plt.imread(image["name"])
    file_img_name = ".\\filtered_images\\" + "filter_" + image["name"].split(".")[0] + ".png"

    if(image["hasColor"]):
        img = rgb2gray(img)
    
    for i in range(0, len(noises)):
        rows = len(noises)
        cols = len(filters) + 1
        img_noised = noisy(noises[i], img)
        plt.subplot(rows, cols, (i * cols) + 1)
        plt.imshow(img_noised, cmap='gray')
        for j in range(0, len(filters)):
            col = j+2
            plt.subplot(rows, cols, (i * cols) + 2 + j)
            img_smoothed = filters[j](img_noised, w)
            plt.imshow(img_smoothed, cmap='gray')
    plt.savefig(file_img_name)

