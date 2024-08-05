#PROCESSAMENTO E ANÁLISE DE IMAGENS
#PONTIFÍCIA UNIVERSIDADE CATÓLICA DE MINAS GERAIS - COREU - MANHÃ
#CIÊNCIAS DA COMPUTAÇÃO

#GRUPO: Davi Lorenzo B. Braga

#SOBRE A ETAPA 2
#NF = 725137 % 3 = 1 -> Características a serem usadas: Descritores de Haralick
#NC = 725137 % 2 = 1 -> Classificador Raso: XGBoost
#ND = (725137 %4)/2 = 0.5 -> Classificador Profundo: EfficientNet

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

FORMATOS_ACEITOS = (".png",".jpeg",".jpg")
plt.switch_backend('TkAgg')


def ChecaCSV(arquivo):
    try:
        return pd.read_csv(arquivo)
    except TypeError:
        print("Erro ao abrir o arquivo. Favor selecionar outro ou verificar o caminho")
        return "Err"

#Abre a imagem e verifica se o arquivo escolhido é JPG ou PNG
def ChecaImagem(arquivo):
    try:
        imagemEscolhida = Image.open(arquivo)
    except(FileNotFoundError):
        print("Imagem não encontrada:  " + arquivo)
        imagemEscolhida = "Err"
    tipo = os.path.splitext(arquivo)
    if tipo[-1] not in FORMATOS_ACEITOS:
        print("Formato de imagem não suportado. Use JPEG ou PNG.")
        imagemEscolhida = "Err"
    
    return imagemEscolhida

def RecortaImagem(arquivo,id,classe,x,y):
    # Se nao existir um diretório para a classe descrita, irá criar um novo
    if not os.path.exists(classe):
        os.makedirs(classe)

    imagem = Image.open(arquivo)

    # Define a área de recorte (Especificado para 100x100 conforme definido no trabalho)
    left = x - 50
    up = y - 50
    right = x + 50
    down = y + 50

    # Certifica que o recorte está dentro dos limites da imagem
    width, height = imagem.size
    if left < 0:
        left = 0
    if up < 0:
        up = 0
    if right > width:
        right = width
    if down > height:
        down = height

    # Garantir que left < right e up < down após o ajuste
    if left >= right:
        left = max(0, right - 100)
        right = left + 100 if left + 100 <= width else width
    if up >= down:
        up = max(0, down - 100)
        down = up + 100 if up + 100 <= height else height

    cortado = imagem.crop((left, up, right, down))
    destino = classe + "/"+ str(id)+".jpg" 
    cortado.save(destino)

def EscalaDeCinza(imagem):
    result = imagem.convert("L")
    result.point(lambda p: p // 16 * 16) ##reduzindo para 16 tons
    return result

def HistogramaCinza(imagem):
    imagem = EscalaDeCinza(imagem) ##Realiza a conversao para escala de cinza
    histograma, bins = np.histogram(imagem, bins=16, range=(0, 256))
    plt.bar(bins[:-1], histograma, width=16, edgecolor='black')
    plt.title('Histograma 16 Tons de Cinza')
    plt.xlabel('Nível de Cinza')
    plt.ylabel('Frequência')
    plt.xticks(bins)
    plt.show()

def HistogramaHSV(imagem):
    imagem = imagem.convert("HSV") # Convertendo a imagem para o espaço de cores HSV (HUE, SATURATION, VALUE)
    hsvArray = np.array(imagem) # Convertendo a imagem HSV para um array numpy e separando de acordo com os espaços
   
    #Realizando as alterações nos canais definidas manualmente
    h = hsvArray[:, :, 0]
    s = hsvArray[:, :, 1]
    v = hsvArray[:, :, 2]

    # Canal H (Hue)
    plt.subplot(1, 3, 1)
    hist_h, bins_h = np.histogram(h, bins=256, range=(0,256))
    plt.bar(bins_h[:-1], hist_h, width=1, edgecolor='black', color='r')
    plt.title('Histograma Hue (H)')
    plt.xlabel('Hue')
    plt.ylabel('Frequência')
    plt.xticks(np.arange(0, 256, step=32))
    
    # Canal S (Saturação)
    plt.subplot(1, 3, 2)
    hist_s, bins_s = np.histogram(s, bins=256, range=(0, 256))
    plt.bar(bins_s[:-1], hist_s, width=1, edgecolor='black', color='g')
    plt.title('Histograma Saturação (S)')
    plt.xlabel('Saturação')
    plt.ylabel('Frequência')
    plt.xticks(np.arange(0, 256, step=32))
    
    # Canal V (Value)
    plt.subplot(1, 3, 3)
    hist_v, bins_v = np.histogram(v, bins=256, range=(0, 256))
    plt.bar(bins_v[:-1], hist_v, width=1, edgecolor='black', color='b')
    plt.title('Histograma Value (V)')
    plt.xlabel('Value')
    plt.ylabel('Frequência')
    plt.xticks(np.arange(0, 256, step=32))
    
    plt.tight_layout()
    plt.show()

def Quantiza(canal,niveis): #Ajusta o canal com base nos niveis definidos manualmente
    if(niveis == 0): niveis = 1
    resp = np.floor(canal / (256 / niveis)).astype(int)
    return resp

def HistogramaHSV2D(imagem,canalH,canalV):
    imagem = imagem.convert("HSV") # Convertendo a imagem para o espaço de cores HSV (HUE, SATURATION, VALUE)
    hsvArray = np.array(imagem) # Convertendo a imagem HSV para um array numpy e separando de acordo com os espaços

    h = int(canalH)
    v = int(canalV)
    #Realizando as alterações nos canais definidas manualmente
    hQuant = Quantiza(hsvArray[:, :, 0],h)
    vQuant = Quantiza(hsvArray[:, :, 2],v)

    hist_2d = np.histogram2d(hQuant.flatten(), vQuant.flatten(), bins=[h, v], range=[[0, h], [0, v]])

    # Acessando apenas o histograma
    histograma = hist_2d[0]

    # Plota o histograma 2D
    plt.figure(figsize=(8, 6))
    plt.imshow(histograma.T, origin='lower', aspect='auto', extent=[0, h, 0, v], cmap='viridis')
    plt.colorbar(label='Frequência')
    plt.xlabel('Hue (H)')
    plt.ylabel('Value (V)')
    plt.title('Histograma HSV 2D')
    plt.show()

def CoOcorrencia(imagem, distancias):
    imagem = EscalaDeCinza(imagem)  # Converte imagem para escala de cinza 
    arrayImagem = np.asarray(imagem)
    arrayImagem = Quantiza(arrayImagem,16)  # Quantiza para 16 tons de cinza
    linhas, colunas = arrayImagem.shape
    niveis_cinza = 16  # Número de níveis de cinza para a quantização
    matrizes = {}  # Irá armazenar os resultados dos cálculos

    for distancia in distancias:
        result = np.zeros((niveis_cinza, niveis_cinza), dtype=int) 
        for i in range(linhas):
            for j in range(colunas):
                if (i + distancia < linhas) and (j + distancia < colunas):
                    pixelAtual = arrayImagem[i, j]
                    proxPixel = arrayImagem[i + distancia, j + distancia]
                    result[pixelAtual, proxPixel] += 1
        matrizes[distancia] = result
    
    return matrizes

def Haralick(imagem, distancias):   ##Retorna um dicionario com os seguintes descritores: ENTROPIA, HOMOGENEIDADE E CONTRASTE
    matrizesDeCoOcorrencia = CoOcorrencia(imagem, distancias)
    haralick=[]

    for distancia in distancias:
        matriz =  matrizesDeCoOcorrencia[distancia]
        descritor = {
            "Distância": distancia,
            "Entropia": EntropiaHaralick(matriz),
            "Homogeneidade": HomogenHaralick(matriz),
            "Contraste": ContrasteHaralick(matriz),
        }
        haralick.append(descritor)

    return haralick

def EntropiaHaralick(matriz):
    if(np.sum(matriz) == 0): return 0
    result = matriz/np.sum(matriz) ##Normal
    log = np.log(result + 1e-10) ## O 1e-10 evita o log de zero
    result = np.sum(result * log) * -1
    return round(result, 2)

def HomogenHaralick(matriz):
    if(np.sum(matriz) == 0): return 0
    result = matriz/np.sum(matriz) ##Normal
    result = np.sum(result / (1 + np.abs(np.indices(matriz.shape)[0] - np.indices(matriz.shape)[1])))
    return round(result, 2)

def ContrasteHaralick(matriz):
    if(np.sum(matriz) == 0): return 0
    result = matriz/np.sum(matriz) ##Normal
    linhas, colunas = np.indices(matriz.shape)
    result = np.sum((linhas-colunas)**2 * result)
    return round(result, 2)

def Hu(imagem):
    imagem = np.array(imagem)  # Transforma em array

    # Calculo dos momentos da imagem
    altura, largura = imagem.shape
    momentos = np.zeros((4, 4))  # Aumenta a matriz para 4x4 para acomodar todos os momentos necessários
    for i in range(altura):
        for j in range(largura):
            momentos[0, 0] += imagem[i, j]
            momentos[0, 1] += j * imagem[i, j]
            momentos[0, 2] += j**2 * imagem[i, j]
            momentos[1, 0] += i * imagem[i, j]
            momentos[1, 1] += i * j * imagem[i, j]
            momentos[1, 2] += i * j**2 * imagem[i, j]
            momentos[2, 0] += i**2 * imagem[i, j]
            momentos[2, 1] += i**2 * j * imagem[i, j]
            momentos[2, 2] += i**2 * j**2 * imagem[i, j]
            momentos[3, 0] += i**3 * imagem[i, j]
            momentos[0, 3] += j**3 * imagem[i, j]
            momentos[1, 3] += i * j**2 * imagem[i, j]
            momentos[2, 3] += j * i**2 * imagem[i, j]

    # Coordenadas do centroide
    centroideX = momentos[1, 0] / momentos[0, 0]
    centroideY = momentos[0, 1] / momentos[0, 0]

    # Calculo dos momentos centrais
    mu = np.zeros((4, 4))
    for i in range(altura):
        for j in range(largura):
            x = i - centroideX
            y = j - centroideY
            for p in range(4):
                for q in range(4):
                    if p + q <= 3:
                        mu[p, q] += (x**p) * (y**q) * imagem[i, j]

    # Calculo dos momentos normalizados
    nu = np.zeros((4, 4))
    for p in range(4):
        for q in range(4):
            if p + q >= 2:
                nu[p, q] = mu[p, q] / (momentos[0, 0]**(1 + (p + q) / 2))

    # Calculo dos momentos invariantes de Hu
    hu = np.zeros(7)
    hu[0] = nu[2, 0] + nu[0, 2]
    hu[1] = (nu[2, 0] - nu[0, 2])**2 + 4 * nu[1, 1]**2
    hu[2] = (nu[3, 0] - 3 * nu[1, 2])**2 + (3 * nu[2, 1] - nu[0, 3])**2
    hu[3] = (nu[3, 0] + nu[1, 2])**2 + (nu[2, 1] + nu[0, 3])**2
    hu[4] = (nu[3, 0] - 3 * nu[1, 2]) * (nu[3, 0] + nu[1, 2]) * ((nu[3, 0] + nu[1, 2])**2 - 3 * (nu[2, 1] + nu[0, 3])**2) + (3 * nu[2, 1] - nu[0, 3]) * (nu[2, 1] + nu[0, 3]) * (3 * (nu[3, 0] + nu[1, 2])**2 - (nu[2, 1] + nu[0, 3])**2)
    hu[5] = (nu[2, 0] - nu[0, 2]) * ((nu[3, 0] + nu[1, 2])**2 - (nu[2, 1] + nu[0, 3])**2) + 4 * nu[1, 1] * (nu[3, 0] + nu[1, 2]) * (nu[2, 1] + nu[0, 3])
    hu[6] = (3 * nu[2, 1] - nu[0, 3]) * (nu[3, 0] + nu[1, 2]) * ((nu[3, 0] + nu[1, 2])**2 - 3 * (nu[2, 1] + nu[0, 3])**2) - (nu[3, 0] - 3 * nu[1, 2]) * (nu[2, 1] + nu[0, 3]) * (3 * (nu[3, 0] + nu[1, 2])**2 - (nu[2, 1] + nu[0, 3])**2)

    return hu