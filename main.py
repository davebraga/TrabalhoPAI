#PROCESSAMENTO E ANÁLISE DE IMAGENS
#PONTIFÍCIA UNIVERSIDADE CATÓLICA DE MINAS GERAIS - COREU - MANHÃ
#CIÊNCIAS DA COMPUTAÇÃO

#GRUPO: Davi Lorenzo B. Braga

#SOBRE A ETAPA 2
#NF = 725137 % 3 = 1 -> Características a serem usadas: Descritores de Haralick
#NC = 725137 % 2 = 1 -> Classificador Raso: XGBoost
#ND = (725137 %4)/2 = 0.5 -> Classificador Profundo: EfficientNet


import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
import parte1, parte2
import threading
from tkinter import filedialog, messagebox, simpledialog

def selecionarArquivo():
    filename = filedialog.askopenfilename(title="Selecione um arquivo de imagem")
    if filename:
        filename = parte1.ChecaImagem(filename)
        manipularImagem(filename)

def manipularImagem(filename):
    # Limpar a janela principal para adicionar novos widgets
    for widget in root.winfo_children():
        widget.destroy()

    # Botões para as opções de manipulação da imagem
    btn_tons_cinza = tk.Button(root, text="Converter para tons de cinza", command=lambda: opcaoImagem(1,filename))
    btn_tons_cinza.pack(pady=5)

    btn_histograma = tk.Button(root, text="Gerar histograma", command=lambda: histogramas(filename))
    btn_histograma.pack(pady=5)

    btn_haralick = tk.Button(root, text="Caracterizar com Haralick", command=lambda: opcaoImagem(3,filename))
    btn_haralick.pack(pady=5)

    btn_momentos_hu = tk.Button(root, text="Caracterizar com Momentos de Hu", command=lambda: opcaoImagem(4,filename))
    btn_momentos_hu.pack(pady=5)

    btn_outra_imagem = tk.Button(root, text="Escolher outra imagem", command=selecionarArquivo)
    btn_outra_imagem.pack(pady=5)

    btn_voltar = tk.Button(root, text="Voltar", command=inicializar)
    btn_voltar.pack(pady=5)


def opcaoImagem(opcao,imagemEscolhida):
    if opcao == 1:
        grayscale = parte1.EscalaDeCinza(imagemEscolhida)
        plt.imshow(grayscale, cmap='gray')
        plt.axis('off')  # Oculta os eixos
        plt.show()
    elif opcao == 3:
        haralick(imagemEscolhida)
    elif opcao == 4:
        Hu(imagemEscolhida)

def histogramas(imagemEscolhida):
    # Limpar a janela principal para adicionar novos widgets
    for widget in root.winfo_children():
        widget.destroy()

    btn_tons_cinza = tk.Button(root, text="Histograma em tons de cinza", command=lambda: opcaoHistograma(1,imagemEscolhida))
    btn_tons_cinza.pack(pady=5)

    btn_histograma = tk.Button(root, text="Histograma HSV", command=lambda: opcaoHistograma(2,imagemEscolhida))
    btn_histograma.pack(pady=5)

    btn_haralick = tk.Button(root, text="Histograma HSV 2D", command=lambda: opcaoHistograma(3,imagemEscolhida))
    btn_haralick.pack(pady=5)

    btn_voltar = tk.Button(root, text="Voltar", command=lambda: manipularImagem(imagemEscolhida))
    btn_voltar.pack(pady=5)

def opcaoHistograma(opcao,imagemEscolhida):
    if(opcao==1):
        parte1.HistogramaCinza(imagemEscolhida)
    elif(opcao==2):
        parte1.HistogramaHSV(imagemEscolhida)
    elif(opcao==3):
        canalH = simpledialog.askinteger("Hue", "Digite o numero de canais para HUE (Digite '256' caso queira os valores padrão)\n")
        canalV = simpledialog.askinteger("Value", "Digite o numero de canais para VALUE (Digite '256' caso queira os valores padrão)\n")
        if(canalH==0 or canalV==0):
            messagebox.showinfo("Erro", "Valor '0' não permitido!") 
        else: 
            parte1.HistogramaHSV2D(imagemEscolhida,canalH,canalV)

def haralick(imagemEscolhida):
    # Limpar a janela principal para adicionar novos widgets
    for widget in root.winfo_children():
        widget.destroy()

    btn_tons_cinza = tk.Button(root, text="Visualizar as matrizes de Co-ocorrência [1,2,4,8,16 e 32]", command=lambda: opcaoHaralick(1,imagemEscolhida))
    btn_tons_cinza.pack(pady=5)

    btn_histograma = tk.Button(root, text="Cálculo dos Descritores de Haralick (Entropia, Homogeneidade e Contraste)", command=lambda: opcaoHaralick(2,imagemEscolhida))
    btn_histograma.pack(pady=5)

    btn_voltar = tk.Button(root, text="Voltar", command=lambda: manipularImagem(imagemEscolhida))
    btn_voltar.pack(pady=5)

def opcaoHaralick(opcaoHaralick,imagemEscolhida):
    distancias = [1,2,4,8,16,32] ## Será usado por ambas opções
    if(opcaoHaralick == 1):
        matrizesCoOcorr= parte1.CoOcorrencia(imagemEscolhida,distancias)
        plt.figure(figsize=(12, 8))
        pos=1
        for distancia in distancias: ## Vai gerar o grafico de cada um dos resultados encontrados
            matriz = matrizesCoOcorr[distancia]
            plt.subplot(3, 2, pos) # Tres colunas e duas linhas
            plt.imshow(matriz, cmap="gray", interpolation="nearest")
            plt.title(f"Distância {distancia} ")
            plt.xlabel("Tons de Cinza")
            plt.ylabel("Tons de Cinza")  
            pos+=1  
        plt.tight_layout()
        plt.show()
    elif(opcaoHaralick == 2):
        descritores = parte1.Haralick(imagemEscolhida,distancias)
        message=""
        for descritor in descritores:
            message+="Distância "+ str(descritor['Distância']) +"\n"
            message+="Entropia: \n"+ str(descritor['Entropia']) +"\n"
            message+= "Homogeneidade: \n" + str(descritor['Homogeneidade']) +"\n"
            message+= "Contraste: \n"+ str(descritor['Contraste'])+ "\n"
            message+="================= \n"
        threading.Thread(target=lambda: messagebox.showinfo("Resultados",message)).start()

def Hu(imagemEscolhida):
    # Limpar a janela principal para adicionar novos widgets
    for widget in root.winfo_children():
        widget.destroy()

    btn_tons_cinza = tk.Button(root, text="Calcular para tons de cinza", command=lambda: opcaoHu(1,imagemEscolhida))
    btn_tons_cinza.pack(pady=5)

    btn_histograma = tk.Button(root, text="Calcular baseado no histograma HSV", command=lambda: opcaoHu(2,imagemEscolhida))
    btn_histograma.pack(pady=5)

    btn_voltar = tk.Button(root, text="Voltar", command=lambda: manipularImagem(imagemEscolhida))
    btn_voltar.pack(pady=5)

def opcaoHu(opcao,imagemEscolhida):
    message=""
    if(opcao==1):
        imagem = imagemEscolhida.convert("L") #Converte para tons de cinza com 256 tons
        hu = parte1.Hu(imagem)
        for i in range(len(hu)):
            message+= "Hu "+"["+ str(i+1) + "] = " + str(round(hu[i],3))+ "\n"
        print("================================================= \n")
        threading.Thread(target=lambda: messagebox.showinfo("Resultados",message)).start()

    elif(opcao==2):
        imagemHSV = imagemEscolhida.convert('HSV')
        h, s, v = imagemHSV.split()
        resultH = hu = parte1.Hu(h)
        resultS = hu = parte1.Hu(s)
        resultV = hu = parte1.Hu(v)
        
        message+=("HUE\n")
        for i in range(len(resultH)):
            message+= "Hu "+"["+ str(i+1) + "] = " + str(round(resultH[i],3))+ "\n"
        
        message+=("SATURAÇÃO\n")
        for i in range(len(resultS)):
            message+= "Hu "+"["+ str(i+1) + "] = " + str(round(resultS[i],3))+ "\n"
        
        message+=("VALOR\n")
        for i in range(len(resultV)):
            message+= "Hu "+"["+ str(i+1) + "] = " + str(round(resultV[i],3))+ "\n"

        threading.Thread(target=lambda: messagebox.showinfo("Resultados",message)).start()

def treinarReconhecimento():
    conjTreino, conjTeste = parte2.SorteiaConjuntos()

    # Limpar a janela principal para adicionar novos widgets
    for widget in root.winfo_children():
        widget.destroy()
    
    btn_tons_cinza = tk.Button(root, text="Classificadores Rasos", command=lambda: opcaoReconhecimento(1,conjTreino, conjTeste))
    btn_tons_cinza.pack(pady=5)

    btn_histograma = tk.Button(root, text="Classificadores Profundos", command=lambda: opcaoReconhecimento(2,conjTreino, conjTeste))
    btn_histograma.pack(pady=5)

    btn_voltar = tk.Button(root, text="Voltar", command=inicializar)
    btn_voltar.pack(pady=5)

def opcaoReconhecimento(opcao,conjTreino, conjTeste):
    if(opcao==1):

        # Extração de características
        featureTreino = parte2.ExtraiFeatures(conjTreino)
        featureTeste = parte2.ExtraiFeatures(conjTeste)

        # Separação das labels
        labelTreino = [classe for classe, _ in conjTreino]
        labelTeste = [classe for classe, _ in conjTeste]

        # Convertendo as listas de características em arrays numpy para o classificador
        featureTreino = np.array(featureTreino)
        featureTeste = np.array(featureTeste)

        # Classificação e avaliação
        resultados = parte2.ClassificadoresRasos(featureTreino, labelTreino, featureTeste, labelTeste)
        mensagem = ""
        i=1
        for resultado in resultados:
            if(i==1): mensagem+="CLASSIFICACAO BINARIA\n"
            else:  mensagem+="CLASSIFICACAO MULTICLASSE\n"
            acuracia, matrizConfusao = resultado
            mensagem+=(f"Taxa de Acurácia:\n") 
            mensagem+= str(resultado["Acuracia"])+"\n"
            mensagem+=(f"Matriz de Confusão:\n")
            mensagem+= str(resultado["MatrizConfusao"])+"\n \n"
            i+=1

        threading.Thread(target=lambda: messagebox.showinfo("Resultados",mensagem)).start()

        
        

    elif(opcao==2):

        # Realizar o treinamento dos classificadores binário e multiclasse
        dadosTreino, dadosValidacao = parte2.GerarDatasets(conjTreino, conjTeste)
        resultados = parte2.ClassificadorProfundo(dadosTreino, dadosValidacao, 10)
        plt.figure(figsize=(14, 7))
        i=1
        mensagem = ""
        # Exibir os resultados
        for resultado in resultados:
            if(i==1): mensagem+="CLASSIFICACAO BINARIA\n"
            else:  mensagem+="CLASSIFICACAO MULTICLASSE\n"
            acuracia, matrizConfusao = resultado["Resultado"]
            mensagem+=(f"Taxa de Acurácia:\n") 
            mensagem+= str(round(acuracia,3))+"\n"
            mensagem+=(f"Matriz de Confusão:\n")
            mensagem+= str(matrizConfusao)+"\n \n"

            historico = resultado["Historico"]
            epocas = range(1, len(historico['accuracy']) + 1)

            plt.subplot(1, 2, i)
            plt.plot(epocas, historico['accuracy'], 'ro-', label='Acurácia de Treino')
            plt.plot(epocas, historico['val_accuracy'], 'b^-', label='Acurácia de Validação')
            
            if i == 1:
                plt.title('Acurácia de Treino e Validação - Classificação Binária')
            else:
                plt.title('Acurácia de Treino e Validação - Classificação Multiclasse')
            
            plt.xlabel('Épocas')
            plt.ylabel('Acurácia')
            plt.legend()
            
            i += 1

        threading.Thread(target=lambda: messagebox.showinfo("Resultados",mensagem)).start()
        plt.tight_layout()
        plt.show()
        

def selecionarTabela():
    filename = filedialog.askopenfilename(title="Selecione arquivo .csv")
    if filename:
        recortarImagens(filename)

def recortarImagens(arquivo):
    tabela = parte1.ChecaCSV(arquivo)
    
    #Selecionando apenas os itens necessários para recorte na tabela
    array = tabela[["image_filename","cell_id", "bethesda_system","nucleus_x", "nucleus_y"]]
    # Transformando para um array numPy
    array = array.values

    for linha in array:
            parte1.RecortaImagem("dataset/"+linha[0],linha[1],linha[2],linha[3],linha[4])

    messagebox.showinfo("Sucesso","Recorte das imagens feito com sucesso!")

def inicializar():
    # Limpar a janela principal para adicionar novos widgets
    for widget in root.winfo_children():
        widget.destroy()

    # Selecionar arquivo
    btn_selecionar_arquivo = tk.Button(root, text="Manipular uma imagem", command=selecionarArquivo)
    btn_selecionar_arquivo.pack(pady=5)

    #Treinar Reconhecimento
    btn_treinar = tk.Button(root, text="Treinar e processar reconhecimento de células", command=treinarReconhecimento)
    btn_treinar.pack(pady=5)

    #Recortar imagens
    btn_recortar = tk.Button(root, text="Realizar recorte das imagens com base no arquivo .csv", command=selecionarTabela)
    btn_recortar.pack(pady=5)

    # Sair do programa
    btn_sair = tk.Button(root, text="Sair do programa", command=root.destroy)
    btn_sair.pack(pady=5)

# Criando a janela principal
root = tk.Tk()
root.title("Trabalho P.A.I.")

# Inicializar a janela com os botões principais
inicializar()

root.mainloop()
