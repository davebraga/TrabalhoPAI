#PROCESSAMENTO E ANÁLISE DE IMAGENS
#PONTIFÍCIA UNIVERSIDADE CATÓLICA DE MINAS GERAIS - COREU - MANHÃ
#CIÊNCIAS DA COMPUTAÇÃO

#GRUPO: Davi Lorenzo B. Braga

#SOBRE A ETAPA 2
#NF = 725137 % 3 = 1 -> Características a serem usadas: Descritores de Haralick
#NC = 725137 % 2 = 1 -> Classificador Raso: XGBoost
#ND = (725137 %4)/2 = 0.5 -> Classificador Profundo: EfficientNet

import os,random,cv2
import numpy as np
from parte1 import Haralick ,ChecaImagem
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def ClassesUsadas():
    return ["ASC-H","ASC-US", "HSIL", "LSIL", "Negative for intraepithelial lesion","SCC"]

def SorteiaConjuntos():
    Classes = ClassesUsadas()
    conjTreino = []
    conjTeste= []

    #Obtem o diretório no qual o codigo se encontra
    diretorioAtual = os.getcwd()

    for classe in Classes:
        diretorio = os.path.join(diretorioAtual,classe)

        # Lista todas as imagens na pasta da classe
        imagens = [f for f in os.listdir(diretorio) if os.path.isfile(os.path.join(diretorio, f))]

        random.shuffle(imagens)
        
        # Divide as imagens em treino e teste na proporção 4:1
        split_index = int(len(imagens) * 0.8)
        conjTreino.extend([(classe, os.path.join(diretorio, img)) for img in imagens[:split_index]])
        conjTeste.extend([(classe, os.path.join(diretorio, img)) for img in imagens[split_index:]])

    return [conjTreino,conjTeste]

## Ao realizar dataset, é preciso que todas as imagens tenham o mesmo tamanho, para isso, essa imagem irá redefinir todas elas para um tamanho default

def PadronizaImagem(caminho, dimensao):
    largura, altura = dimensao
    try:
        img = cv2.imread(caminho)
        if img is None:
            print(f"Erro: Não foi possível carregar a imagem em {caminho}. Pulando para a próxima imagem.")
            return None
        img_redimensionada = cv2.resize(img, (largura, altura), interpolation=cv2.INTER_AREA)
        return img_redimensionada
    except Exception as e:
        print(f"Erro ao processar a imagem em {caminho}: {str(e)}")
        return None

def RemoverImagensComProblemas(conjunto):
    caminhos_sem_problemas = []
    imagens_removidas = 0  # Inicializa o contador de imagens removidas
    for classe, caminho in conjunto:
        img = PadronizaImagem(caminho, (224, 224))
        if img is not None:
            caminhos_sem_problemas.append((classe, caminho))
        else:
            imagens_removidas += 1  # Incrementa o contador de imagens removidas
    print(f"Total de {imagens_removidas} imagens removidas.")
    return caminhos_sem_problemas

## =====================================================================================================================================================
## ================================================================CLASSIFICADORES RASOS================================================================

#Realizando as features com base nos Descritores de Haralick
def ExtraiFeatures(conjunto):
    features=[]
    distancias = [1,2,4,8,16,32]
    for item in conjunto:
        classe, caminho = item #Leve gambiarra para obter o caminho da imagem
        imagem = ChecaImagem(caminho)
        if(caminho!= "Err"):
            haralickImagem = Haralick(imagem,distancias)
            # Transforma a saída de Haralick em um vetor plano de características
            vetor = []
            for d in haralickImagem:
                vetor.extend([d["Entropia"], d["Homogeneidade"], d["Contraste"]])

            features.append(vetor)

    return features

def ClassificadoresRasos(featureTreino, labelTreino, featureTeste,labelTeste): # Utilizando XGBoost
    classifBinario = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    classifMulti = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

    #Transformado em um array numpy
    le = LabelEncoder()
    labelTreino_encoded = le.fit_transform(labelTreino)
    labelTeste_encoded = le.transform(labelTeste)

    resultados = []

    # Classificação binária
    treinoBinario = (labelTreino_encoded == le.transform(["Negative for intraepithelial lesion"])[0]).astype(int)
    testeBinario = (labelTeste_encoded == le.transform(["Negative for intraepithelial lesion"])[0]).astype(int)
    
    classifBinario.fit(np.array(featureTreino), treinoBinario)
    predicaoBinario = classifBinario.predict(np.array(featureTeste))
    acuraciaBinario = accuracy_score(testeBinario, predicaoBinario)
    matrizConfusaoBinario = confusion_matrix(testeBinario, predicaoBinario)
    resultado={
        "Acuracia":  round(acuraciaBinario, 3),
        "MatrizConfusao": matrizConfusaoBinario 
    }
    resultados.append(resultado)
    #print("=============================CLASSIFICACAO BINARIA=============================\n")
    #print(f"Taxa de Acurácia: {round(acuraciaBinario, 3)}\n")
    #print(f"Matriz de Confusão:\n{matrizConfusaoBinario}\n")

    # Classificação multiclasse
    classifMulti.fit(np.array(featureTreino), labelTreino_encoded)
    predicaoMulti = classifMulti.predict(np.array(featureTeste))
    acuraciaMulti = accuracy_score(labelTeste_encoded, predicaoMulti)
    matrizConfusaoMulti = confusion_matrix(labelTeste_encoded, predicaoMulti)
    resultado={
        "Acuracia":  round(acuraciaMulti, 3),
        "MatrizConfusao": matrizConfusaoMulti 
    }
    resultados.append(resultado)
    #print("=============================CLASSIFICACAO MULTICLASSE=============================\n")
    #print(f"Taxa de Acurácia: {round(acuraciaMulti, 3)}\n")
    #print(f"Matriz de Confusão:\n{matrizConfusaoMulti}\n")

    return resultados

## =========================================================================================================================================================
## ================================================================CLASSIFICADORES PROFUNDOS================================================================

def GerarDatasets(conjTreino, conjTeste):
    conjTreinoSemProblemas = RemoverImagensComProblemas(conjTreino)
    conjTesteSemProblemas = RemoverImagensComProblemas(conjTeste)

    datagenTreino = ImageDataGenerator(rescale=1.0/255.0,
                                       rotation_range=20,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')

    datagenTeste = ImageDataGenerator(rescale=1.0/255.0)

    le = LabelEncoder()
    ytreino = le.fit_transform([classe for classe, _ in conjTreinoSemProblemas])
    yteste = le.transform([classe for classe, _ in conjTesteSemProblemas])

    Xtreino = np.array([PadronizaImagem(caminho, (224, 224)) for _, caminho in conjTreinoSemProblemas])
    Xteste = np.array([PadronizaImagem(caminho, (224, 224)) for _, caminho in conjTesteSemProblemas])

    treino_generator = datagenTreino.flow(Xtreino, ytreino, batch_size=32)
    teste_generator = datagenTeste.flow(Xteste, yteste, batch_size=32)

    return treino_generator, teste_generator

def ClassificadorProfundo(dadosTreino, dadosValidacao, epocas):
    result = []
    modeloBin = GeraClassificador(len(ClassesUsadas()), True)
    historico_binario = TreinaClassificador(modeloBin, dadosTreino, dadosValidacao, epocas)
    resultado_binario = AvaliaClassificador(modeloBin, dadosValidacao, True)
    result.append({"Historico": historico_binario, "Resultado": resultado_binario})

    modeloMulti = GeraClassificador(len(ClassesUsadas()), False)
    historico_multiclasse = TreinaClassificador(modeloMulti, dadosTreino, dadosValidacao, epocas)
    resultado_multiclasse = AvaliaClassificador(modeloMulti, dadosValidacao, False)
    result.append({"Historico": historico_multiclasse, "Resultado": resultado_multiclasse})

    return result

def GeraClassificador(numClasses, binario):
    base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)

    if binario:
        predictions = Dense(1, activation="sigmoid")(x)
        loss = "binary_crossentropy"
        metrics = ["accuracy"]
    else:
        predictions = Dense(numClasses, activation="softmax")(x)
        loss = "sparse_categorical_crossentropy"
        metrics = ["accuracy"]

    model = Model(inputs=base.input, outputs=predictions)

    for layer in base.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=0.001), loss=loss, metrics=metrics)

    return model

def TreinaClassificador(modelo, geradorTreino, geradorValidacao, epocas):
    passosPorEpoca = geradorTreino.n // geradorTreino.batch_size
    passosValidacao = geradorValidacao.n // geradorValidacao.batch_size

    if geradorTreino is not None and geradorValidacao is not None:
        historico = modelo.fit(
            geradorTreino,
            epochs=epocas,
            validation_data=geradorValidacao,
            steps_per_epoch=passosPorEpoca,
            validation_steps=passosValidacao
        )

        return historico.history

    return None

def AvaliaClassificador(model, dadosValidacao, binario):
    dadosValidacao.reset()
    loss, acuracia = model.evaluate(dadosValidacao)
    print(f"Acurácia: {acuracia * 100:.2f}%")

    predicoes = model.predict(dadosValidacao)
    if binario:
        classePredicoes = (predicoes > 0.5).astype(int).flatten()
    else:
        classePredicoes = np.argmax(predicoes, axis=1)

    classeValidacao = []
    for i in range(len(dadosValidacao)):
        classeValidacao.extend(dadosValidacao[i][1])

    classeValidacao = np.array(classeValidacao)

    matrizConfusao = confusion_matrix(classeValidacao, classePredicoes)
    return [acuracia, matrizConfusao]
