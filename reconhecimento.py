from pprint import pprint

import cv2
import numpy as np
import zipfile
import pandas
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# print(tf.__version__)


path = 'imagem/mae.jpg'



img = cv2.imread(path)
# cv2.imshow('Pajezera',img)
# k = cv2.waitKey(0)

faces_cascade = 'Material/haarcascade_frontalface_default.xml'
path_treinada = 'Material/modelo_01_expressoes.h5'

# faz a deteccao da face
detection = cv2.CascadeClassifier(faces_cascade)

classificador = load_model(path_treinada, compile=False)

# classes da imagem
expressoes = ['Raiva', 'Nojo','Medo',"Feliz","Triste",'Surpreso',"Neutro"]



# EXTRAIR A FACE DA IMAGEM PARA DEPOIS RECONEHCER A EMOÇÃO
original = img.copy()
faces = detection.detectMultiScale(
    original,
    scaleFactor= 1.1,
    minNeighbors= 3,
    minSize= (20,20)
)

#PEGANDO A AREA DETECTADA

semcor = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
area_rosto =[]

# pego apenas as faces reconehcidas
v=[]
for i , vl in enumerate(faces):
    # pprint(vl)
    v.append(vl)
    # pprint(v[3])

    area_rosto.append(semcor[v[i][1]:v[i][1] +v[i][2], v[i][0]:v[i][0]+ v[i][3] ])

def redimensiona(rosto):

    # vou redimensionar por conta do processamento
    rosto = cv2.resize(rosto, (48, 48))
    # transformo a codificação do rosto em float e transformo para um numero entre 0 e 1
    rosto = rosto.astype('float')
    rosto = rosto / 255
    rosto = img_to_array(rosto)
    rosto = np.expand_dims(rosto, axis=0)
    return rosto


def previsao(var):
    # probabilidade para cada uma das classes (expressoes)
    cls = classificador.predict(var)[0]
    # pprint(cls)
    provavel_emocao = np.max(cls)

    provavel = expressoes[cls.argmax()]

    return provavel, cls


# pprint(area_rosto)
# lop de todos os rostos encontrados
for i , val in enumerate(area_rosto):

    rosto = redimensiona(val)
    em = previsao(rosto)
    emocao = em[0]
    pred = em[1]
    pprint(pred)
    pprint(expressoes)
    v = v[i]

    #apresentar a imagem com a emoção marcando o rosto
    cv2.putText(
        original,
        emocao,
        (v[0], v[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0,0,255),
        2,
        cv2.LINE_AA
    )
    cv2.rectangle(

        original,
        (v[0], v[1]),
        (v[0] + v[2] , v[1] + v[3]),
        (0,0,255),
        2
    )



    probabilidades = np.ones((250,300,3), dtype='uint8') * 255
    # se eu tiver apenas uma face
    if len(area_rosto) == 1:
        for( g , (emo, prob)) in enumerate(zip(expressoes, pred)):
            print(g , emo, prob * 100)
            text = "{}: {:.2f}%".format(emo, prob * 100)
            larg = int(prob * 300)
            cv2.rectangle(probabilidades, (7, (g * 35) + 5), (larg, (g * 35) + 35), (200, 250, 20), -1)
            cv2.putText(probabilidades, text, (10, (g * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1,
                        cv2.LINE_AA)


cv2.imshow('pajezera',original)
cv2.imshow('emoções',probabilidades)

k = cv2.waitKey(0)