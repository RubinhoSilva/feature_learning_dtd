import os
import time
from os.path import isfile

import imgaug as ia
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.applications import *
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Input, Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam

from data_generator import DataGenerator


# warnings.filterwarnings("ignore", category=DeprecationWarning)


def pre_processing(path_diretorio):
    """
        Esta função faz o pre processamento dos dados, ou seja, ela carrega e organiza tudo que é necessário para
        executar as arquituras de extração de caracteristicas.

        :param path_diretorio: Caminho até as imagens
        :return: Retorna uma lista contendo o label de todas as imagens de todas as categorias
    """

    # lista contendo todas as categorias de textura. Obtem os valores das pastas disponibilizadas
    categorias = os.listdir(path_diretorio)

    # obtem a quantidade de categorias
    qtd_categorias = len(categorias)

    labels = []
    # percorre as categorias. Cada uma recebe um ID, de acordo com sua posição na lista
    for id_categoria, categoria in enumerate(categorias):
        # obtem todos os arquivos dessa categoria, e percorre por eles no for
        for arquivo in os.listdir(os.path.join(path_diretorio, categoria)):
            # verifica se realmente é um arquivo
            # if isfile(arquivo):
            # adiciona e uma nova lista os dados formatados. Cada elemento esta da seguinte forma
            # ['categoria/nome_arquivo.extensao_arquivo', 'categoria', id_categoria]
            labels.append(['{}/{}'.format(categoria, arquivo), categoria, id_categoria])

    # transforma a lista para que ele tenha cabeçalho
    return pd.DataFrame(labels, columns=['imagem_nome', 'categoria', 'id_categoria']), qtd_categorias


def __build_model__(ModelName, qtd_categorias, dimensoes_entrada, froze=0.8):
    inp = Input(shape=dimensoes_entrada)

    if ModelName == "Xception":
        base_model = Xception(input_tensor=inp, include_top=False, weights='imagenet')
    elif ModelName == "VGG16":
        base_model = VGG16(input_tensor=inp, include_top=False, weights='imagenet')
    elif ModelName == "VGG19":
        base_model = VGG19(input_tensor=inp, include_top=False, weights='imagenet')
    elif ModelName == "ResNet50":
        base_model = ResNet50(input_tensor=inp, include_top=False, weights='imagenet')
    # ResNet152, ResNeXt50, ResNeXt101
    elif ModelName == "InceptionV3":
        base_model = InceptionV3(input_tensor=inp, include_top=False, weights='imagenet')
    elif ModelName == "InceptionResNetV2":
        base_model = InceptionResNetV2(input_tensor=inp, include_top=False, weights='imagenet')
    elif ModelName == "MobileNet":
        base_model = MobileNet(input_tensor=inp, include_top=False, weights='imagenet')
    elif ModelName == "MobileNetV2":
        base_model = MobileNetV2(input_tensor=inp, include_top=False, weights='imagenet')
    elif ModelName == "DenseNet121":
        base_model = DenseNet121(input_tensor=inp, include_top=False, weights='imagenet')
    elif ModelName == "DenseNet201":
        base_model = DenseNet201(input_tensor=inp, include_top=False, weights='imagenet')
    elif ModelName == "NASNetMobile":
        base_model = NASNetMobile(input_tensor=inp, include_top=False, weights='imagenet')
    elif ModelName == "NASNetLarge":
        base_model = NASNetLarge(input_tensor=inp, include_top=False, weights='imagenet')
    elif ModelName == "EfficientNetV2L":
        base_model = EfficientNetV2L(input_tensor=inp, include_top=False, weights='imagenet')
    elif ModelName == "ResNet50":
        base_model = ResNet50(input_tensor=inp, include_top=False, weights='imagenet')

    # frozen the first .froze% layers
    NtrainableLayers = round(len(base_model.layers) * froze)
    for layer in base_model.layers[:NtrainableLayers]:
        layer.trainable = False
    for layer in base_model.layers[NtrainableLayers:]:
        layer.trainable = True

    x_model = base_model.output
    x_model = GlobalAveragePooling2D(name='globalaveragepooling2d')(x_model)
    ## OR ##
    # x_model = Flatten()(x_model)
    # x_model = BatchNormalization()(x_model)
    # x_model = Dropout(0.5)(x_model)

    predictions = Dense(qtd_categorias, activation='softmax', name='output_layer')(x_model)
    model = Model(inputs=base_model.input, outputs=predictions)

    return model


def training(labels, qtd_imagens_lote, altura_imagem, largura_imagem, arquitetura, valor_paciencia, qtd_epocas,
             taxa_aprendizagem, qtd_categorias, dimensoes_entrada, diretorio):
    # variavel irá conter a acuracia de todos os folds
    acuracias = []

    # esse for percorre os 10 folds
    for i in range(1, 11):
        print("Fold {}".format(i))

        if os.path.exists("{}/Fold_{}.txt".format(arquitetura, i)):
            arquivo_acurarias = open("{}/Fold_{}.txt".format(arquitetura, i), "r")
            print("\nFold {}/{} acurácia = {}".format(i, 10, arquivo_acurarias.readline()))
            arquivo_acurarias.close()
            continue

        labels_treino = __get_labels__('train', i, labels)
        labels_validacao = __get_labels__('val', i, labels)

        # Não sei o que faz
        K.clear_session()

        # arquivo para em que salvam o melhores resultados
        arquivo_melhor_model = "{}/weights.best.hdf5".format(arquitetura)

        checkpoint = ModelCheckpoint(
            arquivo_melhor_model,
            monitor='val_accuracy',
            verbose=1,
            save_best_only=True,
            mode='max',
            save_weights_only=True
        )

        reduceLROnPlat = ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.2,
            patience=valor_paciencia,
            verbose=1,
            mode='max',
            cooldown=2,
            min_lr=1e-7
        )

        early = EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=valor_paciencia * 2
        )

        callbacks_list = [checkpoint, reduceLROnPlat]

        model = __build_model__(arquitetura, qtd_categorias, dimensoes_entrada)

        if os.path.exists("{}/weights.best.hdf5".format(arquitetura)):
            model.load_weights("{}/weights.best.hdf5".format(arquitetura))

        model.compile(optimizer=Adam(lr=taxa_aprendizagem), loss='categorical_crossentropy', metrics=['accuracy'])

        dados_treino = DataGenerator.criar_treino(labels_treino, qtd_imagens_lote, (altura_imagem, largura_imagem, 3),
                                                  qtd_categorias, diretorio, augment=True)
        dados_validacao = DataGenerator.criar_validacao(labels_validacao, qtd_imagens_lote,
                                                        (altura_imagem, largura_imagem, 3), qtd_categorias, diretorio,
                                                        augment=False)

        # Em cada epoca (epoch), irá percorrer de forma aleatoria os dados de treino, e em seguida serão validados. A melhor perfomance é salva.

        history = model.fit_generator(
            generator=dados_treino,
            steps_per_epoch=labels_treino.shape[0] // qtd_imagens_lote,
            validation_data=dados_validacao,
            validation_steps=labels_validacao.shape[0] // qtd_imagens_lote,
            epochs=qtd_epocas,
            callbacks=callbacks_list
        )

        # print("\nFine tuning")
        # for l in model.layers[:]:
        #     l.trainable = True
        #
        # model.compile(optimizer=SGD(lr=Learning_rate * 1e-02, momentum=0.9), loss='categorical_crossentropy',
        #               metrics=['accuracy'])
        #
        # history = model.fit_generator(
        #     generator=train_gen,
        #     steps_per_epoch=train.shape[0] // Batch_size,
        #     validation_data=val_gen,
        #     validation_steps=val.shape[0] // Batch_size,
        #     epochs=Nepochs // 2,
        #     callbacks=callbacks_list
        # )
        # trainig_history.append(history)

        print('Carregando melhor modelo')
        model.load_weights(arquivo_melhor_model)

        # TODO A acurácia é medida pelos dados de testes?
        labels_teste = __get_labels__('test', i, labels)
        dados_teste = DataGenerator.criar_validacao(labels_teste, qtd_imagens_lote, (altura_imagem, largura_imagem, 3),
                                                    qtd_categorias, diretorio, augment=False)
        acuracia = model.evaluate_generator(dados_teste, steps=labels_teste.shape[0] // qtd_imagens_lote)[1]
        acuracias.append(acuracia)

        # 10 é pq são 10 folds
        print("\nFold {}/{} acurácia = {:.3f}".format(i, 10, acuracia))

        arquivo_acurarias = open("{}/Fold_{}.txt".format(arquitetura, i), "x")
        arquivo_acurarias.write(str(acuracia))
        arquivo_acurarias.close()

    print(acuracias)


def __get_labels__(tipo, fold, todos_labels):
    """

    :type tipo: Define o tipo dos labels, se é para treino, validação ou teste
    :param fold: Numero do fold em esta obtendo os dados
    :param todos_labels: Contém todos os labels de todas as categorias
    :return: Retorna uma lista contendo apenas os labels do tipo e do fold escolhido
    """
    # carrega o arquivo contendo o label (path até a imagem) das imagens
    with open('dtd/labels/{}{}.txt'.format(tipo, fold)) as arquivo:
        # lê todas as linhas do arquivo para a variavel label_imagens
        label_imagens = arquivo.readlines()

    imagens = []
    for label in label_imagens:
        # constroi uma lista contendo o caminho para as imagens
        imagens.append(label.strip())
        # a função strip remove espaços e caracteres especiais

    # de todos os labels, filtra para retornar apenas os que pertencem a este fold
    return todos_labels[todos_labels['imagem_nome'].isin(imagens)]


def main(path_diretorio, seed, arquitetura, altura_imagem, largura_imagem, qtd_imagens_lote, valor_paciencia,
         qtd_epocas, taxa_aprendizagem):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    ia.seed(seed)

    # NÃO SEI O QUE FAZ AINDA
    pd.set_option('max_colwidth', 400)
    plt.rcParams['figure.figsize'] = [16, 10]
    plt.rcParams['font.size'] = 16

    # variavel que vai dizer quanto tempo demorou pra executar tudo
    t_start = time.time()

    # NÃO SEI O QUE FAZ AINDA
    if K.image_data_format() == 'channels_first':
        dimensoes_entrada = (3, largura_imagem, altura_imagem)
    else:
        dimensoes_entrada = (largura_imagem, altura_imagem, 3)

    todos_labels, qtd_categorias = pre_processing(path_diretorio)
    training(todos_labels, qtd_imagens_lote, altura_imagem, largura_imagem, arquitetura, valor_paciencia, qtd_epocas,
             taxa_aprendizagem, qtd_categorias, dimensoes_entrada, path_diretorio)

    t_finish = time.time()
    print(f"Tempo de execução: {(t_finish - t_start) / 3600} horas")
