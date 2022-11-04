from keras.utils import Sequence
from sklearn.utils import shuffle
import numpy as np
import cv2


class DataGenerator(Sequence):

    def criar_treino(dataset, qtd_imagens_lote, dimensoes, qtd_categorias, diretorio, augment=True):
        """
        Esta função, basicamente, dado os labels recebidos, embaralha eles, e retorna esses dados pronto para usar

        :param dataset:
        :param qtd_imagens_lote:
        :param dimensoes:
        :param qtd_categorias:
        :param diretorio:
        :param augment:
        :return:
        """
        # obrigatoriamente, dimensoes precisa ter o 3º elemento com valor de 3. Porque?
        assert dimensoes[2] == 3

        while True:
            # embaralha os labels recebidos
            dataset = shuffle(dataset)

            # a variavel dataset.shape, contém a dimensão de nosso dataset. Nesse caso, ele contém x linhas (cada linha é um label/imagem) e 3 colunas (são aquelas definidas anteriormente, imagem_nome', 'categoria', 'id_categoria)
            for inicio in range(0, dataset.shape[0], qtd_imagens_lote):
                fim = min(inicio + qtd_imagens_lote, dataset.shape[0])

                # variavel irá conter as imagens que pertencem a esse lote
                lote_imagens = []

                # seleciona do inicio ao fim as linhas do dataset, e também todas as colunas
                lote_treino = dataset.iloc[inicio:fim, :]

                # preenche de 0 um array com a quantidade de linhas que a variavel lote_treino possui. A quantidade de colunas depende da variavel qtd_categorias
                lote_labels = np.zeros((lote_treino.shape[0], qtd_categorias))
                for i in range(lote_treino.shape[0]):
                    imagem = DataGenerator.__load_image__(lote_treino.iloc[i, 0], dimensoes, diretorio)

                    # TODO Ver para o que serve esse código
                    # if augment:
                    #     image = DataGenerator.augment(imagem)

                    # a imagem (objeto do cv2) é adicionado na lista de lote_imagens, cada valor é dividido por 255. Por que esse valor? O . no final muda algo?
                    lote_imagens.append(imagem / 255.)

                    # altera para 1, a linha i coluna referente ao id da categoria
                    lote_labels[i][lote_treino.iloc[i, 2]] = 1

                yield np.array(lote_imagens, np.float32), lote_labels

    def criar_validacao(dataset, qtd_imagens_lote, dimensoes, qtd_categorias, diretorio, augment=False):
        # obrigatoriamente, dimensoes precisa ter o 3º elemento com valor de 3. Porque?
        assert dimensoes[2] == 3
        while True:
            # a variavel dataset.shape, contém a dimensão de nosso dataset. Nesse caso, ele contém x linhas (cada linha é um label/imagem) e 3 colunas (são aquelas definidas anteriormente, imagem_nome', 'categoria', 'id_categoria)
            for inicio in range(0, dataset.shape[0], qtd_imagens_lote):
                fim = min(inicio + qtd_imagens_lote, dataset.shape[0])

                # variavel irá conter as imagens que pertencem a esse lote
                lote_imagens = []

                # seleciona do inicio ao fim as linhas do dataset, e também todas as colunas
                lote_validacao = dataset.iloc[inicio:fim, :]

                # preenche de 0 um array com a quantidade de linhas que a variavel lote_treino possui. A quantidade de colunas depende da variavel qtd_categorias
                lote_labels = np.zeros((lote_validacao.shape[0], qtd_categorias))
                for i in range(lote_validacao.shape[0]):
                    imagem = DataGenerator.__load_image__(lote_validacao.iloc[i, 0], dimensoes, diretorio)

                    # TODO Ver para o que serve esse código
                    # if augment:
                    #     image = DataGenerator.augment(imagem)

                    # a imagem (objeto do cv2) é adicionado na lista de lote_imagens, cada valor é dividido por 255. Por que esse valor? O . no final muda algo?
                    lote_imagens.append(imagem / 255.)

                    # altera para 1, a linha i coluna referente ao id da categoria
                    lote_labels[i][lote_validacao.iloc[i, 2]] = 1

                yield np.array(lote_imagens, np.float32), lote_labels

    def __load_image__(nome_arquivo, dimensoes, diretorio):
        # carrega a imagem do diretorio
        imagem = cv2.imread(str(diretorio) + str(nome_arquivo))

        # redimensiona a imagem para as dimensoes passadas
        imagem = cv2.resize(imagem, (dimensoes[0], dimensoes[1]))
        return imagem
