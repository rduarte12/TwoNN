import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SGConv
from torch_geometric.nn import APPNP
from torch.nn import Linear
import torch_geometric.transforms as T
import random
import numpy as np
import math
import os
import sys
import math
import random
import gc
from typing import Optional
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F
import torch_sparse
from torch_scatter import scatter_add
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import natsort
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import BallTree
from sklearn.metrics import classification_report, confusion_matrix
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.nn import GCNConv, SGConv, APPNP
from torch_geometric.utils import (
    add_remaining_self_loops,
    add_self_loops as add_self_loops_fn,
    is_torch_sparse_tensor,
    scatter,
    spmm,
    to_edge_index
)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.sparse import set_sparse_value
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, SparseTensor
from grande_models import GrandeSGConv, GrandeAPPNP
from torch_geometric.nn.conv.gatv2_conv import GATv2Conv 


class GrandeSGC(torch.nn.Module):
    def __init__(self, num_features, num_classes, sigma=0.2):
        super(GrandeSGC, self).__init__()

        self.conv1 = GrandeSGConv(num_features, num_classes, K=2, cached=False, sigma=sigma)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        return F.log_softmax(x, dim=1)  

class SGC(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(SGC, self).__init__()
        self.conv1 = SGConv(num_features, num_classes, K=2, cached=True)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        return F.log_softmax(x, dim=1)
    
class GrandeAPPNPNet(torch.nn.Module):
    def __init__(self, num_features, num_classes, neurons = 256, sigma=0.2):
        super(GrandeAPPNPNet, self).__init__()
        hidden = neurons
        K = 10
        alpha = 0.1
        self.lin1 = Linear(num_features, hidden)
        self.lin2 = Linear(hidden, num_classes)
        self.prop1 = GrandeAPPNP(K, alpha, sigma=sigma)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        dropout = 0.5
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)
    
class APPNPNet(torch.nn.Module):
    def __init__(self, num_features, num_classes, neurons = 256):
        super(APPNPNet, self).__init__()
        hidden = neurons
        K = 10
        alpha = 0.1
        self.lin1 = Linear(num_features, hidden)
        self.lin2 = Linear(hidden, num_classes)
        self.prop1 = APPNP(K, alpha, cached=True)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        dropout = 0.5
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)


class GCNClassifier:
    def __init__(self, gcn_type, rks, pN, graph_type = "rec", number_neighbors=80, numbers_of_neurons=256, sigma=0.2, numeroNeuronios=256, num_epochs=200, learning_rate=0.001):
        """
        Inicializa o classificador GCN com os parâmetros necessários.

        Args:
            gcn_type (str): Tipo de GCN: "gcn_grande", "gcn_degree", "sgc_grande", "sgc_degree", "appnp_grande", "appnp_degree"
            rks (list): Lista de rankings para construção do grafo.
            pN (int): Número total de nó s no grafo.
            number_neighbors (int): Número de vizinhos para construção do grafo KNN (default=40).
            numbers_of_neurons (int): Número de neurônios na camada oculta.
            sigma (float): Parâmetro sigma para o modelo GrandeAPPNPNet.
            numeroNeuronios (int): Número de neurônios na camada oculta.
            num_epochs (int): Número de épocas de treinamento.
            learning_rate (float): Taxa de aprendizado.
        """
        # Definição dos parâmetros e dispositivo (GPU ou CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pK = number_neighbors
        self.pN = pN # Número total de nós no grafo
        self.rks = rks
        self.pLR = learning_rate # Taxa de aprendizado
        self.pNNeurons = numeroNeuronios  # Número de neurônios na camada oculta
        self.pNEpochs = num_epochs  # Número de épocas de treinamento
        self.gcn_type = gcn_type
        self.sigma = sigma
        self.graph = graph_type

    def prepare(self, test_index, train_index, features, labels):
        """
        Prepara os dados para treinamento e validação.

        Args:
            test_index (list): Índices dos nó s para teste.
            train_index (list): Índices dos nó s para treinamento.
            features (list): Lista de features de entrada para os nó s.
            labels (list): Lista de rótulos para os nó s.
        """
        # Criação de máscaras para treinamento, validação e teste
        #print('Creating masks ...')
        self.train_mask = [False] * self.pN
        self.val_mask = [False] * self.pN
        self.test_mask = [False] * self.pN

        for index in train_index:
            self.train_mask[index] = True
        for index in test_index:
            self.test_mask[index] = True

        # Conversão das máscaras para tensores PyTorch
        self.train_mask = torch.tensor(self.train_mask).to(self.device)
        self.val_mask = torch.tensor(self.val_mask).to(self.device)
        self.test_mask = torch.tensor(self.test_mask).to(self.device)


        # Definição dos rótulos
        #print('Set labels ...')
        y = labels
        self.numberOfClasses = max(y) + 1
        self.y = torch.tensor(y).to(self.device)

        # Definição das features
        # Ensure features tensor requires gradients for training
        self.x = torch.tensor(features).to(self.device)
        self.pNFeatures = len(features[0])  # Número de features por nó

        # Construção do grafo
        self.create_graph()

    def create_graph(self):
        """
        Cria o grafo KNN com base nos rankings fornecidos.
        """
        #print('Making edge list ...')
        self.top_k = self.pK
        edge_index = []
    
        # Construção da liksta d earestas baseadas nos rankings Reciprocal KNN
        if self.graph == "rec":
            refList = [[] for i in range(self.pN)]
            for img1 in range(len(self.rks)):
                for pos in range(self.top_k):
                    img2 = self.rks[img1][pos]
                    refList[img2].append(img1)
            edge_index = []
            for img1 in range(len(self.rks)):
                for pos in range(self.top_k):
                    img2 = self.rks[img1][pos]
                    if img2 in refList[img1]:
                        edge_index.append([img1, img2])
        elif self.graph == "knn":
            edge_index = []
            for img1 in range(len(self.rks)):
                for pos in range(self.top_k):
                    img2 = self.rks[img1][pos]
                    edge_index.append([img1, img2])
            edge_index = torch.tensor(edge_index)
            self.edge_index = edge_index.t().contiguous()
        elif self.graph == "two_nn":
            edge_index = []
            list_of_ks = []
            refList = []
            lst_rks = self.rks
            features = self.x.cpu().numpy()  # Certifique-se de que features seja um array numpy
            # index de vizinho a considerar segundo e terceiro pois oprimeiro é self loop
            # distancia entre terceiro e segundo vizinho
            for img1 in range(len(lst_rks)):

                feat_img1 = features[img1]
                feat_viz3 = features[lst_rks[img1][2]]
                feat_viz2 = features[lst_rks[img1][1]]

                # Similaridade Cosseno = dot(A, B) / (norm(A) * norm(B))
                sim_viz3 = np.dot(feat_viz3, feat_viz2) / (np.linalg.norm(feat_viz3) * np.linalg.norm(feat_viz2))
                sim_viz2 = np.dot(feat_viz2, feat_img1) / (np.linalg.norm(feat_viz2) * np.linalg.norm(feat_img1))
                
                # Distância Cosseno = 1 - Similaridade (limitado entre 0 e 2 para evitar erros de float)
                dist_viz3 = np.clip(1.0 - sim_viz3, 0.0, 2.0)
                dist_viz2 = np.clip(1.0 - sim_viz2, 0.0, 2.0)

                u = dist_viz3 / max(dist_viz2, 1e-10)
                
                d = 1 / (np.log(u + 1e-10) + 1e-10)

                k_img1 = np.floor((d)).astype(int)  # Ajusta o número de vizinhos com base na razão das distâncias
                k_img1 = min(k_img1, self.top_k)  # Garante que k_img1 não exceda top_k
                k_img1 = max(k_img1, 5)  # Garante que k_img1 seja pelo menos 3
                list_of_ks.append(k_img1)
                for pos in range(k_img1):
                    img2 = lst_rks[img1][pos]
                    edge_index.append([img1, img2])
            print(f"Average number of neighbors: {np.mean(list_of_ks)}")
            print(f"Max number of neighbors: {np.max(list_of_ks)}")
            print(f"Min number of neighbors: {np.min(list_of_ks)}")
        elif self.graph == "two_nn_rec":

            edge_index = []
            list_of_ks = []
            lst_rks = self.rks
            features = self.x.cpu().numpy()
            
            num_nodes = len(lst_rks)
            
            # PASSO 1: Calcular o k dinâmico (Two-NN) para cada nó
            for img1 in range(num_nodes):
                # Distância entre terceiro e segundo vizinho (assumindo index 0 = self)
                """ u = np.linalg.norm(
                    features[lst_rks[img1][3]] - features[img1]
                ) / np.linalg.norm(
                    features[lst_rks[img1][2]] - features[img1]
                ) """
                
                feat_img1 = features[img1]
                feat_viz3 = features[lst_rks[img1][2]]
                feat_viz2 = features[lst_rks[img1][1]]

                # Similaridade Cosseno = dot(A, B) / (norm(A) * norm(B))
                sim_viz3 = np.dot(feat_viz3, feat_viz2) / (np.linalg.norm(feat_viz3) * np.linalg.norm(feat_viz2))
                sim_viz2 = np.dot(feat_viz2, feat_img1) / (np.linalg.norm(feat_viz2) * np.linalg.norm(feat_img1))
                
                # Distância Cosseno = 1 - Similaridade (limitado entre 0 e 2 para evitar erros de float)
                dist_viz3 = np.clip(1.0 - sim_viz3, 0.0, 2.0)
                dist_viz2 = np.clip(1.0 - sim_viz2, 0.0, 2.0)

                # distancia Euclidiana
                """ dist_viz3 = np.linalg.norm(feat_viz3 - feat_img1)
                dist_viz2 = np.linalg.norm(feat_viz2 - feat_img1) """
                
                # Calcula u prevenindo divisão por zero no denominador
                u = dist_viz3 / max(dist_viz2, 1e-10)
                
                d = 1 / (np.log(u + 1e-10) + 1e-10)
                
                k_img1 = np.floor(d).astype(int)
                k_img1 = min(k_img1, self.top_k)  # Max threshold
                k_img1 = max(k_img1, 5)           # Min threshold
                list_of_ks.append(k_img1)
                
            """ print(f"Two-NN-Rec -> Average neighbors: {np.mean(list_of_ks):.2f}")
            print(f"Two-NN-Rec -> Max neighbors: {np.max(list_of_ks)}")
            print(f"Two-NN-Rec -> Min neighbors: {np.min(list_of_ks)}") """

            # PASSO 2: Construir a lista de referências usando o k dinâmico
            # refList[i] vai conter todos os nós que possuem 'i' em seus k vizinhos dinâmicos
            refList = [[] for _ in range(num_nodes)] # Usando num_nodes em vez de self.pN para segurança
            for img1 in range(num_nodes):
                k_img1 = list_of_ks[img1]
                for pos in range(k_img1):
                    img2 = lst_rks[img1][pos]
                    refList[img2].append(img1)

            # PASSO 3: Construir as arestas verificando a reciprocidade
            for img1 in range(num_nodes):
                k_img1 = list_of_ks[img1]
                for pos in range(k_img1):
                    img2 = lst_rks[img1][pos]
                    # Só adiciona a aresta se a reciprocidade for verdadeira
                    if img2 in refList[img1]:
                        edge_index.append([img1, img2])
            
            # PASSO 4: Converter para tensor e formato adequado
            edge_index = torch.tensor(edge_index)
            if len(edge_index) > 0:
                self.edge_index = edge_index.t().contiguous()
            else:
                # Fallback de segurança caso a reciprocidade destrua todas as arestas
                self.edge_index = torch.empty((2, 0), dtype=torch.long)

        else:
            raise ValueError(
                "Invalid graph_type. Choose from 'rec', 'knn', 'two_nn', or 'two_nn_rec'."
            )
        
        # Conversão da lista de arestas para formato PyTorch
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        self.edge_index = edge_index.t().contiguous().to(self.device)

    def build_model(self):
        """Cria e retorna um modelo novo a partir de self.gcn_type (sempre fresh)."""
        gt = str(self.gcn_type).lower()

        if gt == "sgc_grande":
            return GrandeSGC(self.pNFeatures, self.numberOfClasses, sigma=self.sigma).to(self.device)
        elif gt == "sgc_degree":
            return SGC(self.pNFeatures, self.numberOfClasses).to(self.device)
        elif gt == "appnp_grande":
            return GrandeAPPNPNet(self.pNFeatures, self.numberOfClasses, neurons=self.pNNeurons, sigma=self.sigma).to(self.device)
        elif gt == "appnp_degree":
            return APPNPNet(self.pNFeatures, self.numberOfClasses, neurons=self.pNNeurons).to(self.device)
        else:
            raise ValueError("Invalid gcn_type. Choose from 'gcn_grande', 'gcn_degree', 'sgc_grande', 'sgc_degree', 'appnp_grande', 'appnp_degree'.")


    def train_and_predict(self):
        """
        Treina o modelo GCN e realiza a predição.

        Returns:
            embeddings (Tensor): Representações geradas pelo modelo.
            pred (list): Predições para os nó s de teste.
        """

        data = Data(x=self.x.float(),             # features
                    edge_index=self.edge_index,   # graph
                    y=self.y,                     # labels
                    test_mask=self.test_mask,     # train/val/test split
                    train_mask=self.train_mask,
                    val_mask=self.val_mask)

        # Inicialização do modelo GCN
        model = self.build_model()
        #verificar se o nome do modole
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.pLR, weight_decay=5e-4)

        # Treinamento do modelo
        #print('Training...')
        model.train()
        
        for epoch in range(self.pNEpochs):
            optimizer.zero_grad()
            out = model(data)
            data.y = data.y.clone().detach().to(torch.long)  # Garantir que os rótulos estejam no tipo correto
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])  # Cálculo da perda
            loss.backward()   # Backpropagation
            optimizer.step()  # Atualização dos pesos

        # Avaliação do modelo final
        #print('Evaluating...')
        model.eval()
        with torch.no_grad():
            out = model(data)                      # executa uma única vez em modo eval
            _, pred = out.max(dim=1)               # pred contém a predição por nó
            pred = torch.masked_select(pred, data.test_mask)  # apenas nós de teste
            embeddings = out.detach()              # desconecta do grafo (economia de memória)
        return embeddings, pred.tolist()
