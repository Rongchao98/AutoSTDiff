import os.path
import random
from collections import Counter
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import pandas as pd
from discreteDataset import MYDATA
import numpy as np


def haversine(lat1, lon1, lat2, lon2):
    # Radius of the Earth in kilometers
    R = 6371.0

    # Convert coordinates from degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # Differences in coordinates
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine formula
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c  # in kilometers
    return distance


# Define the Enhanced GNN Model
class GCNModel(nn.Module):
    def __init__(self, num_features, hidden_dim, embedding_dim):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, embedding_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = self.dropout(x)
        x = self.conv3(x, edge_index)

        return x


class Classifier(nn.Module):
    def __init__(self, embedding_dim, num_locations, hidden_dim=64):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(hidden_dim, num_locations)

    def forward(self, embeddings):
        x = F.relu(self.bn1(self.fc1(embeddings)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class D2C(nn.Module):
    def __init__(self,
                traj_data,
                gps_data,
                device=None,
                hidden_dim=64,
                embedding_dim=32,
                top_k=8,
                graph_d_path='graph/transition_graph.pt',
                graph_t_path='graph/distance_graph.pt',
                embedding_path='graph/l_embeddings.pt'):
        super().__init__()
        self.num_locations = len(gps_data)
        if os.path.exists(graph_d_path) and os.path.exists(graph_t_path):
            graph_d = torch.load(graph_d_path)
            graph_t = torch.load(graph_t_path)
        else:
            # 1. Transition
            transition_counts = Counter()
            for trajectory in traj_data:
                for i in range(len(trajectory) - 1):
                    source = trajectory[i][0]
                    target = trajectory[i + 1][0]
                    transition_counts[(source, target)] += 1

            # Create edges and edge attributes for transition graph
            transition_edges = []
            transition_edge_attr = []
            for (source, target), count in transition_counts.items():
                transition_edges.append([source, target])
                transition_edge_attr.append(count)

            transition_edge_index = torch.tensor(transition_edges, dtype=torch.long).t().contiguous()
            transition_edge_attr = torch.tensor(transition_edge_attr, dtype=torch.float).unsqueeze(1)

            # 2. Distance: Calculate all distances and store them
            num_locations = len(gps_data)
            distances = np.zeros((num_locations, num_locations))
            for i in range(num_locations):
                for j in range(num_locations):
                    if i != j:
                        distances[i][j] = haversine(gps_data[i][0], gps_data[i][1],
                                             gps_data[j][0], gps_data[j][1])

            # Select top-k nearest locations and create edges
            poi_poi_edges = []
            distance_edge_attr = []
            for i in range(num_locations):
                nearest_indices = np.argsort(distances[i])[1:top_k + 1]
                for j in nearest_indices:
                    poi_poi_edges.append([i, j])
                    distance_edge_attr.append(distances[i][j])

            distance_edge_index = torch.tensor(poi_poi_edges, dtype=torch.long).t().contiguous()
            distance_edge_attr = torch.tensor(distance_edge_attr, dtype=torch.float).unsqueeze(1)

            node_features = torch.eye(num_locations)

            graph_t = Data(x=node_features, edge_index=transition_edge_index, edge_attr=transition_edge_attr)
            graph_d = Data(x=node_features, edge_index=distance_edge_index, edge_attr=distance_edge_attr)
            torch.save(graph_t, graph_t_path)
            torch.save(graph_d, graph_d_path)

        self.device = device
        self.gcn_t = GCNModel(self.num_locations, hidden_dim, embedding_dim).to(device)
        self.gcn_d = GCNModel(self.num_locations, hidden_dim, embedding_dim).to(device)
        self.decoder = Classifier(embedding_dim, self.num_locations).to(device)
        self.graph_d = graph_d.to(device)
        self.graph_t = graph_t.to(device)
        self.embedding_path = embedding_path
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embeddings = torch.load(embedding_path).to(device) if os.path.exists(embedding_path) else None
        if self.embeddings is None:
            self.train()

    def pre_train(self, l_r=0.01, epoch=2):
        optimizer = torch.optim.Adam(list(self.gcn_d.parameters()) + list(self.gcn_t.parameters())
                                     + list(self.decoder.parameters()),
                                     lr=l_r)
        target = torch.arange(self.num_locations).to(self.device)

        for epoch in range(epoch):
            self.gcn_t.train()
            self.gcn_d.train()
            self.decoder.train()
            optimizer.zero_grad()

            emb_t = self.gcn_t(self.graph_t)
            emb_d = self.gcn_d(self.graph_d)
            out = self.decoder(emb_d + emb_t)

            # Compute loss and update
            loss = F.nll_loss(out, target)
            loss.backward()
            optimizer.step()

            print(f"Embedding and rounding training: Epoch {epoch}, Loss: {loss.item()}")

        emb_t = self.gcn_t(self.graph_t)
        emb_d = self.gcn_d(self.graph_d)
        self.embeddings = emb_d + emb_t
        # torch.save(self.embeddings, self.embedding_path)

    # ids -> (B, seq_len),
    # return (B, seq_len, D)
    def embed(self, ids):
        B, Seq_len = ids.shape[0], ids.shape[1]
        emb_t = self.gcn_t(self.graph_t)
        emb_d = self.gcn_d(self.graph_d)
        embeddings = emb_d + emb_t

        valid_node_ids = torch.clamp(ids, 0)
        selected_embeddings = embeddings[valid_node_ids]
        mask = (ids == -1).to(self.device)
        selected_embeddings[mask] = torch.zeros(self.embedding_dim).to(self.device)

        return selected_embeddings.view(B, Seq_len, self.embedding_dim)

    def round(self, emb):
        return torch.argmax(self.decoder(emb), dim=1)

    def forward(self, x_start, target):
        out = self.decoder(x_start)
        return F.nll_loss(out, target)


if __name__ == '__main__':
    # Dummy trajectory data and GPS data for illustration
    data = MYDATA("FourSquare_NYC", 0)
    gps = data.GPS  # Replace with your actual GPS data file
    traj = data.trajectories
    device = torch.device("cpu")
    d2c = D2C(traj, gps, device=device)
    emb = torch.randn(16, 16)
    res = d2c.round(emb)
    location_ids = torch.argmax(res, dim=1)
    ori_embs = d2c.embed(location_ids)

    # graph_d_path = 'graph/transition_graph.pt'
    # graph_t_path = 'graph/distance_graph.pt'
    # num_features = len(gps_data)  # Latitude and longitude
    # hidden_dim = 64
    # embedding_dim = 32  # Example embedding dimension
    # num_locations = len(gps_data)
    # top_k = 8  # Set the value of k
    #
    # if os.path.exists(graph_d_path) and os.path.exists(graph_t_path):
    #     graph_d = torch.load(graph_d_path)
    #     graph_t = torch.load(graph_t_path)
    # else:
    #     # 1. Transition
    #     # Count transitions
    #     transition_counts = Counter()
    #     for trajectory in traj_data:
    #         for i in range(len(trajectory) - 1):
    #             source = trajectory[i][0]
    #             target = trajectory[i + 1][0]
    #             transition_counts[(source, target)] += 1
    #
    #     # Create edges and edge attributes for transition graph
    #     transition_edges = []
    #     transition_edge_attr = []
    #     for (source, target), count in transition_counts.items():
    #         transition_edges.append([source, target])
    #         transition_edge_attr.append(count)
    #
    #     transition_edge_index = torch.tensor(transition_edges, dtype=torch.long).t().contiguous()
    #     transition_edge_attr = torch.tensor(transition_edge_attr, dtype=torch.float).unsqueeze(1)
    #
    #     # 2. Distance: Calculate all distances and store them
    #     distances = np.zeros((num_locations, num_locations))
    #     for i in range(num_locations):
    #         for j in range(num_locations):
    #             if i != j:
    #                 distance = haversine(gps_data[i][0], gps_data[i][1],
    #                                      gps_data[j][0], gps_data[j][1])
    #
    #     # Select top-k nearest locations and create edges
    #     poi_poi_edges = []
    #     distance_edge_attr = []
    #     for i in range(num_locations):
    #         nearest_indices = np.argsort(distances[i])[1:top_k + 1]
    #         for j in nearest_indices:
    #             poi_poi_edges.append([i, j])
    #             distance_edge_attr.append(distances[i][j])
    #
    #     distance_edge_index = torch.tensor(poi_poi_edges, dtype=torch.long).t().contiguous()
    #     distance_edge_attr = torch.tensor(distance_edge_attr, dtype=torch.float).unsqueeze(1)
    #
    #     # num_locations = len(gps_data)
    #     # distance_edges = []
    #     # distance_edge_attr = []
    #     #
    #     # for i in range(num_locations):
    #     #     for j in range(num_locations):
    #     #         if i != j:
    #     #             # distance = haversine(gps_data[i][0], gps_data[i][1],
    #     #             #                      gps_data[j][0], gps_data[j][1])
    #     #             distance = random.randint(10, 15)
    #     #             distance_edges.append([i, j])
    #     #             distance_edge_attr.append(distance)
    #     #
    #     # distance_edge_index = torch.tensor(distance_edges, dtype=torch.long).t().contiguous()
    #     # distance_edge_attr = torch.tensor(distance_edge_attr, dtype=torch.float).view(-1, 1)
    #
    #     node_features = torch.eye(num_locations)
    #
    #     graph_t = Data(x=node_features, edge_index=transition_edge_index)
    #     graph_d = Data(x=node_features, edge_index=distance_edge_index, edge_attr=distance_edge_attr)
    #     torch.save(graph_t, graph_t_path)
    #     torch.save(graph_d, graph_d_path)
    #
    # # # Normalize and convert GPS data to tensor for node features
    # # gps_data = pd.DataFrame(gps_data, columns=['latitude', 'longitude'])
    # #
    # # # Apply min-max normalization to latitude and longitude separately
    # # gps_data['latitude'] = (gps_data['latitude'] - gps_data['latitude'].min()) / (
    # #             gps_data['latitude'].max() - gps_data['latitude'].min())
    # # gps_data['longitude'] = (gps_data['longitude'] - gps_data['longitude'].min()) / (
    # #             gps_data['longitude'].max() - gps_data['longitude'].min())
    # #
    # # # Convert normalized GPS data to a tensor for node features
    # # node_features = torch.tensor(gps_data.values, dtype=torch.float)
    #
    # # gps_data_normalized = (gps_data - gps_data.min()) / (gps_data.max() - gps_data.min())
    # # node_features = torch.tensor(gps_data_normalized, dtype=torch.float)
    #
    # # Initialize models
    # gcn_t = GCNModel(num_features, hidden_dim, embedding_dim)
    # gcn_d = GCNModel(num_features, hidden_dim, embedding_dim)
    # decoder = Classifier(embedding_dim, num_locations)
    #
    # # Example: Training loop
    # optimizer = torch.optim.Adam(list(gcn_d.parameters()) + list(gcn_t.parameters()) + list(decoder.parameters()), lr=0.01)
    # target = torch.arange(len(gps_data))
    #
    # for epoch in range(200):
    #     gcn_t.train()
    #     gcn_d.train()
    #     decoder.train()
    #     optimizer.zero_grad()
    #
    #     emb_t = gcn_t(graph_t)
    #     emb_d = gcn_d(graph_d)
    #     out = decoder(emb_d + emb_t)
    #
    #     # Compute loss and update
    #     loss = F.nll_loss(out, target)
    #     loss.backward()
    #     optimizer.step()
    #
    #     print(f"Epoch {epoch}, Loss: {loss.item()}")

