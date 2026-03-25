import random
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch_geometric.nn import GCNConv, global_mean_pool
from tqdm import tqdm


class RGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Define a single layer of a Relational Graph Convolution Layer

        Parameters
        ----------
        in_channels: int
            Description
        out_channels: int
            Description
        """
        super().__init__()

        self.conv_pos = GCNConv(in_channels, out_channels)
        self.conv_neg = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index, edge_type, edge_weight):
        """
        Defina a forward pass through the R-GCN layer.

        Parameters
        ----------
        x:

        edge_index:

        edge_type:

        edge_weight:


        Returns
        -------

        """
        # Masking positive and negative edges
        pos_mask = edge_type == 0
        neg_mask = edge_type == 1

        # Positive stream
        if pos_mask.any():
            pos_out = self.conv_pos(x, edge_index[:, pos_mask], edge_weight[pos_mask])
        else:
            pos_out = torch.zeros(
                x.size(0), self.conv_pos.out_channels, device=x.device
            )

        # Negative stream
        if neg_mask.any():
            neg_out = self.conv_neg(x, edge_index[:, neg_mask], edge_weight[neg_mask])
        else:
            neg_out = torch.zeros(
                x.size(0), self.conv_neg.out_channels, device=x.device
            )

        return pos_out + neg_out


class RGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim, dropout=0.2):
        """
        Define a three RGCN-layer model.

        Parameters
        ----------
        in_channels: int

        hidden_channels: int

        latent_dim: int

        n_classes: int

        dropout: float

        """
        super().__init__()
        # Define dropout
        self.dropout = dropout
        # Define layers
        self.layer1 = RGCNLayer(in_channels, hidden_channels)
        self.layer2 = RGCNLayer(hidden_channels, hidden_channels)
        self.layer3 = RGCNLayer(hidden_channels, latent_dim)

    def forward(self, data):
        """
        Forward pass throguh R-GCN model.

        Parameters
        ----------
        data:


        Returns
        -------
        z_node:

        z_graph:

        """
        x, edge_index, edge_type, edge_attr = (
            data.x,
            data.edge_index,
            data.edge_type,
            data.edge_attr,
        )

        # Pass through model
        h = self.layer1(x, edge_index, edge_type, edge_attr)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.layer2(h, edge_index, edge_type, edge_attr)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        z_node = self.layer3(h, edge_index, edge_type, edge_attr)
        z_graph = global_mean_pool(z_node, data.batch)

        return z_node, z_graph


class GraphClassifier(nn.Module):
    def __init__(self, in_dim, n_classes):
        """
        Define a graph classifier to predict a graph class based on its graph-level embedding from a GNN model.

        Parameters
        ----------
        in_dim: int
            Dimension of the graph-level embedding (i.e. latent dimension from GNN model).
        n_classes: int
            Number of classes in the graph dataset.
        """
        super().__init__()
        # Layers definition
        self.layer1 = nn.Linear(in_dim, in_dim // 2)
        self.actfn = nn.ReLU()
        self.layer2 = nn.Linear(in_dim // 2, n_classes)

    def forward(self, x):
        """
        Forward pass through classifier.

        Parameters
        ----------
        x:

        Returns
        -------

        """
        # Pass first layer
        h = self.layer1(x)
        # Activation function
        h = self.actfn(h)
        # Pass second layer
        logits = self.layer2(h)

        return logits

    def loss(self, pred, true):
        """
        Define loss function for graph classifier

        Parameters
        ----------
        pred:

        true:


        Returns
        -------

        """
        loss = nn.CrossEntropyLoss()

        return loss(pred, true)


class EdgePredictor(nn.module):
    def __init__(self, in_dim):
        """
        Define an edge predictor to infer the edge weight between two nodes based on its node-level embedding from a GNN model.

        Parameters
        ----------
        in_dim: int
            Dimension of the node-level embedding (i.e. latent dimension from GNN model).
        """
        super().__init__()
        # Layers definition
        self.layer1 = nn.Linear(in_dim * 2, in_dim)
        self.actfn = nn.ReLU()
        self.layer2 = nn.Linear(in_dim, 1)

    def forward(self, nodeA, nodeB):
        """
        Forward pass through predictor.

        Parameters
        ----------
        nodeA:

        nodeB:


        Returns
        -------

        """
        # Concatenate node embeddings
        x = torch.cat([nodeA, nodeB], dim=1)
        # Pass first layer
        h = self.layer1(x)
        # Activation function
        h = self.actfn(h)
        # Pass second layer
        weight = self.layer2(h)

        return weight.squeeze()

    def loss(self, pred, true):
        """
        Define loss function for edge predictor.

        Parameters
        ----------
        pred:

        true:


        Returns
        -------

        """
        loss = nn.MSELoss()

        return loss(pred, true)


class GNNModel(nn.Module):
    def __init__(self, gnn, classifier, edge_predictor):
        """
        Define a Graph Neural Network (GNN) model.

        Parameters
        ----------
        gnn:

        classifier:

        edge_predictor:

        """
        super().__init__()

        self.gnn = gnn
        self.classifier = classifier
        self.edge_predictor = edge_predictor

    def encode(self, data):

        return self.gnn(data)

    def forward(self, data):

        z_node, z_graph = self.gnn(data)

        edge_index = data.edge_index

        src, dst = edge_index[0], edge_index[1]
        edge_features = torch.cat([z_node[src], z_node[dst]], dim=1)
        pred_edge = self.edge_predictor(edge_features)

        pred_class = self.classifier(z_graph)

        return pred_edge, pred_class


def train_gnn(
    model,
    train_loader,
    device,
    epochs=100,
    lr=1e-3,
    w_l2=1e-4,
    w_classifier=1.0,
    w_edge_pred=0.5,
):

    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=w_l2)

    total_steps = len(train_loader) * epochs

    progress_bar = tqdm(
        range(total_steps),
        desc="Training GNN model",
    )

    for epoch in range(epochs):

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            # Prediction
            pred_edge, pred_class = model(batch)
            # Classifier loss
            class_loss = w_classifier * model.classifier.loss(pred_class, batch.y)
            # Edge predictor loss
            edge_loss = w_edge_pred * model.edge_predictor.loss(
                pred_edge, batch.edge_attr
            )

            loss = class_loss + edge_loss

            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(
                total_loss=f"{loss.item():.4f}",
                class_loss=f"{class_loss.item():.4f}",
                edge_loss=f"{edge_loss.item():.4f}",
                epoch=f"{epoch}/{epochs + 1}",
            )
            progress_bar.update()

    progress_bar.close()


# class SAE(nn.Module):

# class GEA(nn.Module):
