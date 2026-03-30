import random
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch_geometric.nn import GCNConv, global_mean_pool
from tqdm import tqdm


class GNN(nn.Module):
    def __init__(self, net):
        """
        Define a GNN model that takes in a Graph Neural Network (GNN) as the encoder to learn node and graph-level embeddings.

        Parameters
        ----------
        net: nn.Module
            A GNN model that encodes node and graph-level embeddings. It's forward pass should output the node-level embedding.
        """
        super().__init__()
        self.gnn_net = net

    def forward(self, data):
        """
        Define a forward pass through the RGCN model.

        Parameters
        ----------
        data:


        Returns
        -------
        z_node:

        z_graph:

        """
        z_node = self.gnn_net(data)
        z_graph = global_mean_pool(z_node, data.batch)
        return z_node, z_graph


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

    def forward(self, data):
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
        x, edge_index, edge_type, edge_weight = (
            data.x,
            data.edge_index,
            data.edge_type,
            data.edge_attr,
        )

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


# class RGCN(nn.Module):
#     def __init__(self, in_channels, hidden_channels, latent_dim, dropout=0.2):
#         """
#         Define a three RGCN-layer model.

#         Parameters
#         ----------
#         in_channels: int

#         hidden_channels: int

#         latent_dim: int

#         n_classes: int

#         dropout: float

#         """
#         super().__init__()
#         # Define dropout
#         self.dropout = dropout
#         # Define layers
#         self.layer1 = RGCNLayer(in_channels, hidden_channels)
#         self.layer2 = RGCNLayer(hidden_channels, hidden_channels)
#         self.layer3 = RGCNLayer(hidden_channels, latent_dim)

#     def forward(self, data):
#         """
#         Forward pass throguh R-GCN model.

#         Parameters
#         ----------
#         data:


#         Returns
#         -------
#         z_node:

#         z_graph:

#         """
#         x, edge_index, edge_type, edge_attr = (
#             data.x,
#             data.edge_index,
#             data.edge_type,
#             data.edge_attr,
#         )

#         # Pass through model
#         h = self.layer1(x, edge_index, edge_type, edge_attr)
#         h = F.relu(h)
#         h = F.dropout(h, p=self.dropout, training=self.training)

#         h = self.layer2(h, edge_index, edge_type, edge_attr)
#         h = F.relu(h)
#         h = F.dropout(h, p=self.dropout, training=self.training)

#         z_node = self.layer3(h, edge_index, edge_type, edge_attr)
#         z_graph = global_mean_pool(z_node, data.batch)

#         return z_node, z_graph


class GraphClassifier(nn.Module):
    def __init__(self, net):
        """
        Define a graph classifier to predict a graph class based on its graph-level embedding from a GNN model.

        Parameters
        ----------
        net: nn.Module
            The network that takes in the graph-level embedding from a GNN model and outputs the predicted class logits.
        """
        super().__init__()
        self.classifier_net = net
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        """
        Forward pass through classifier.

        Parameters
        ----------
        x:

        Returns
        -------

        """
        logits = self.classifier_net(x)

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
        return self.criterion(pred, true)


# class GraphClassifier(nn.Module):
#     def __init__(self, in_dim, n_classes):
#         """
#         Define a graph classifier to predict a graph class based on its graph-level embedding from a GNN model.

#         Parameters
#         ----------
#         in_dim: int
#             Dimension of the graph-level embedding (i.e. latent dimension from GNN model).
#         n_classes: int
#             Number of classes in the graph dataset.
#         """
#         super().__init__()
#         # Layers definition
#         self.layer1 = nn.Linear(in_dim, in_dim // 2)
#         self.actfn = nn.ReLU()
#         self.layer2 = nn.Linear(in_dim // 2, n_classes)
#         # Loss criterion
#         self.criterion = nn.CrossEntropyLoss()

#     def forward(self, x):
#         """
#         Forward pass through classifier.

#         Parameters
#         ----------
#         x:

#         Returns
#         -------

#         """
#         # Pass first layer
#         h = self.layer1(x)
#         # Activation function
#         h = self.actfn(h)
#         # Pass second layer
#         logits = self.layer2(h)

#         return logits

#     def loss(self, pred, true):
#         """
#         Define loss function for graph classifier

#         Parameters
#         ----------
#         pred:

#         true:


#         Returns
#         -------

#         """

#         return self.criterion(pred, true)


class EdgePredictor(nn.Module):
    def __init__(self, net):
        """
        Define an edge predictor to infer the edge weight between two nodes based on its node-level embedding from a GNN model.

        Parameters
        ----------
        net: nn.Module
            The network that takes in the concatenated node-level embeddings of two nodes and outputs the predicted edge weight.
        """
        super().__init__()
        self.edge_predictor_net = net
        self.criterion = nn.MSELoss()

    def forward(self, nodes):
        """
        Forward pass through predictor.

        Parameters
        ----------
        nodes:


        Returns
        -------

        """
        weight = self.edge_predictor_net(nodes)

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
        return self.criterion(pred, true)


# class EdgePredictor(nn.Module):
#     def __init__(self, in_dim):
#         """
#         Define an edge predictor to infer the edge weight between two nodes based on its node-level embedding from a GNN model.

#         Parameters
#         ----------
#         in_dim: int
#             Dimension of the node-level embedding (i.e. latent dimension from GNN model).
#         """
#         super().__init__()
#         # Layers definition
#         self.layer1 = nn.Linear(in_dim * 2, in_dim)
#         self.actfn = nn.ReLU()
#         self.layer2 = nn.Linear(in_dim, 1)
#         # Loss criterion
#         self.criterion = nn.MSELoss()

#     def forward(self, nodes):
#         """
#         Forward pass through predictor.

#         Parameters
#         ----------
#         nodes:


#         Returns
#         -------

#         """
#         # Pass first layer
#         h = self.layer1(nodes)
#         # Activation function
#         h = self.actfn(h)
#         # Pass second layer
#         weight = self.layer2(h)

#         return weight.squeeze()

#     def loss(self, pred, true):
#         """
#         Define loss function for edge predictor.

#         Parameters
#         ----------
#         pred:

#         true:


#         Returns
#         -------

#         """
#         return self.criterion(pred, true)


class GNNModel(nn.Module):
    def __init__(
        self,
        in_channels,
        n_classes,
        gnn_type="RGCN",
        gnn_net=[256, 256],
        gnn_actfn=nn.ReLU(),
        gnn_dropout=0.2,
        latent_dim=128,
        classifier_net=[64],
        classifier_actfn=nn.ReLU(),
        predictor_net=[128],
        predictor_actfn=nn.ReLU(),
    ):
        """
        Define a Graph Neural Network (GNN) model.

        Parameters
        ----------
        in_channels: int

        n_classes: int

        gnn_net: list

        gnn_actfn: nn.Module

        gnn_dropout: float

        latent_dim: int

        classifier_net: list

        classifier_actfn: nn.Module

        predictor_net: list

        predictor_actfn: nn.Module

        """
        super().__init__()

        # Defining GNN
        if gnn_type == "RGCN":
            gnn_net = [in_channels] + gnn_net + [latent_dim]
            modules = []
            for i in range(len(gnn_net) - 1):
                if i < len(gnn_net) - 2:
                    modules.append(
                        RGCNLayer(gnn_net[i], gnn_net[i + 1])
                    )  # RGCNLayer(in_channels, out_channels)
                    modules.append(gnn_actfn)
                    modules.append(nn.Dropout(gnn_dropout))
                else:
                    modules.append(RGCNLayer(gnn_net[i], gnn_net[i + 1]))

        else:
            raise NotImplementedError(
                f"GNN type {gnn_type} not implemented. Please choose from ['RGCN']."
            )

        self.gnn = GNN(nn.Sequential(*modules))

        # Defining classifier
        classifier_net = [latent_dim] + classifier_net + [n_classes]
        modules = []
        for i in range(len(classifier_net) - 1):
            if i < len(classifier_net) - 2:
                modules.append(nn.Linear(classifier_net[i], classifier_net[i + 1]))
                modules.append(classifier_actfn)
            else:
                modules.append(nn.Linear(classifier_net[i], classifier_net[i + 1]))

        self.classifier = GraphClassifier(nn.Sequential(*modules))

        # Defining edge predictor
        predictor_net = [latent_dim * 2] + predictor_net + [1]
        modules = []
        for i in range(len(predictor_net) - 1):
            if i < len(predictor_net) - 2:
                modules.append(nn.Linear(predictor_net[i], predictor_net[i + 1]))
                modules.append(predictor_actfn)
            else:
                modules.append(nn.Linear(predictor_net[i], predictor_net[i + 1]))

        self.edge_predictor = EdgePredictor(nn.Sequential(*modules))

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

            batch.x = batch.x.detach().requires_grad_(False)
            batch.edge_attr = batch.edge_attr.detach().requires_grad_(False)
            batch.y = batch.y.detach()

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
