import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import umap
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch_geometric.nn import GCNConv, global_mean_pool
from tqdm import tqdm


class GNN(nn.Module):
    def __init__(self, gnn_layers, gnn_actfn, gnn_dropout):
        """
        Define a GNN model that takes in a Graph Neural Network (GNN) as the encoder to learn node and graph-level embeddings.

        Parameters
        ----------
        net: nn.Module
            A GNN model that encodes node and graph-level embeddings. It's forward pass should output the node-level embedding.
        """
        super().__init__()
        self.gnn_layers = gnn_layers
        self.gnn_actfn = gnn_actfn
        self.gnn_dropout = nn.Dropout(gnn_dropout)

    def forward(self, data, x=None):
        """
        Define a forward pass through the RGCN model.

        Parameters
        ----------
        data:
        x : torch.Tensor, optional
            Pre-projected node features. When provided, used instead of data.x
            so that GNNModel can apply input_proj before the RGCN stack.

        Returns
        -------
        z_node:

        z_graph:

        """
        x = data.x if x is None else x
        edge_index, edge_type, edge_weight = (
            data.edge_index,
            data.edge_type,
            data.edge_attr,
        )

        for i, layer in enumerate(self.gnn_layers):
            x = layer(x, edge_index, edge_type, edge_weight)
            if i < len(self.gnn_layers) - 1:
                x = self.gnn_actfn(x)
                x = self.gnn_dropout(x)

        z_node = x
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
        input_proj_dim=128,
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

        input_proj_dim: int or None
            If set, node features are split into expression and embedding parts.
            The embedding part (x[:, 1:], i.e. all dims except the first) is
            projected with Linear(in_channels-1, input_proj_dim-1) + ReLU, then
            the expression scalar (x[:, 0:1]) is concatenated back explicitly:

                x_proj = cat([expr, proj(esm_emb)], dim=1)  → [N, input_proj_dim]

            Keeping expression as a dedicated dimension ensures it is never
            compressed away by the projection weights being dominated by the
            1280 ESM-2 dimensions. Set to None to pass raw features directly.

        classifier_net: list

        classifier_actfn: nn.Module

        predictor_net: list

        predictor_actfn: nn.Module

        """
        super().__init__()

        # Input projection: compress embedding dims, preserve expression explicitly.
        # x layout from gene_networks_to_pyg: [expr (1) | static_embedding (in_channels-1)]
        if input_proj_dim is not None and in_channels != input_proj_dim:
            self.input_proj = nn.Sequential(
                nn.Linear(in_channels - 1, input_proj_dim - 1),
                nn.ReLU(),
            )
            self._proj_active = True
            gnn_in = input_proj_dim
        else:
            self.input_proj = nn.Identity()
            self._proj_active = False
            gnn_in = in_channels

        # Defining GNN
        if gnn_type == "RGCN":
            gnn_net = [gnn_in] + gnn_net + [latent_dim]
            gnn_layers = nn.ModuleList()
            for i in range(len(gnn_net) - 1):
                gnn_layers.append(RGCNLayer(gnn_net[i], gnn_net[i + 1]))

        else:
            raise NotImplementedError(
                f"GNN type {gnn_type} not implemented. Please choose from ['RGCN']."
            )

        self.gnn = GNN(gnn_layers, gnn_actfn, gnn_dropout)

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

    def _project(self, x):
        if self._proj_active:
            expr = x[:, :1]                           # [N, 1]  — varies per sample
            emb  = x[:, 1:]                           # [N, in_channels-1]  — static per gene
            return torch.cat([expr, self.input_proj(emb)], dim=1)  # [N, input_proj_dim]
        return self.input_proj(x)                     # Identity when projection disabled

    def encode(self, data):

        return self.gnn(data, x=self._project(data.x))

    def forward(self, data):

        z_node, z_graph = self.gnn(data, x=self._project(data.x))

        edge_index = data.edge_index

        src, dst = edge_index[0], edge_index[1]
        edge_features = torch.cat([z_node[src], z_node[dst]], dim=1)
        pred_edge = self.edge_predictor(edge_features)

        pred_class = self.classifier(z_graph)

        return pred_edge, pred_class

    def plot_graph_embeddings(self, data, label_names=None, method="umap",
                              test_data=None):
        """
        Plot a UMAP of graph-level embeddings.

        Parameters
        ----------
        data : DataLoader
            Primary loader (training set, or all graphs if no split).
        label_names : list, optional
            Maps integer y labels to display strings. Must be indexed by label int.
        test_data : DataLoader, optional
            Held-out test loader. When provided its graphs are encoded alongside
            training graphs, UMAP is fit on all points jointly, and train/test
            are distinguished by marker shape so you can see whether the model
            generalises without losing the cluster structure that only the full
            dataset can reveal.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.eval()
        embeddings, labels, splits = [], [], []

        def _collect(loader, split_tag):
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(device)
                    _, z_graph = self.encode(batch)
                    embeddings.append(z_graph.cpu().numpy())
                    ys = batch.y.cpu().numpy()
                    labels.extend(
                        [label_names[y] for y in ys] if label_names else ys.tolist()
                    )
                    splits.extend([split_tag] * len(ys))

        _collect(data, "Train")
        if test_data is not None:
            _collect(test_data, "Test")

        X_emb = np.concatenate(embeddings, axis=0)
        reducer = umap.UMAP(n_components=2, random_state=42)
        umap_coords = reducer.fit_transform(X_emb)

        df_plot = pd.DataFrame(umap_coords, columns=["UMAP1", "UMAP2"])
        df_plot["Label"] = labels
        df_plot["Split"] = splits

        plt.figure(figsize=(8, 6))
        if test_data is not None:
            sns.scatterplot(
                data=df_plot, x="UMAP1", y="UMAP2",
                hue="Label", style="Split",
                markers={"Train": "o", "Test": "X"},
                alpha=0.8, s=40,
            )
            plt.title("Graph-level GNN embeddings (UMAP) — ○ train  ✕ test")
        else:
            sns.scatterplot(data=df_plot, x="UMAP1", y="UMAP2", hue="Label",
                            alpha=0.8, s=40)
            plt.title("Graph-level GNN embeddings (UMAP)")
        plt.tight_layout()
        plt.show()


def train_gnn(
    model,
    train_loader,
    device,
    epochs=100,
    lr=1e-3,
    w_l2=1e-4,
    w_classifier=1.0,
    w_edge_pred=0.5,
    val_loader=None,
):
    """
    Train the GNN model.

    Parameters
    ----------
    val_loader : DataLoader, optional
        When provided, a full evaluation pass is run at the end of every epoch
        and val_loss / val_acc are reported in the progress bar. The model is
        never trained on these batches.
    """
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=w_l2)

    total_steps = len(train_loader) * epochs

    progress_bar = tqdm(
        range(total_steps),
        desc="Training GNN model",
    )

    val_acc_display = None
    val_loss_display = None

    for epoch in range(epochs):

        epoch_correct = 0
        epoch_total = 0

        for batch in train_loader:
            batch = batch.to(device)

            batch.x = batch.x.detach().requires_grad_(False)
            batch.edge_attr = batch.edge_attr.detach().requires_grad_(False)
            batch.y = batch.y.detach()

            optimizer.zero_grad()
            pred_edge, pred_class = model(batch)
            class_loss = w_classifier * model.classifier.loss(pred_class, batch.y)
            edge_loss = w_edge_pred * model.edge_predictor.loss(
                pred_edge, batch.edge_attr
            )
            loss = class_loss + edge_loss
            loss.backward()
            optimizer.step()

            preds = pred_class.argmax(dim=-1)
            epoch_correct += (preds == batch.y).sum().item()
            epoch_total += batch.y.size(0)

            train_acc = epoch_correct / epoch_total if epoch_total > 0 else 0.0
            pf = dict(
                loss=f"{loss.item():.4f}",
                acc=f"{train_acc:.2%}",
                epoch=f"{epoch + 1}/{epochs}",
            )
            if val_acc_display is not None:
                pf["val_loss"] = f"{val_loss_display:.4f}"
                pf["val_acc"] = f"{val_acc_display:.2%}"
            progress_bar.set_postfix(**pf)
            progress_bar.update()

        # ── Validation pass (end of epoch) ────────────────────────────────────
        if val_loader is not None:
            model.eval()
            val_correct, val_total = 0, 0
            val_loss_sum, val_n_batches = 0.0, 0
            with torch.no_grad():
                for vbatch in val_loader:
                    vbatch = vbatch.to(device)
                    pred_edge_v, pred_class_v = model(vbatch)
                    c_loss_v = w_classifier * model.classifier.loss(pred_class_v, vbatch.y)
                    e_loss_v = w_edge_pred * model.edge_predictor.loss(
                        pred_edge_v, vbatch.edge_attr
                    )
                    val_loss_sum += (c_loss_v + e_loss_v).item()
                    val_n_batches += 1
                    val_correct += (pred_class_v.argmax(dim=-1) == vbatch.y).sum().item()
                    val_total += vbatch.y.size(0)
            val_acc_display = val_correct / val_total
            val_loss_display = val_loss_sum / val_n_batches
            model.train()

    progress_bar.close()


class ShallowSAE(nn.Module):
    def __init__(self, in_dim, latent_dim, sparsity_weight=1e-3):
        """
        Define a shallow sparse autoencoder (SAE) model as implemented in Anthropic's paper "Decomposing Language Models with Dictionary Learning"

        Parameters
        ----------

        """
        super().__init__()

        # MSE loss + L1 sparsity penalty
        self.criterion = nn.MSELoss()
        self.sparsity_weight = sparsity_weight

        # Encoder parameters
        self.W_enc = nn.Parameter(torch.randn(in_dim, latent_dim) / in_dim**0.5)
        self.b_enc = nn.Parameter(torch.zeros(latent_dim))

        # Decoder parameters
        self.W_dec = nn.Parameter(torch.randn(latent_dim, in_dim) / latent_dim**0.5)
        self.b_dec = nn.Parameter(torch.zeros(in_dim))

    def forward(self, x):
        # Shift input by decoder bias to center around zero
        x_enc = x - self.b_dec
        # Linear encoder transformation followed by ReLU activation to enforce non-negativity
        z = F.relu(x_enc @ self.W_enc + self.b_enc)
        # Linear decoder transformation to reconstruct input
        x_recon = (z @ self.W_dec) + self.b_dec

        return z, x_recon

    def loss(self, pred_x, true_x, z):
        loss_recon = self.criterion(pred_x, true_x)
        loss_sparsity = self.sparsity_weight * torch.mean(
            torch.sum(torch.abs(z), dim=1)
        )

        return loss_recon + loss_sparsity

    def normalize_weights(self):
        with torch.no_grad():
            norms = torch.norm(self.W_dec, dim=1, keepdim=True)
            self.W_dec.data = self.W_dec.data / (norms + 1e-12)


def train_sae_graph(
    sae_model,
    gnn_model,
    train_loader,
    device,
    epochs=1000,
    lr=1e-3,
    w_l2=1e-4,
):

    sae_model.train()
    gnn_model.eval()

    optimizer = optim.Adam(sae_model.parameters(), lr=lr, weight_decay=w_l2)

    total_steps = len(train_loader) * epochs

    progress_bar = tqdm(
        range(total_steps),
        desc="Training SAE model",
    )

    for epoch in range(epochs):

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Extract graph-level embeddings
            with torch.no_grad():
                _, z_graph = gnn_model.encode(batch)

            # Forward pass through SAE
            z, pred_z_graph = sae_model(z_graph)

            # Calculate loss
            loss = sae_model.loss(pred_z_graph, z_graph, z)

            # Backpropagation and optimization step
            loss.backward()
            optimizer.step()
            sae_model.normalize_weights()  # normalize decoder weights to prevent collapse to zero and encourage diversity in learned features

            # Update progress bar
            progress_bar.set_postfix(
                total_loss=f"{loss.item():.4f}",
                epoch=f"{epoch}/{epochs + 1}",
            )
            progress_bar.update()

    progress_bar.close()


def train_sae_node(
    sae_model,
    gnn_model,
    train_loader,
    device,
    epochs=1000,
    lr=1e-3,
    w_l2=1e-4,
):

    sae_model.train()
    gnn_model.eval()

    optimizer = optim.Adam(sae_model.parameters(), lr=lr, weight_decay=w_l2)

    total_steps = len(train_loader) * epochs

    progress_bar = tqdm(
        range(total_steps),
        desc="Training SAE model",
    )

    for epoch in range(epochs):

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Extract node-level embeddings
            with torch.no_grad():
                z_node, _ = gnn_model.encode(
                    batch
                )  # z_node shape: [total_nodes_in_batch, latent_dim]

            # Forward pass through SAE
            z, pred_z_node = sae_model(z_node)

            # Calculate loss
            loss = sae_model.loss(pred_z_node, z_node, z)

            # Backpropagation and optimization step
            loss.backward()
            optimizer.step()
            sae_model.normalize_weights()  # normalize decoder weights to prevent collapse to zero and encourage diversity in learned features

            # Update progress bar
            progress_bar.set_postfix(
                total_loss=f"{loss.item():.4f}",
                epoch=f"{epoch}/{epochs + 1}",
            )
            progress_bar.update()

    progress_bar.close()


def train_sae_edge(
    sae_model,
    gnn_model,
    train_loader,
    device,
    epochs=1000,
    lr=1e-3,
    w_l2=1e-4,
):

    sae_model.train()
    gnn_model.eval()

    optimizer = optim.Adam(sae_model.parameters(), lr=lr, weight_decay=w_l2)

    total_steps = len(train_loader) * epochs

    progress_bar = tqdm(
        range(total_steps),
        desc="Training SAE model",
    )

    for epoch in range(epochs):

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Extract edge-level embeddings
            with torch.no_grad():
                z_node, _ = gnn_model.encode(
                    batch
                )  # z_node shape: [total_nodes_in_batch, latent_dim]

                # Construct edge embeddings
                edge_index = batch.edge_index
                src, dst = edge_index[0], edge_index[1]

                # z_edge = mean(z_node[src], z_node[dst])
                z_edge = (
                    z_node[src] + z_node[dst]
                ) / 2.0  # z_edge shape: [total_edges_in_batch, latent_dim]

            # Forward pass through SAE
            z, pred_z_edge = sae_model(z_edge)

            # Calculate loss
            loss = sae_model.loss(pred_z_edge, z_edge, z)

            # Backpropagation and optimization step
            loss.backward()
            optimizer.step()
            sae_model.normalize_weights()  # normalize decoder weights to prevent collapse to zero and encourage diversity in learned features

            # Update progress bar
            progress_bar.set_postfix(
                total_loss=f"{loss.item():.4f}",
                epoch=f"{epoch}/{epochs + 1}",
            )
            progress_bar.update()

    progress_bar.close()


# class GEA(nn.Module):
