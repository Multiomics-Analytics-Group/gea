import random
from pathlib import Path

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


def set_seed(seed: int = 42) -> None:
    """
    Set all random seeds for reproducibility across Python, NumPy, and PyTorch.

    Call this once before constructing any model or data loader.
    cuDNN is set to deterministic mode so convolutional ops are reproducible
    at a small performance cost.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
        n_classes_ct=None,
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

        n_classes_ct: int or None
            If set, adds a second graph-level classification head (e.g. for cell
            type) on top of the shared graph embedding, enabling a multi-task
            objective. When None (default) only the primary phenotype head exists
            and the model behaves exactly as before.

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

        # Optional second graph-level classifier (e.g. cell type). Backward compatible:
        # when n_classes_ct is None no head is built and forward returns None for it.
        # Reuses the same hidden architecture as the primary phenotype head.
        self.n_classes_ct = n_classes_ct
        if n_classes_ct is not None:
            ct_hidden = classifier_net[1:-1]  # hidden layer sizes of the primary head
            ct_net = [latent_dim] + list(ct_hidden) + [n_classes_ct]
            modules = []
            for i in range(len(ct_net) - 1):
                modules.append(nn.Linear(ct_net[i], ct_net[i + 1]))
                if i < len(ct_net) - 2:
                    modules.append(classifier_actfn)
            self.classifier_ct = GraphClassifier(nn.Sequential(*modules))
        else:
            self.classifier_ct = None

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
        pred_class_ct = (
            self.classifier_ct(z_graph) if self.classifier_ct is not None else None
        )

        return pred_edge, pred_class, pred_class_ct

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
    w_classifier_ct=1.0,
    val_loader=None,
):
    """
    Train the GNN model.

    Parameters
    ----------
    w_classifier_ct : float
        Loss weight for the optional second classification head (e.g. cell type).
        Only used when the model was built with ``n_classes_ct`` and the batches
        carry a ``y_ct`` attribute; otherwise ignored.
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
    val_acc_ct_display = None

    for epoch in range(epochs):

        epoch_correct = 0
        epoch_correct_ct = 0
        epoch_total = 0

        for batch in train_loader:
            batch = batch.to(device)

            batch.x = batch.x.detach().requires_grad_(False)
            batch.edge_attr = batch.edge_attr.detach().requires_grad_(False)
            batch.y = batch.y.detach()

            optimizer.zero_grad()
            pred_edge, pred_class, pred_class_ct = model(batch)
            class_loss = w_classifier * model.classifier.loss(pred_class, batch.y)
            edge_loss = w_edge_pred * model.edge_predictor.loss(
                pred_edge, batch.edge_attr
            )
            loss = class_loss + edge_loss

            # Optional second objective (e.g. cell type)
            has_ct = pred_class_ct is not None and hasattr(batch, "y_ct")
            if has_ct:
                ct_loss = w_classifier_ct * model.classifier_ct.loss(
                    pred_class_ct, batch.y_ct
                )
                loss = loss + ct_loss

            loss.backward()
            optimizer.step()

            preds = pred_class.argmax(dim=-1)
            epoch_correct += (preds == batch.y).sum().item()
            epoch_total += batch.y.size(0)
            if has_ct:
                epoch_correct_ct += (
                    pred_class_ct.argmax(dim=-1) == batch.y_ct
                ).sum().item()

            train_acc = epoch_correct / epoch_total if epoch_total > 0 else 0.0
            pf = dict(
                loss=f"{loss.item():.4f}",
                acc=f"{train_acc:.2%}",
                epoch=f"{epoch + 1}/{epochs}",
            )
            if has_ct:
                pf["acc_ct"] = f"{epoch_correct_ct / epoch_total:.2%}"
            if val_acc_display is not None:
                pf["val_loss"] = f"{val_loss_display:.4f}"
                pf["val_acc"] = f"{val_acc_display:.2%}"
                if val_acc_ct_display is not None:
                    pf["val_acc_ct"] = f"{val_acc_ct_display:.2%}"
            progress_bar.set_postfix(**pf)
            progress_bar.update()

        # ── Validation pass (end of epoch) ────────────────────────────────────
        if val_loader is not None:
            model.eval()
            val_correct, val_correct_ct, val_total = 0, 0, 0
            val_loss_sum, val_n_batches = 0.0, 0
            with torch.no_grad():
                for vbatch in val_loader:
                    vbatch = vbatch.to(device)
                    pred_edge_v, pred_class_v, pred_class_ct_v = model(vbatch)
                    c_loss_v = w_classifier * model.classifier.loss(pred_class_v, vbatch.y)
                    e_loss_v = w_edge_pred * model.edge_predictor.loss(
                        pred_edge_v, vbatch.edge_attr
                    )
                    batch_loss_v = c_loss_v + e_loss_v
                    has_ct_v = pred_class_ct_v is not None and hasattr(vbatch, "y_ct")
                    if has_ct_v:
                        batch_loss_v = batch_loss_v + w_classifier_ct * model.classifier_ct.loss(
                            pred_class_ct_v, vbatch.y_ct
                        )
                        val_correct_ct += (
                            pred_class_ct_v.argmax(dim=-1) == vbatch.y_ct
                        ).sum().item()
                    val_loss_sum += batch_loss_v.item()
                    val_n_batches += 1
                    val_correct += (pred_class_v.argmax(dim=-1) == vbatch.y).sum().item()
                    val_total += vbatch.y.size(0)
            val_acc_display = val_correct / val_total
            val_loss_display = val_loss_sum / val_n_batches
            val_acc_ct_display = (
                val_correct_ct / val_total
                if (val_total > 0 and model.classifier_ct is not None)
                else None
            )
            model.train()

    progress_bar.close()


# ── Embedding extraction (trained GNN → EmbeddingDataset) ─────────────────────

# Separators used to build the entity string of a sample. Both are unlikely to
# occur inside a sample name or an Ensembl gene id, so entities stay parseable.
ENTITY_SEP = "::"      # between graph id and the gene part
GENE_PAIR_SEP = "--"   # between the two genes of an edge


@torch.no_grad()
def extract_embeddings(
    model,
    loader,
    device,
    level="graph",
    gene_names=None,
    label_names=None,
    ct_names=None,
    undirected=True,
    dtype=np.float32,
):
    """
    Run a trained GNN over a loader and collect the embeddings of one level,
    together with the identity of every row, in the layout read by
    :class:`gea.dataloader.EmbeddingDataset`.

    Parameters
    ----------
    model : GNNModel
        Trained model. Only ``encode`` and the prediction heads are used; the
        model is set to eval mode and no gradients are tracked.
    loader : torch_geometric.loader.DataLoader
        Must use ``shuffle=False`` so row order is reproducible: SAE activations
        are later joined back onto these rows by position.
    device : torch.device
    level : {"graph", "node", "edge"}
        Which representation to export:

        - ``"graph"`` → one row per graph, ``z_graph`` (mean-pooled nodes).
        - ``"node"``  → one row per (graph, gene), ``z_node``.
        - ``"edge"``  → one row per (graph, gene pair), ``mean(z_src, z_dst)``,
          the same edge embedding used by ``extract_edge_activations``.
    gene_names : list of str, optional
        Gene identifiers in node order, i.e. the order used when the graphs were
        built by ``gene_networks_to_pyg`` (the columns of the normalised count
        matrix). Required for ``level="node"`` and ``level="edge"``: the graphs
        themselves only carry integer node indices.
    label_names : list of str, optional
        Maps integer ``y`` (phenotype) to display strings.
    ct_names : list of str, optional
        Maps integer ``y_ct`` (optional second label, e.g. cell type) to display
        strings.
    undirected : bool
        Edge level only. ``gene_networks_to_pyg`` stores both directions of every
        edge and the edge embedding is symmetric, so by default only the
        ``src < dst`` copy is kept — otherwise every gene pair appears twice with
        an identical embedding, which biases SAE training and doubles the file.
    dtype : numpy dtype
        Storage dtype of the embedding matrix. ``np.float16`` halves the size of
        large edge-level exports at some precision cost.

    Returns
    -------
    dict
        Ready to be written with ``np.savez(path, **result)`` (see
        ``save_embeddings``) and read back by ``EmbeddingDataset``:

        - ``embeddings`` : [N, D] embedding matrix.
        - ``entities`` : identity of each row as a string —
          ``"<graph_id>"`` (graph level),
          ``"<graph_id>::<gene>"`` (node level),
          ``"<graph_id>::<geneA>--<geneB>"`` (edge level).
        - ``annotations`` : one dict of numeric metadata per row.

          The biological groups are **one-hot vectors**, not class integers:
          ``disease`` is ``[0, 1]`` rather than ``1`` and ``cell_type`` is
          ``[0, 0, 0, 0, 0, 1, 0]`` rather than ``5``, so they can be regressed
          against SAE features directly. ``disease_classes`` /
          ``cell_type_classes`` (below) give the order of their positions;
          ``cell_type`` is omitted when the graphs carry no ``y_ct``.

          The remaining entries are the values that cannot be recovered from the
          row's identity: ``expression``, ``degree``, ``node_idx`` and
          ``graph_idx`` (node level); the **signed** co-expression ``weight``,
          ``src_idx``, ``dst_idx`` and ``graph_idx`` (edge level). Graph level has
          only the one-hot groups. The index entries are join keys, not identity:
          ``graph_idx`` is the row of the same graph in the graph-level export,
          and ``node_idx`` / ``src_idx`` / ``dst_idx`` are local node indices in
          the PyG graph, as used by ``plot_feature_subgraph``.
        - ``prediction`` : predicted class of the graph (graph and node level,
          a node inheriting the prediction of its graph) or the predicted edge
          weight (edge level).
        - ``target`` : ground-truth phenotype ``y`` (graph and node level) or the
          value the edge predictor was trained on, ``|weight|`` (edge level).
        - ``graph_id``, ``gene`` / ``gene_a`` + ``gene_b``, ``disease``,
          ``cell_type``, ``disease_classes``, ``cell_type_classes``, ``level`` :
          the same identity as columns rather than packed into the entity string,
          plus the one-hot class order. ``EmbeddingDataset`` ignores these; they
          exist so ``load_embedding_metadata`` can build the table used to map
          SAE features back onto genes without string parsing.

    Notes
    -----
    Edge weights are stored signed because the sign is the biologically
    meaningful part (co-expression vs anti-correlation), while the model itself
    only sees ``|weight|`` in ``edge_attr`` and the sign in ``edge_type``. The
    sign therefore replaces a separate ``edge_type`` annotation, which would
    carry no extra information.
    """
    if level not in ("graph", "node", "edge"):
        raise ValueError(
            f"level must be one of 'graph', 'node', 'edge'; got {level!r}."
        )
    if level in ("node", "edge") and gene_names is None:
        raise ValueError(
            f"gene_names is required for level={level!r}: the PyG graphs only "
            "store integer node indices, so gene identity has to come from the "
            "gene order used to build them."
        )
    if gene_names is not None:
        gene_names = list(gene_names)

    model.to(device)
    model.eval()

    graphs = getattr(loader, "dataset", [])
    has_ct = len(graphs) > 0 and hasattr(graphs[0], "y_ct")

    def _classes(names, head, attr):
        """Class order of a graph-level label, as a list of display strings."""
        if names is not None:
            return [str(n) for n in names]
        if head is not None:
            n = head.classifier_net[-1].out_features
        else:
            n = max(int(getattr(g, attr)) for g in graphs) + 1
        return [str(i) for i in range(n)]

    disease_classes = _classes(label_names, model.classifier, "y")
    ct_classes = (
        _classes(ct_names, model.classifier_ct, "y_ct") if has_ct else []
    )
    # One shared vector per class: rows of the same class reference the same array,
    # so a million one-hot annotations cost no extra memory and pickle to a
    # back-reference rather than a copy.
    disease_onehot = list(np.eye(len(disease_classes), dtype=np.float32))
    ct_onehot = list(np.eye(len(ct_classes), dtype=np.float32)) if ct_classes else []

    def _groups(yy, yc):
        """One-hot annotation entries for the biological groups of one row."""
        groups = {"disease": disease_onehot[int(yy)]}
        if ct_onehot:
            groups["cell_type"] = ct_onehot[int(yc)]
        return groups

    emb_chunks, pred_chunks, target_chunks = [], [], []
    annotations, entities = [], []
    columns = {}      # identity columns: name → list of str, one entry per row
    graph_offset = 0  # index of the first graph of the current batch

    def _add_col(name, values):
        columns.setdefault(name, []).extend(values)

    def _names(arr, mapping):
        """Display strings for an integer label array, one per row."""
        return [
            "" if np.isnan(v) else mapping[int(v)]
            for v in np.asarray(arr, dtype=np.float64)
        ]

    for batch in tqdm(loader, desc=f"Extracting {level}-level embeddings"):
        batch = batch.to(device)
        z_node, z_graph = model.encode(batch)
        n_graphs = batch.num_graphs

        # Per-graph identity and labels, indexed by position within the batch
        graph_ids = (
            list(batch.sample_name)
            if hasattr(batch, "sample_name")
            else [f"graph_{graph_offset + i}" for i in range(n_graphs)]
        )
        y = batch.y.cpu().numpy().reshape(-1)
        y_ct = (
            batch.y_ct.cpu().numpy().reshape(-1).astype(np.float64)
            if hasattr(batch, "y_ct")
            else np.full(n_graphs, np.nan)
        )
        graph_idx = np.arange(graph_offset, graph_offset + n_graphs)
        pred_class = model.classifier(z_graph).argmax(dim=-1).cpu().numpy()

        if level == "graph":
            emb_chunks.append(z_graph.cpu().numpy())

            # Row order already is graph order, so a graph_idx annotation would
            # only repeat the row number; identity lives in the entity instead.
            annotations.extend(_groups(yy, yc) for yy, yc in zip(y, y_ct))
            entities.extend(graph_ids)

            _add_col("graph_id", graph_ids)
            _add_col("disease", _names(y, disease_classes))
            _add_col("cell_type", _names(y_ct, ct_classes))

            pred_chunks.append(pred_class)
            target_chunks.append(y)

        elif level == "node":
            n_per_graph = np.diff(batch.ptr.cpu().numpy())
            if not np.all(n_per_graph == len(gene_names)):
                raise ValueError(
                    f"gene_names has {len(gene_names)} entries but graphs in this "
                    f"batch have {sorted(set(n_per_graph.tolist()))} nodes. It must "
                    "list every node of every graph, in node order."
                )

            emb_chunks.append(z_node.cpu().numpy())

            expr = batch.x[:, 0].cpu().numpy()  # expression is feature dim 0
            degree = (
                torch.bincount(batch.edge_index[0], minlength=batch.num_nodes)
                .cpu()
                .numpy()
            )
            node_graph = batch.batch.cpu().numpy()      # graph of each node
            node_idx = np.tile(np.arange(len(gene_names)), n_graphs)

            annotations.extend(
                {
                    "expression": float(e),
                    "node_idx": float(ni),
                    "degree": float(d),
                    "graph_idx": float(graph_idx[g]),
                    **_groups(y[g], y_ct[g]),
                }
                for e, ni, d, g in zip(expr, node_idx, degree, node_graph)
            )
            entities.extend(
                f"{graph_ids[g]}{ENTITY_SEP}{gene_names[ni]}"
                for g, ni in zip(node_graph, node_idx)
            )

            _add_col("graph_id", [graph_ids[g] for g in node_graph])
            _add_col("gene", gene_names * n_graphs)
            _add_col("disease", _names(y[node_graph], disease_classes))
            _add_col("cell_type", _names(y_ct[node_graph], ct_classes))

            pred_chunks.append(pred_class[node_graph])
            target_chunks.append(y[node_graph])

        else:  # level == "edge"
            ei = batch.edge_index
            edge_graph = batch.batch[ei[0]]
            offset = batch.ptr[edge_graph]          # first node index of each edge's graph
            src_local = (ei[0] - offset).cpu().numpy()
            dst_local = (ei[1] - offset).cpu().numpy()

            # Both directions are present, so src < dst keeps exactly one copy
            keep = src_local < dst_local if undirected else np.ones(len(src_local), bool)
            keep_t = torch.as_tensor(keep, device=ei.device)
            src, dst = ei[0][keep_t], ei[1][keep_t]
            src_local, dst_local = src_local[keep], dst_local[keep]

            emb_chunks.append(((z_node[src] + z_node[dst]) / 2.0).cpu().numpy())
            pred_chunks.append(
                model.edge_predictor(torch.cat([z_node[src], z_node[dst]], dim=1))
                .reshape(-1)
                .cpu()
                .numpy()
            )

            abs_w = batch.edge_attr[keep_t].cpu().numpy()
            e_type = batch.edge_type[keep_t].cpu().numpy()
            # edge_type 1 marks an anti-correlated pair, whose weight was stored
            # as |weight| for the model; restore the sign for the analysis side.
            signed_w = abs_w * np.where(e_type == 0, 1.0, -1.0)
            eg = edge_graph[keep_t].cpu().numpy()

            # No edge_type annotation: it is exactly sign(weight).
            annotations.extend(
                {
                    "weight": float(w),
                    "src_idx": float(s),
                    "dst_idx": float(d),
                    "graph_idx": float(graph_idx[g]),
                    **_groups(y[g], y_ct[g]),
                }
                for w, s, d, g in zip(signed_w, src_local, dst_local, eg)
            )
            entities.extend(
                f"{graph_ids[g]}{ENTITY_SEP}{gene_names[s]}{GENE_PAIR_SEP}{gene_names[d]}"
                for g, s, d in zip(eg, src_local, dst_local)
            )

            _add_col("graph_id", [graph_ids[g] for g in eg])
            _add_col("gene_a", [gene_names[s] for s in src_local])
            _add_col("gene_b", [gene_names[d] for d in dst_local])
            _add_col("disease", _names(y[eg], disease_classes))
            _add_col("cell_type", _names(y_ct[eg], ct_classes))

            target_chunks.append(abs_w)

        graph_offset += n_graphs

    result = {
        "embeddings": np.concatenate(emb_chunks, axis=0).astype(dtype),
        "annotations": np.array(annotations, dtype=object),
        "entities": np.array(entities, dtype=object),
        "prediction": np.concatenate(pred_chunks, axis=0),
        "target": np.concatenate(target_chunks, axis=0),
        "level": np.array(level),
        # Position → class name for the one-hot annotations
        "disease_classes": np.array(disease_classes, dtype=object),
        "cell_type_classes": np.array(ct_classes, dtype=object),
    }
    result.update(
        {name: np.array(values, dtype=object) for name, values in columns.items()}
    )

    return result


def save_embeddings(path, embeddings, compress=False):
    """
    Write the dict returned by ``extract_embeddings`` to an ``.npz`` file that
    :class:`gea.dataloader.EmbeddingDataset` can read.

    Parameters
    ----------
    path : str or pathlib.Path
        Output path; NumPy appends ``.npz`` when missing.
    embeddings : dict
        Output of ``extract_embeddings``.
    compress : bool
        Use ``np.savez_compressed``. Smaller files, but slow on large
        edge-level exports.

    Returns
    -------
    str
        The path actually written.
    """
    path = str(path)
    saver = np.savez_compressed if compress else np.savez
    saver(path, **embeddings)
    return path if path.endswith(".npz") else path + ".npz"


def export_embeddings(
    model,
    loader,
    device,
    out_dir,
    prefix="embeddings",
    levels=("graph", "node", "edge"),
    compress=False,
    **kwargs,
):
    """
    Extract and save several embedding levels in one call, as
    ``<out_dir>/<prefix>_<level>.npz``.

    Parameters
    ----------
    model, loader, device
        See ``extract_embeddings``.
    out_dir : str or pathlib.Path
        Created if it does not exist.
    prefix : str
        Filename prefix.
    levels : iterable of str
        Subset of ``("graph", "node", "edge")``.
    compress : bool
        Passed to ``save_embeddings``.
    **kwargs
        Forwarded to ``extract_embeddings`` (``gene_names``, ``label_names``,
        ``ct_names``, ``undirected``, ``dtype``).

    Returns
    -------
    dict
        Maps level → written file path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for level in levels:
        emb = extract_embeddings(model, loader, device, level=level, **kwargs)
        paths[level] = save_embeddings(
            out_dir / f"{prefix}_{level}.npz", emb, compress=compress
        )
        n, d = emb["embeddings"].shape
        print(f"{level:>5}-level: {n:>9,} rows × {d} dims → {paths[level]}")

    return paths


class ShallowSAE(nn.Module):
    def __init__(self, in_dim, latent_dim, sparsity_weight=1e-3):
        """
        Define a shallow sparse autoencoder (SAE) model as implemented in Anthropic's paper "Decomposing Language Models with Dictionary Learning"

        Parameters
        ----------

        """
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim

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


def train_sae(
    sae_model,
    train_loader,
    device,
    epochs=1000,
    lr=1e-3,
    w_l2=1e-4,
):

    sae_model.to(device)
    sae_model.train()

    optimizer = optim.Adam(sae_model.parameters(), lr=lr, weight_decay=w_l2)

    total_steps = len(train_loader) * epochs

    progress_bar = tqdm(
        range(total_steps),
        desc="Training SAE model",
    )

    for epoch in range(epochs):

        for batch in train_loader:
            embeddings = batch['embedding'].to(device)
            optimizer.zero_grad()

            z, pred_z_graph = sae_model(embeddings)

            # Calculate loss
            loss = sae_model.loss(pred_z_graph, embeddings, z)

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
