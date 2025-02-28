from torch import nn
import torch.nn.functional as F
import torch
from skimage.segmentation import slic
from maemodel import model_config
from sklearn.cluster import KMeans


class MAEExtractor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.vol_size = cfg["img_size"]
        self.embed_dim = cfg["embed_dim"]
        self.patch_size = cfg["patch_size"]
        self.model = model_config(cfg)
        if cfg["pretrained_path"] is not None:
            self.model.load_state_dict(torch.load(cfg["pretrained_path"], map_location=f'cuda:{cfg["device"]}'))

    def get_raw_feature(self, x):
        b, c, h, w = x.shape
        x = self.model.infer_latent(x)
        x = x[:, 1:, :].view(b, h // self.patch_size, w // self.patch_size, -1).permute(0, 3, 1, 2)
        return x

    def forward(self, x):
        b, c, orgH, orgW = x.shape
        # Calculating necessary padding for both dimensions
        pad_width = (self.vol_size - (orgW % self.vol_size)) % self.vol_size
        pad_height = (self.vol_size - (orgH % self.vol_size)) % self.vol_size

        # Adding padding symmetrically
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        pad_up = pad_height // 2
        pad_down = pad_height - pad_up

        x = F.pad(x, [pad_left, pad_right, pad_up, pad_down], "constant", 0)
        _, _, newH, newW = x.shape
        out = torch.zeros((b, self.embed_dim, newH // self.patch_size, newW // self.patch_size))
        mask = torch.zeros((b, self.embed_dim, newH // self.patch_size, newW // self.patch_size))
        step = self.vol_size // 2

        for i_h in range(0, newH - step, step):
            for i_w in range(0, newW - step, step):
                sample = x[:, :, i_h: i_h + self.vol_size, i_w: i_w + self.vol_size]
                rep = self.get_raw_feature(sample)
                # Compute the weights for blending
                h_end = min(i_h + self.vol_size, newH)
                w_end = min(i_w + self.vol_size, newW)

                rep = rep.detach().cpu()
                weights = torch.ones_like(rep)
                # Apply the weights to the overlapping region
                out[:, :, i_h // self.patch_size: h_end // self.patch_size,
                i_w // self.patch_size: w_end // self.patch_size] += rep * weights
                mask[:, :, i_h // self.patch_size: h_end // self.patch_size,
                i_w // self.patch_size: w_end // self.patch_size] += weights
        # Divide by the mask to average the overlapping regions
        out = torch.div(out, mask)
        out = F.interpolate(out, size=(newH, newW), mode='bilinear', align_corners=True)
        return out[:, :, pad_up:pad_up + orgH, pad_left:pad_left + orgW]


class EigenDecompose(nn.Module):
    def __init__(self, in_chs=192, out_chs=24, mid_chs=None):
        super(EigenDecompose, self).__init__()
        if mid_chs is None:
            mid_chs = [96, 48]
        self.weight_gcn_1 = nn.init.trunc_normal_(
            nn.Parameter(torch.FloatTensor(in_chs, mid_chs[0]), requires_grad=True))
        self.bias_gcn_1 = nn.init.trunc_normal_(nn.Parameter(torch.FloatTensor(mid_chs[0]), requires_grad=True))

        self.weight_gcn_2 = nn.init.trunc_normal_(
            nn.Parameter(torch.FloatTensor(mid_chs[0], mid_chs[1]), requires_grad=True))
        self.bias_gcn_2 = nn.init.trunc_normal_(nn.Parameter(torch.FloatTensor(mid_chs[1]), requires_grad=True))

        self.weight_gcn_3 = nn.init.trunc_normal_(
            nn.Parameter(torch.FloatTensor(mid_chs[1], out_chs), requires_grad=True))
        self.bias_gcn_3 = nn.init.trunc_normal_(nn.Parameter(torch.FloatTensor(out_chs), requires_grad=True))

    def forward(self, gcn_adj, in_signal):
        # Branch 1
        g_conv_1 = torch.matmul(torch.matmul(gcn_adj, in_signal), self.weight_gcn_1) + self.bias_gcn_1
        g_conv_2 = torch.matmul(torch.matmul(gcn_adj, g_conv_1), self.weight_gcn_2) + self.bias_gcn_2
        g_conv_3 = torch.matmul(torch.matmul(gcn_adj, g_conv_2), self.weight_gcn_3) + self.bias_gcn_3

        # Normalize
        re_base = torch.nn.functional.normalize(g_conv_3, p=2, dim=0)
        return re_base


class Graph_Learning(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.compactness = cfg["compactness"]
        self.n_clusters = cfg["n_clusters"]
        self.embed_dim = cfg["embed_dim"]
        self.n_segments = cfg["n_segments"]
        self.alpha = cfg["alpha"]
        self.cfg = cfg

        self.MAE_Extractor = MAEExtractor(cfg)

        self.GNN_Eigens = EigenDecompose(self.embed_dim * 2, cfg["n_eigens"])
        self.Assignment_Spec = nn.Sequential(nn.Linear(cfg["n_eigens"], cfg["mlp_hidden_dim_spec"]), nn.ELU(),
                                             nn.Dropout(0.25),
                                             nn.Linear(cfg["mlp_hidden_dim_spec"], self.n_clusters), nn.ELU())

        self.centoids = nn.Parameter(torch.Tensor(self.n_clusters, self.embed_dim * 2))
        self.initialized = False
        self.v = cfg["student_v"]

    def initialize_centers(self, x, n_segments, chunk_size=1200):
        device = x.device
        # SLIC superpixel decomposition
        x_numpy = x.permute(1, 2, 0).detach().cpu().numpy()
        segments = slic(x_numpy, n_segments=n_segments, compactness=self.compactness, channel_axis=2)
        segments = torch.tensor(segments).to(device)
        with torch.no_grad():
            feas = self.MAE_Extractor(x.unsqueeze(0)).squeeze(0)
            feas = feas.permute(1, 2, 0).to(device)
            graph_lap, graph_adj, spfea = self.build_graph_deep(segments, feas, alpha=self.alpha, chunk_size=chunk_size)
        spfea = spfea.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0)
        kmeans.n_init = 'auto'
        kmeans._n_threads = 1
        kmeans.fit(spfea)
        with torch.no_grad():
            self.centoids.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)

        self.initialized = True
        print("Successful initialize K-Means centers from selected image slice!")
        return kmeans.cluster_centers_

    def build_graph_deep(self, segments, feas_for_graph, alpha=4.0, chunk_size=1200):
        device = feas_for_graph.device
        # Calculate the total number of segments (dimension of graph)
        mean_feas_channels = feas_for_graph.mean(dim=2, keepdim=True)
        std_feas_channels = feas_for_graph.std(dim=2, keepdim=True)
        feas_for_graph = (feas_for_graph - mean_feas_channels) / (std_feas_channels + 1e-16)

        # Compute the total number of segments
        num_segments = segments.max().item() + 1

        # Compute mean and std feature for each segment
        features_mean = torch.stack(
            [feas_for_graph[(segments == i).to(device)].mean(dim=0) for i in range(1, num_segments)])
        features_std = torch.stack(
            [feas_for_graph[(segments == i).to(device)].std(dim=0) for i in range(1, num_segments)])
        features = torch.cat((features_mean, features_std), dim=1)

        # RBF Kernel with chunks
        weights = torch.zeros((num_segments - 1, num_segments - 1), device=device)

        for i in range(0, num_segments - 1, chunk_size):
            end_i = min(i + chunk_size, num_segments - 1)
            for j in range(0, num_segments - 1, chunk_size):
                end_j = min(j + chunk_size, num_segments - 1)
                diff = features[i:end_i].unsqueeze(1) - features[j:end_j]
                sq_dist = (diff ** 2).sum(dim=2)
                weights[i:end_i, j:end_j] = torch.exp(-0.5 * torch.sqrt(sq_dist))

        weights = weights - (weights.max() / alpha)
        weights[weights < 0] = 0

        # Compute the degree matrix
        degree_matrix = torch.diag(weights.sum(dim=1)).to(device)

        # Compute the normalized adjacency matrix
        degree_inv_sqrt = torch.diag(torch.pow(degree_matrix.diag(), -0.5)).to(device)
        normalized_adj_matrix = degree_inv_sqrt @ weights @ degree_inv_sqrt

        # Compute the normalized graph Laplacian
        identity_matrix = torch.eye(num_segments - 1).to(device)
        normalized_laplacian = identity_matrix - normalized_adj_matrix

        return normalized_laplacian, normalized_adj_matrix, features

    def cal_loss_emb(self, eigens, graph_lap):
        n, k = eigens.shape
        device = eigens.device
        eye_matrix = torch.eye(k).to(device)

        loss_ort = F.mse_loss(eye_matrix, torch.matmul(eigens.T, eigens))
        loss_diag = torch.mean(torch.abs(torch.matmul(eigens.T, torch.matmul(graph_lap, eigens)) * (1.0 - eye_matrix)))
        loss = loss_ort + loss_diag
        return loss

    def target_distribution(self, p):
        weight = p ** 2 / p.sum(0)
        return (weight.t() / weight.sum(1)).t()

    def source_distribution(self, spec_clu):
        q = 1.0 / (1.0 + torch.sum(torch.pow(spec_clu.unsqueeze(1) - self.centoids, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q

    def cal_loss_reg(self, r, spec):
        loss_reg = F.kl_div(spec.log(), r, reduction='batchmean')
        return loss_reg

    def cal_loss_clu(self, spa):
        q = self.source_distribution(spa)
        r = self.target_distribution(q)
        loss_clu = F.kl_div(q.log(), r, reduction='batchmean')
        return loss_clu, r, q

    def extract_feas(self, x):
        with torch.no_grad():
            feas = self.MAE_Extractor(x.unsqueeze(0)).squeeze(0)
        return feas

    def extract_eigens(self, x, alpha_eigens=8.0):
        device = x.device

        # SLIC superpixel decomposition
        x_numpy = x.permute(1, 2, 0).detach().cpu().numpy()
        segments = slic(x_numpy, n_segments=self.n_segments, compactness=self.compactness, channel_axis=2)
        segments = torch.tensor(segments).to(device)

        with torch.no_grad():
            feas = self.MAE_Extractor(x.unsqueeze(0)).squeeze(0)
            feas = feas.permute(1, 2, 0).to(device)
            graph_lap, graph_adj, spfea = self.build_graph_deep(segments, feas, alpha=alpha_eigens)

        # Learning Eigens
        eigens = self.GNN_Eigens(graph_adj, spfea)

        loss_eigens = self.cal_loss_emb(eigens, graph_lap)
        loss = self.cfg["w_emb"] * loss_eigens

        return segments, loss, eigens

    def test(self, x, feas=None, n_segments=3000, chunk_size=1200):
        c, h, w = x.shape
        device = x.device

        with torch.no_grad():
            if feas is None:
                feas = self.MAE_Extractor(x.unsqueeze(0)).squeeze(0)
            feas = feas.permute(1, 2, 0).to(device)
            # SLIC superpixel decomposition
            x_numpy = x.permute(1, 2, 0).detach().cpu().numpy()
            segments = slic(x_numpy, n_segments=n_segments, compactness=self.compactness, channel_axis=2)
            segments = torch.tensor(segments).to(device)
            graph_lap, graph_adj, spfea = self.build_graph_deep(segments, feas, alpha=self.alpha, chunk_size=chunk_size)
            segments = segments.detach().cpu().numpy()
            segments_flat = segments.flatten()
        # Learning Eigens
        eigens = self.GNN_Eigens(graph_adj, spfea)

        # Assigning Clusters in Spectral Branch
        clusters_spec_prob = F.softmax(self.Assignment_Spec(eigens), dim=1)
        clusters_spec = torch.argmax(clusters_spec_prob, dim=-1)
        clusters_spec = clusters_spec.detach().cpu().numpy()
        # Assign clusters to segments using vectorized operation
        multi_seg_image_spec = clusters_spec[segments_flat - 1].reshape(h, w, 1)

        # Assigning Clusters in Spatial Branch
        q = self.source_distribution(spfea)
        r = self.target_distribution(q)
        r = torch.argmax(r, dim=-1)
        clusters_spa = r.detach().cpu().numpy()
        multi_seg_image_spa = clusters_spa[segments_flat - 1].reshape(h, w, 1)

        return multi_seg_image_spec, multi_seg_image_spa, segments, eigens

    def forward(self, x):
        device = x.device

        # SLIC superpixel decomposition
        x_numpy = x.permute(1, 2, 0).detach().cpu().numpy()
        segments = slic(x_numpy, n_segments=self.n_segments, compactness=self.compactness, channel_axis=2)
        segments = torch.tensor(segments).to(device)

        with torch.no_grad():
            feas = self.MAE_Extractor(x.unsqueeze(0)).squeeze(0)
            feas = feas.permute(1, 2, 0).to(device)
            graph_lap, graph_adj, spfea = self.build_graph_deep(segments, feas, alpha=self.alpha)

        # Learning Eigens
        eigens = self.GNN_Eigens(graph_adj, spfea)

        # Assigning Clusters
        clusters_spec_prob = F.softmax(self.Assignment_Spec(eigens), dim=1)  # N*k
        # Spectral Clusters
        clusters_spec = torch.argmax(clusters_spec_prob, dim=-1)  # N*1

        loss_emb = self.cal_loss_emb(eigens, graph_lap)
        loss_clu, r, q = self.cal_loss_clu(spfea)
        loss_reg = self.cal_loss_reg(r, clusters_spec_prob)

        # Spatial Clusters
        clusters_spa = torch.argmax(r, dim=-1)

        # Overall Loss
        loss = self.cfg["w_emb"] * loss_emb + self.cfg["w_clu"] * loss_clu + self.cfg["w_reg"] * loss_reg

        return segments, clusters_spec, clusters_spa, loss, self.centoids, eigens


