import torch
from torch_geometric.nn import GCNConv

class TGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, improved=False, cached=False, add_self_loops=True):
        super(TGCN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self._create_parameters_and_layers()

    def _create_update_gate_parameters_and_layers(self):
        self.conv_z = GCNConv(self.in_channels, self.out_channels, improved=self.improved, cached=self.cached, add_self_loops=self.add_self_loops)
        self.linear_z = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_reset_gate_parameters_and_layers(self):
        self.conv_r = GCNConv(self.in_channels, self.out_channels, improved=self.improved, cached=self.cached, add_self_loops=self.add_self_loops)
        self.linear_r = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_candidate_state_parameters_and_layers(self):
        self.conv_h = GCNConv(self.in_channels, self.out_channels, improved=self.improved, cached=self.cached, add_self_loops=self.add_self_loops)
        self.linear_h = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], X.shape[1], self.out_channels).to(X.device)
        return H

    def _calculate_update_gate(self, X, edge_index, edge_weight, H):
        X = X.view(-1, X.shape[-1])  # Flatten the batch for GCNConv
        conv_z_out = self.conv_z(X, edge_index, edge_weight)
        conv_z_out = conv_z_out.view(H.shape[0], H.shape[1], -1)  # Reshape back to batch size
        Z = torch.cat([conv_z_out, H], axis=2)
        Z = self.linear_z(Z)
        Z = torch.sigmoid(Z)
        return Z

    def _calculate_reset_gate(self, X, edge_index, edge_weight, H):
        X = X.view(-1, X.shape[-1])  # Flatten the batch for GCNConv
        conv_r_out = self.conv_r(X, edge_index, edge_weight)
        conv_r_out = conv_r_out.view(H.shape[0], H.shape[1], -1)  # Reshape back to batch size
        R = torch.cat([conv_r_out, H], axis=2)
        R = self.linear_r(R)
        R = torch.sigmoid(R)
        return R

    def _calculate_candidate_state(self, X, edge_index, edge_weight, H, R):
        X = X.view(-1, X.shape[-1])  # Flatten the batch for GCNConv
        conv_h_out = self.conv_h(X, edge_index, edge_weight)
        conv_h_out = conv_h_out.view(H.shape[0], H.shape[1], -1)  # Reshape back to batch size
        H_tilde = torch.cat([conv_h_out, H * R], axis=2)
        H_tilde = self.linear_h(H_tilde)
        H_tilde = torch.tanh(H_tilde)
        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde
        return H

    def forward(self, X, edge_index, edge_weight=None, H=None):
        H = self._set_hidden_state(X, H)
        Z = self._calculate_update_gate(X, edge_index, edge_weight, H)
        R = self._calculate_reset_gate(X, edge_index, edge_weight, H)
        H_tilde = self._calculate_candidate_state(X, edge_index, edge_weight, H, R)
        H = self._calculate_hidden_state(Z, H, H_tilde)
        return H
