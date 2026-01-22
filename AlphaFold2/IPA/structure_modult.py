import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple

from geometry import Rotation, Rigid


class InvariantPointAttention(nn.Module):
    def __init__(
        self,
        d_single: int,
        d_pair: int,
        num_heads: int = 12,
        num_query_points: int = 4,
        num_value_points: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_single = d_single
        self.d_pair = d_pair
        self.num_heads = num_heads
        self.num_query_points = num_query_points
        self.num_value_points = num_value_points
        self.d_head = d_single // num_heads

        self.q_proj = nn.Linear(d_single, num_heads * self.d_head)
        self.k_proj = nn.Linear(d_single, num_heads * self.d_head)
        self.v_proj = nn.Linear(d_single, num_heads * self.d_head)

        self.q_point_proj = nn.Linear(d_single, num_heads * num_query_points * 3)
        self.k_point_proj = nn.Linear(d_single, num_heads * num_query_points * 3)
        self.v_point_proj = nn.Linear(d_single, num_heads * num_value_points * 3)

        self.pair_bias_proj = nn.Linear(d_pair, num_heads)

        output_dim = num_heads * self.d_head + num_heads * num_value_points * 4 + num_heads * d_pair
        self.out_proj = nn.Linear(output_dim, d_single)

        self.w_c = nn.Parameter(torch.tensor(1.0 / math.sqrt(3)))
        self.w_l = nn.Parameter(torch.tensor(1.0 / 3))
        self.gamma = nn.Parameter(torch.ones(num_heads))

        self.layer_norm = nn.LayerNorm(d_single)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        single: torch.Tensor,
        pair: torch.Tensor,
        frames: Rigid,
        mask: torch.Tensor
    ) -> torch.Tensor:
        batch, L, _ = single.shape
        
        single_norm = self.layer_norm(single)

        q = self.q_proj(single_norm).view(batch, L, self.num_heads, self.d_head)
        k = self.k_proj(single_norm).view(batch, L, self.num_heads, self.d_head)
        v = self.v_proj(single_norm).view(batch, L, self.num_heads, self.d_head)

        q_points = self.q_point_proj(single_norm).view(batch, L, self.num_heads, self.num_query_points, 3)
        k_points = self.k_point_proj(single_norm).view(batch, L, self.num_heads, self.num_query_points, 3)
        v_points = self.v_point_proj(single_norm).view(batch, L, self.num_heads, self.num_value_points, 3)

        rot_mats = frames.rotation.rot_mats
        trans = frames.translation

        q_points_global = torch.einsum('blij,blhpj->blhpi', rot_mats, q_points) + trans.unsqueeze(2).unsqueeze(3)
        k_points_global = torch.einsum('blij,blhpj->blhpi', rot_mats, k_points) + trans.unsqueeze(2).unsqueeze(3)
        v_points_global = torch.einsum('blij,blhpj->blhpi', rot_mats, v_points) + trans.unsqueeze(2).unsqueeze(3)

        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        seq_logits = torch.matmul(q_t, k_t.transpose(-1, -2)) * self.w_c / math.sqrt(self.d_head)

        q_pts = q_points_global.permute(0, 2, 1, 3, 4)
        k_pts = k_points_global.permute(0, 2, 1, 3, 4)
        dist_sq = torch.sum((q_pts.unsqueeze(3) - k_pts.unsqueeze(2)) ** 2, dim=-1)
        geo_logits = -0.5 * torch.sum(dist_sq, dim=-1) * self.gamma.view(1, -1, 1, 1) * self.w_l

        pair_bias = self.pair_bias_proj(pair).permute(0, 3, 1, 2)

        logits = seq_logits + geo_logits + pair_bias

        mask_2d = (mask.unsqueeze(1) * mask.unsqueeze(2)).unsqueeze(1)
        logits = logits.masked_fill(~mask_2d.bool(), -1e9)
        attn_weights = F.softmax(logits, dim=-1)
        attn_weights = self.dropout(attn_weights)

        v_t = v.transpose(1, 2)
        seq_output = torch.matmul(attn_weights, v_t)

        v_pts = v_points_global.permute(0, 2, 1, 3, 4)
        pts_output = torch.einsum('bhij,bhjpd->bhipd', attn_weights, v_pts)

        rot_mats_inv = rot_mats.transpose(-1, -2)
        pts_output = pts_output.permute(0, 2, 1, 3, 4)
        pts_local = torch.einsum('blij,blhpj->blhpi', rot_mats_inv, pts_output - trans.unsqueeze(2).unsqueeze(3))

        pts_norm = torch.norm(pts_local, dim=-1, keepdim=True)
        pts_features = torch.cat([pts_norm, pts_local], dim=-1)

        pair_output = torch.einsum('bhij,bijd->bhid', attn_weights, pair)

        seq_output = seq_output.permute(0, 2, 1, 3).reshape(batch, L, -1)
        pts_features = pts_features.reshape(batch, L, -1)
        pair_output = pair_output.permute(0, 2, 1, 3).reshape(batch, L, -1)

        combined = torch.cat([seq_output, pts_features, pair_output], dim=-1)
        output = self.out_proj(combined)

        return output


class BackboneUpdate(nn.Module):
    def __init__(self, d_single: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_single)
        self.linear = nn.Linear(d_single, 6)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, single: torch.Tensor, frames: Rigid) -> Rigid:
        batch, L, _ = single.shape
        x = self.layer_norm(single)
        update = self.linear(x)

        quat_update = update[..., :3]
        trans_update = update[..., 3:]

        angle = torch.norm(quat_update, dim=-1, keepdim=True)
        axis = quat_update / (angle + 1e-8)

        half_angle = angle / 2
        w = torch.cos(half_angle)
        xyz = axis * torch.sin(half_angle)
        quaternion = torch.cat([w, xyz], dim=-1)

        rot_update = Rotation.from_quaternion(quaternion)

        new_rotation = frames.rotation.compose(rot_update)
        new_translation = frames.translation + frames.rotation.apply(trans_update)

        return Rigid(new_rotation, new_translation)


class Transition(nn.Module):
    def __init__(self, d_input: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        d_hidden = d_input * expansion
        self.layer_norm = nn.LayerNorm(d_input)
        self.linear1 = nn.Linear(d_input, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_input)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class StructureModule(nn.Module):
    def __init__(
        self,
        d_single: int = 384,
        d_pair: int = 128,
        num_layers: int = 8,
        num_heads: int = 12,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_layers = num_layers
        
        self.input_proj = nn.Linear(d_single, d_single)
        
        self.ipa_layers = nn.ModuleList([
            InvariantPointAttention(d_single, d_pair, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.transitions = nn.ModuleList([
            Transition(d_single, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.backbone_updates = nn.ModuleList([
            BackboneUpdate(d_single)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        single: torch.Tensor,
        pair: torch.Tensor,
        seq_mask: torch.Tensor,
        verbose: bool = False
    ) -> Tuple[torch.Tensor, Rigid]:
        batch, L, _ = single.shape
        device = single.device
        
        single = self.input_proj(single)
        frames = Rigid.identity(batch, L, device=device, dtype=single.dtype)
        
        for i in range(self.num_layers):
            single = single + self.dropout(self.ipa_layers[i](single, pair, frames, seq_mask))
            single = single + self.transitions[i](single)
            frames = self.backbone_updates[i](single, frames)
            
            if verbose:
                print(f"  Structure Module layer {i+1}/{self.num_layers}")
        
        positions = frames.translation
        return positions, frames


if __name__ == "__main__":
    print("=" * 70)
    print("STRUCTURE MODULE TEST")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    batch, L = 2, 64
    d_single, d_pair = 384, 128
    num_layers = 2

    single = torch.randn(batch, L, d_single, device=device)
    pair = torch.randn(batch, L, L, d_pair, device=device)
    seq_mask = torch.ones(batch, L, device=device)

    print(f"Input: Single {single.shape}, Pair {pair.shape}")

    ipa = InvariantPointAttention(d_single, d_pair).to(device)
    frames = Rigid.identity(batch, L, device=device)
    out = ipa(single, pair, frames, seq_mask)
    print(f"IPA output: {out.shape}")

    sm = StructureModule(d_single, d_pair, num_layers=num_layers).to(device)
    positions, final_frames = sm(single, pair, seq_mask)
    print(f"Positions: {positions.shape}")

    params = sum(p.numel() for p in sm.parameters())
    print(f"Parameters ({num_layers} layers): {params:,}")

    loss = positions.sum()
    loss.backward()
    print("Gradients computed successfully")
    print("PASSED")
