import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class FAPELoss(nn.Module):
    def __init__(self, clamp_distance: float = 10.0, loss_unit_distance: float = 10.0, eps: float = 1e-8):
        super().__init__()
        self.clamp_distance = clamp_distance
        self.loss_unit_distance = loss_unit_distance
        self.eps = eps

    def forward(
        self,
        pred_frames: torch.Tensor,
        target_frames: torch.Tensor,
        pred_positions: torch.Tensor,
        target_positions: torch.Tensor,
        frames_mask: torch.Tensor,
        positions_mask: torch.Tensor
    ) -> torch.Tensor:
        batch, L = pred_positions.shape[:2]

        pred_rot = pred_frames[:, :, :3, :3]
        pred_trans = pred_frames[:, :, :3, 3]
        target_rot = target_frames[:, :, :3, :3]
        target_trans = target_frames[:, :, :3, 3]

        pred_local = pred_positions.unsqueeze(1) - pred_trans.unsqueeze(2)
        target_local = target_positions.unsqueeze(1) - target_trans.unsqueeze(2)

        pred_rot_t = pred_rot.transpose(-1, -2)
        target_rot_t = target_rot.transpose(-1, -2)

        pred_aligned = torch.einsum('bfij,bfpj->bfpi', pred_rot_t, pred_local)
        target_aligned = torch.einsum('bfij,bfpj->bfpi', target_rot_t, target_local)

        error = torch.sqrt(torch.sum((pred_aligned - target_aligned) ** 2, dim=-1) + self.eps)

        if self.clamp_distance is not None:
            error = torch.clamp(error, max=self.clamp_distance)

        error = error / self.loss_unit_distance

        mask = frames_mask.unsqueeze(2) * positions_mask.unsqueeze(1)
        loss = (error * mask).sum() / (mask.sum() + self.eps)

        return loss


class DistogramLoss(nn.Module):
    def __init__(self, first_break: float = 2.3125, last_break: float = 21.6875, num_bins: int = 64):
        super().__init__()
        self.num_bins = num_bins
        breaks = torch.linspace(first_break, last_break, num_bins - 1)
        self.register_buffer('breaks', breaks)

    def forward(self, logits: torch.Tensor, target_positions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, L = target_positions.shape[:2]

        diff = target_positions.unsqueeze(2) - target_positions.unsqueeze(1)
        distances = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-8)

        target_bins = torch.sum(distances.unsqueeze(-1) > self.breaks.view(1, 1, 1, -1), dim=-1).long()

        loss = F.cross_entropy(logits.reshape(-1, self.num_bins), target_bins.reshape(-1), reduction='none')
        loss = loss.reshape(batch, L, L)

        pair_mask = mask.unsqueeze(2) * mask.unsqueeze(1)
        loss = (loss * pair_mask).sum() / (pair_mask.sum() + 1e-8)

        return loss


class MaskedMSALoss(nn.Module):
    def __init__(self, num_classes: int = 23):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, N_seq, L, num_classes = logits.shape

        loss = F.cross_entropy(logits.reshape(-1, num_classes), target.reshape(-1), reduction='none')
        loss = loss.reshape(batch, N_seq, L)
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)

        return loss


class StructuralViolationLoss(nn.Module):
    def __init__(self, bond_length_tolerance: float = 0.4, clash_overlap_tolerance: float = 1.5):
        super().__init__()
        self.bond_length_tolerance = bond_length_tolerance
        self.clash_overlap_tolerance = clash_overlap_tolerance
        self.ideal_ca_ca_distance = 3.8

    def forward(self, positions: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch, L = positions.shape[:2]

        if len(positions.shape) == 4:
            ca_positions = positions[:, :, 1, :]
            ca_mask = mask[:, :, 1] if len(mask.shape) == 3 else mask
        else:
            ca_positions = positions
            ca_mask = mask

        ca_diff = ca_positions[:, 1:] - ca_positions[:, :-1]
        ca_distances = torch.sqrt(torch.sum(ca_diff ** 2, dim=-1) + 1e-8)

        bond_deviation = torch.abs(ca_distances - self.ideal_ca_ca_distance)
        bond_violation = F.relu(bond_deviation - self.bond_length_tolerance)

        bond_mask = ca_mask[:, 1:] * ca_mask[:, :-1]
        bond_loss = (bond_violation * bond_mask).sum() / (bond_mask.sum() + 1e-8)

        all_diff = ca_positions.unsqueeze(2) - ca_positions.unsqueeze(1)
        all_distances = torch.sqrt(torch.sum(all_diff ** 2, dim=-1) + 1e-8)

        L_range = torch.arange(L, device=positions.device)
        seq_sep = torch.abs(L_range.unsqueeze(0) - L_range.unsqueeze(1))
        non_bonded_mask = (seq_sep > 2).float()

        clash_violation = F.relu(self.clash_overlap_tolerance - all_distances)
        pair_mask = ca_mask.unsqueeze(2) * ca_mask.unsqueeze(1) * non_bonded_mask
        clash_loss = (clash_violation * pair_mask).sum() / (pair_mask.sum() + 1e-8)

        return {'bond_loss': bond_loss, 'clash_loss': clash_loss, 'total_violation': bond_loss + clash_loss}


class LDDTLoss(nn.Module):
    def __init__(self, num_bins: int = 50, cutoff: float = 15.0):
        super().__init__()
        self.num_bins = num_bins
        self.cutoff = cutoff
        bin_edges = torch.linspace(0, 1, num_bins + 1)
        self.register_buffer('bin_edges', bin_edges)

    def compute_lddt(self, pred_positions: torch.Tensor, target_positions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, L = pred_positions.shape[:2]

        pred_diff = pred_positions.unsqueeze(2) - pred_positions.unsqueeze(1)
        pred_dist = torch.sqrt(torch.sum(pred_diff ** 2, dim=-1) + 1e-8)

        target_diff = target_positions.unsqueeze(2) - target_positions.unsqueeze(1)
        target_dist = torch.sqrt(torch.sum(target_diff ** 2, dim=-1) + 1e-8)

        within_cutoff = (target_dist < self.cutoff).float()
        eye = torch.eye(L, device=pred_positions.device).unsqueeze(0)
        within_cutoff = within_cutoff * (1 - eye)

        pair_mask = mask.unsqueeze(2) * mask.unsqueeze(1) * within_cutoff
        dist_diff = torch.abs(pred_dist - target_dist)

        score = 0.25 * ((dist_diff < 0.5).float() + (dist_diff < 1.0).float() +
                        (dist_diff < 2.0).float() + (dist_diff < 4.0).float())

        lddt = (score * pair_mask).sum(dim=-1) / (pair_mask.sum(dim=-1) + 1e-8)
        return lddt

    def forward(self, logits: torch.Tensor, pred_positions: torch.Tensor,
                target_positions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        true_lddt = self.compute_lddt(pred_positions, target_positions, mask)
        target_bins = torch.bucketize(true_lddt, self.bin_edges[1:-1])

        batch, L, num_bins = logits.shape
        loss = F.cross_entropy(logits.reshape(-1, num_bins), target_bins.reshape(-1), reduction='none')
        loss = loss.reshape(batch, L)
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)

        return loss


class AlphaFoldLoss(nn.Module):
    def __init__(
        self,
        fape_weight: float = 0.5,
        distogram_weight: float = 0.3,
        masked_msa_weight: float = 2.0,
        violation_weight: float = 1.0,
        lddt_weight: float = 0.01,
        pae_weight: float = 0.1
    ):
        super().__init__()

        self.weights = {
            'fape': fape_weight, 'distogram': distogram_weight,
            'masked_msa': masked_msa_weight, 'violation': violation_weight,
            'lddt': lddt_weight, 'pae': pae_weight
        }

        self.fape_loss = FAPELoss()
        self.distogram_loss = DistogramLoss()
        self.masked_msa_loss = MaskedMSALoss()
        self.violation_loss = StructuralViolationLoss()
        self.lddt_loss = LDDTLoss()

    def forward(self, predictions: Dict, targets: Dict, masks: Dict) -> Tuple[torch.Tensor, Dict]:
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)

        if 'pred_frames' in predictions and 'target_frames' in targets:
            fape = self.fape_loss(predictions['pred_frames'], targets['target_frames'],
                                  predictions['pred_positions'], targets['target_positions'],
                                  masks['frames_mask'], masks['positions_mask'])
            loss_dict['fape'] = fape
            total_loss = total_loss + self.weights['fape'] * fape

        if 'distogram_logits' in predictions:
            distogram = self.distogram_loss(predictions['distogram_logits'],
                                            targets['target_positions'], masks['seq_mask'])
            loss_dict['distogram'] = distogram
            total_loss = total_loss + self.weights['distogram'] * distogram

        if 'pred_positions' in predictions:
            violations = self.violation_loss(predictions['pred_positions'], masks['positions_mask'])
            loss_dict['violation'] = violations['total_violation']
            total_loss = total_loss + self.weights['violation'] * violations['total_violation']

        loss_dict['total'] = total_loss
        return total_loss, loss_dict


if __name__ == "__main__":
    print("=" * 70)
    print("LOSS FUNCTIONS TEST")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    batch, L = 2, 64

    pred_pos = torch.randn(batch, L, 3, device=device)
    target_pos = pred_pos + torch.randn(batch, L, 3, device=device) * 0.5
    mask = torch.ones(batch, L, device=device)

    frames = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).expand(batch, L, 4, 4).clone()
    frames[:, :, :3, 3] = pred_pos
    target_frames = frames.clone()
    target_frames[:, :, :3, 3] = target_pos

    fape_loss = FAPELoss().to(device)
    fape = fape_loss(frames, target_frames, pred_pos, target_pos, mask, mask)
    print(f"FAPE Loss: {fape.item():.6f}")

    dist_logits = torch.randn(batch, L, L, 64, device=device)
    dist_loss = DistogramLoss().to(device)
    dist = dist_loss(dist_logits, target_pos, mask)
    print(f"Distogram Loss: {dist.item():.6f}")

    viol_loss = StructuralViolationLoss().to(device)
    viols = viol_loss(pred_pos, mask)
    print(f"Violation Loss: {viols['total_violation'].item():.6f}")

    pred_pos.requires_grad_(True)
    fape = fape_loss(frames, target_frames, pred_pos, target_pos, mask, mask)
    fape.backward()
    print(f"Gradients computed, norm: {pred_pos.grad.norm():.4f}")
    print("PASSED")
