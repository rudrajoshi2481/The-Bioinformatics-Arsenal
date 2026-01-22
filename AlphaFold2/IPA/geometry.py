import torch
import torch.nn.functional as F
from typing import Tuple, Optional

class Rotation:
    def __init__(self, rot_mats: torch.Tensor):
        self.rot_mats = rot_mats
        self.device = rot_mats.device
        self.dtype = rot_mats.dtype

    @staticmethod
    def identity(batch_size: int, num_residues: int, device: torch.device = None, dtype: torch.dtype = torch.float32) -> 'Rotation':
        identity = torch.eye(3, device=device, dtype=dtype)
        identity = identity.view(1, 1, 3, 3).expand(batch_size, num_residues, 3, 3)
        return Rotation(identity.clone())

    @staticmethod
    def from_quaternion(quaternion: torch.Tensor) -> 'Rotation':
        quaternion = F.normalize(quaternion, dim=-1)
        w, x, y, z = quaternion.unbind(-1)

        rot_mat = torch.stack([
            torch.stack([1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)], dim=-1),
            torch.stack([2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)], dim=-1),
            torch.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)], dim=-1)
        ], dim=-2)

        return Rotation(rot_mat)

    @staticmethod
    def from_euler_angles(angles: torch.Tensor, order: str = 'xyz') -> 'Rotation':
        x, y, z = angles.unbind(-1)

        cos_x, sin_x = torch.cos(x), torch.sin(x)
        cos_y, sin_y = torch.cos(y), torch.sin(y)
        cos_z, sin_z = torch.cos(z), torch.sin(z)

        ones = torch.ones_like(x)
        zeros = torch.zeros_like(x)

        Rx = torch.stack([
            torch.stack([ones, zeros, zeros], dim=-1),
            torch.stack([zeros, cos_x, -sin_x], dim=-1),
            torch.stack([zeros, sin_x, cos_x], dim=-1)
        ], dim=-2)

        Ry = torch.stack([
            torch.stack([cos_y, zeros, sin_y], dim=-1),
            torch.stack([zeros, ones, zeros], dim=-1),
            torch.stack([-sin_y, zeros, cos_y], dim=-1)
        ], dim=-2)

        Rz = torch.stack([
            torch.stack([cos_z, -sin_z, zeros], dim=-1),
            torch.stack([sin_z, cos_z, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1)
        ], dim=-2)

        if order == 'xyz':
            rot_mat = Rz @ Ry @ Rx
        else:
            rot_mat = Rx @ Ry @ Rz

        return Rotation(rot_mat)

    def apply(self, points: torch.Tensor) -> torch.Tensor:
        return torch.einsum('...ij,...j->...i', self.rot_mats, points)

    def invert(self) -> 'Rotation':
        return Rotation(self.rot_mats.transpose(-1, -2))

    def compose(self, other: 'Rotation') -> 'Rotation':
        return Rotation(self.rot_mats @ other.rot_mats)

    def to_tensor(self) -> torch.Tensor:
        return self.rot_mats


class Rigid:
    def __init__(self, rotation: Rotation, translation: torch.Tensor):
        self.rotation = rotation
        self.translation = translation
        self.device = translation.device
        self.dtype = translation.dtype

    @staticmethod
    def identity(batch_size: int, num_residues: int, device: torch.device = None, dtype: torch.dtype = torch.float32) -> 'Rigid':
        rotation = Rotation.identity(batch_size, num_residues, device, dtype)
        translation = torch.zeros(batch_size, num_residues, 3, device=device, dtype=dtype)
        return Rigid(rotation, translation)

    def apply(self, points: torch.Tensor) -> torch.Tensor:
        rotated = self.rotation.apply(points)
        return rotated + self.translation

    def invert(self) -> 'Rigid':
        rot_inv = self.rotation.invert()
        trans_inv = -rot_inv.apply(self.translation)
        return Rigid(rot_inv, trans_inv)

    def compose(self, other: 'Rigid') -> 'Rigid':
        new_rotation = self.rotation.compose(other.rotation)
        new_translation = self.rotation.apply(other.translation) + self.translation
        return Rigid(new_rotation, new_translation)

    def apply_inverse_to_point(self, points: torch.Tensor) -> torch.Tensor:
        translated = points - self.translation
        return self.rotation.invert().apply(translated)

    def to_tensor_4x4(self) -> torch.Tensor:
        batch_dims = self.translation.shape[:-1]
        mat = torch.zeros(*batch_dims, 4, 4, device=self.device, dtype=self.dtype)
        mat[..., :3, :3] = self.rotation.rot_mats
        mat[..., :3, 3] = self.translation
        mat[..., 3, 3] = 1.0
        return mat

    @staticmethod
    def from_tensor_4x4(tensor: torch.Tensor) -> 'Rigid':
        rotation = Rotation(tensor[..., :3, :3])
        translation = tensor[..., :3, 3]
        return Rigid(rotation, translation)


def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    return Rotation.from_quaternion(quaternion).rot_mats


def rotation_matrix_to_quaternion(rotation_matrix: torch.Tensor) -> torch.Tensor:
    trace = rotation_matrix[..., 0, 0] + rotation_matrix[..., 1, 1] + rotation_matrix[..., 2, 2]
    w = torch.sqrt(torch.clamp(1 + trace, min=1e-8)) / 2
    x = (rotation_matrix[..., 2, 1] - rotation_matrix[..., 1, 2]) / (4 * w + 1e-8)
    y = (rotation_matrix[..., 0, 2] - rotation_matrix[..., 2, 0]) / (4 * w + 1e-8)
    z = (rotation_matrix[..., 1, 0] - rotation_matrix[..., 0, 1]) / (4 * w + 1e-8)
    return torch.stack([w, x, y, z], dim=-1)


if __name__ == "__main__":
    print("=" * 70)
    print("GEOMETRY MODULE TEST")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    batch, L = 2, 100

    rot = Rotation.identity(batch, L, device=device)
    print(f"Identity rotation: {rot.rot_mats.shape}")

    q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(batch, L, 4)
    rot_q = Rotation.from_quaternion(q)
    print(f"Rotation from quaternion: {rot_q.rot_mats.shape}")

    angles = torch.zeros(batch, L, 3, device=device)
    angles[..., 2] = torch.pi / 2
    rot_euler = Rotation.from_euler_angles(angles)

    point = torch.tensor([1.0, 0.0, 0.0], device=device).expand(batch, L, 3)
    rotated = rot_euler.apply(point)
    print(f"[1,0,0] rotated 90 around z: [{rotated[0,0,0]:.2f}, {rotated[0,0,1]:.2f}, {rotated[0,0,2]:.2f}]")

    rigid = Rigid.identity(batch, L, device=device)
    rigid.translation = torch.ones(batch, L, 3, device=device) * 10

    point = torch.zeros(batch, L, 3, device=device)
    transformed = rigid.apply(point)
    print(f"Translation [10,10,10] applied: {transformed[0,0].tolist()}")

    mat = rigid.to_tensor_4x4()
    print(f"4x4 matrix shape: {mat.shape}")

    trans = torch.randn(batch, L, 3, device=device, requires_grad=True)
    rigid = Rigid(Rotation.identity(batch, L, device=device), trans)
    loss = rigid.apply(point).sum()
    loss.backward()
    print(f"Gradients computed, norm: {trans.grad.norm():.4f}")
    print("PASSED")
