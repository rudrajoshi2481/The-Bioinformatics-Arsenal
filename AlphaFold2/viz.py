import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Tuple
import os


def plot_structure_3d(
    positions: torch.Tensor,
    plddt: Optional[torch.Tensor] = None,
    title: str = "Predicted Structure",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    if isinstance(positions, torch.Tensor):
        positions = positions.detach().cpu().numpy()
    if plddt is not None and isinstance(plddt, torch.Tensor):
        plddt = plddt.detach().cpu().numpy()

    if len(positions.shape) == 3:
        positions = positions[0]
    if plddt is not None and len(plddt.shape) == 2:
        plddt = plddt[0]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

    if plddt is not None:
        colors = plt.cm.RdYlGn(plddt / 100.0)
        scatter = ax.scatter(x, y, z, c=plddt, cmap='RdYlGn', vmin=0, vmax=100, s=50, alpha=0.8)
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, label='pLDDT')
    else:
        colors = np.linspace(0, 1, len(x))
        scatter = ax.scatter(x, y, z, c=colors, cmap='viridis', s=50, alpha=0.8)
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, label='Residue Index')

    for i in range(len(x) - 1):
        ax.plot([x[i], x[i+1]], [y[i], y[i+1]], [z[i], z[i+1]], 'k-', alpha=0.3, linewidth=1)

    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title(title)

    max_range = np.max([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]) / 2
    mid_x, mid_y, mid_z = (x.max() + x.min()) / 2, (y.max() + y.min()) / 2, (z.max() + z.min()) / 2
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved structure plot to {save_path}")

    return fig


def plot_plddt_per_residue(
    plddt: torch.Tensor,
    title: str = "pLDDT per Residue",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4)
) -> plt.Figure:
    if isinstance(plddt, torch.Tensor):
        plddt = plddt.detach().cpu().numpy()

    if len(plddt.shape) == 2:
        plddt = plddt[0]

    fig, ax = plt.subplots(figsize=figsize)

    residues = np.arange(1, len(plddt) + 1)
    colors = plt.cm.RdYlGn(plddt / 100.0)

    ax.bar(residues, plddt, color=colors, width=1.0, edgecolor='none')

    ax.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='Very High (>90)')
    ax.axhline(y=70, color='orange', linestyle='--', alpha=0.5, label='Confident (>70)')
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Low (>50)')

    ax.set_xlabel('Residue Index')
    ax.set_ylabel('pLDDT Score')
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.set_xlim(0, len(plddt) + 1)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved pLDDT plot to {save_path}")

    return fig


def plot_pae_matrix(
    pae: torch.Tensor,
    title: str = "Predicted Aligned Error (PAE)",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    if isinstance(pae, torch.Tensor):
        pae = pae.detach().cpu().numpy()

    if len(pae.shape) == 3:
        pae = pae[0]

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(pae, cmap='Greens_r', vmin=0, vmax=31)
    cbar = plt.colorbar(im, ax=ax, label='Expected Position Error (Å)')

    ax.set_xlabel('Aligned Residue')
    ax.set_ylabel('Scored Residue')
    ax.set_title(title)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved PAE plot to {save_path}")

    return fig


def plot_distance_matrix(
    positions: torch.Tensor,
    title: str = "Distance Matrix",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    if isinstance(positions, torch.Tensor):
        positions = positions.detach().cpu().numpy()

    if len(positions.shape) == 3:
        positions = positions[0]

    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=-1))

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(distances, cmap='viridis', vmin=0)
    cbar = plt.colorbar(im, ax=ax, label='Distance (Å)')

    ax.set_xlabel('Residue Index')
    ax.set_ylabel('Residue Index')
    ax.set_title(title)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved distance matrix to {save_path}")

    return fig


def plot_contact_map(
    positions: torch.Tensor,
    threshold: float = 8.0,
    title: str = "Contact Map",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    if isinstance(positions, torch.Tensor):
        positions = positions.detach().cpu().numpy()

    if len(positions.shape) == 3:
        positions = positions[0]

    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=-1))
    contacts = (distances < threshold).astype(float)

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(contacts, cmap='Blues', vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax, label='Contact')

    ax.set_xlabel('Residue Index')
    ax.set_ylabel('Residue Index')
    ax.set_title(f"{title} (threshold={threshold}Å)")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved contact map to {save_path}")

    return fig


def save_pdb(
    positions: torch.Tensor,
    save_path: str,
    chain_id: str = 'A',
    plddt: Optional[torch.Tensor] = None
):
    if isinstance(positions, torch.Tensor):
        positions = positions.detach().cpu().numpy()
    if plddt is not None and isinstance(plddt, torch.Tensor):
        plddt = plddt.detach().cpu().numpy()

    if len(positions.shape) == 3:
        positions = positions[0]
    if plddt is not None and len(plddt.shape) == 2:
        plddt = plddt[0]

    with open(save_path, 'w') as f:
        f.write("HEADER    ALPHAFOLD2 PREDICTION\n")
        f.write("TITLE     PREDICTED STRUCTURE\n")

        for i, pos in enumerate(positions):
            atom_num = i + 1
            res_num = i + 1
            x, y, z = pos
            b_factor = plddt[i] if plddt is not None else 0.0

            line = f"ATOM  {atom_num:5d}  CA  ALA {chain_id}{res_num:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00{b_factor:6.2f}           C\n"
            f.write(line)

        f.write("END\n")

    print(f"Saved PDB to {save_path}")


def visualize_prediction(
    outputs: dict,
    output_dir: str = "./viz_output",
    prefix: str = "alphafold2"
):
    os.makedirs(output_dir, exist_ok=True)

    positions = outputs['positions']
    plddt = outputs.get('plddt', None)
    pae = outputs.get('pae', None)

    fig1 = plot_structure_3d(
        positions, plddt,
        title=f"{prefix} - 3D Structure",
        save_path=os.path.join(output_dir, f"{prefix}_structure_3d.png")
    )
    plt.close(fig1)

    if plddt is not None:
        fig2 = plot_plddt_per_residue(
            plddt,
            title=f"{prefix} - pLDDT per Residue",
            save_path=os.path.join(output_dir, f"{prefix}_plddt.png")
        )
        plt.close(fig2)

    if pae is not None:
        fig3 = plot_pae_matrix(
            pae,
            title=f"{prefix} - Predicted Aligned Error",
            save_path=os.path.join(output_dir, f"{prefix}_pae.png")
        )
        plt.close(fig3)

    fig4 = plot_distance_matrix(
        positions,
        title=f"{prefix} - Distance Matrix",
        save_path=os.path.join(output_dir, f"{prefix}_distance_matrix.png")
    )
    plt.close(fig4)

    fig5 = plot_contact_map(
        positions,
        title=f"{prefix} - Contact Map",
        save_path=os.path.join(output_dir, f"{prefix}_contact_map.png")
    )
    plt.close(fig5)

    save_pdb(
        positions, 
        os.path.join(output_dir, f"{prefix}_structure.pdb"),
        plddt=plddt
    )

    print(f"\nAll visualizations saved to {output_dir}/")


if __name__ == "__main__":
    print("=" * 70)
    print("VISUALIZATION TEST")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    L = 64
    t = torch.linspace(0, 4 * np.pi, L)
    x = torch.cos(t) * (1 + t / 10)
    y = torch.sin(t) * (1 + t / 10)
    z = t / 2

    positions = torch.stack([x, y, z], dim=-1).unsqueeze(0)
    plddt = 50 + 50 * torch.sin(torch.linspace(0, 2 * np.pi, L)).unsqueeze(0)
    pae = torch.abs(torch.randn(1, L, L)) * 10

    outputs = {
        'positions': positions,
        'plddt': plddt,
        'pae': pae
    }

    print(f"Positions shape: {positions.shape}")
    print(f"pLDDT shape: {plddt.shape}")
    print(f"PAE shape: {pae.shape}")

    visualize_prediction(outputs, output_dir="./viz_output", prefix="test_structure")

    print("PASSED")
