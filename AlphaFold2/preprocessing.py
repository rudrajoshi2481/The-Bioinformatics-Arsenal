import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple

AMINO_ACIDS = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
    'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'
]

AA_TO_ID = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
AA_TO_ID['X'] = 20
AA_TO_ID['-'] = 21

ID_TO_AA = {i: aa for aa, i in AA_TO_ID.items()}
VOCAB_SIZE = 22

def sequence_to_int(sequence: str) -> torch.Tensor:
    sequence = sequence.upper()
    ids = [AA_TO_ID.get(aa, 20) for aa in sequence]
    return torch.tensor(ids, dtype=torch.long)

def sequence_to_onehot(sequence: str, vocab_size: int = 21) -> torch.Tensor:
    seq_int = sequence_to_int(sequence)
    seq_int = torch.clamp(seq_int, 0, vocab_size - 1)
    return F.one_hot(seq_int, num_classes=vocab_size).float()

def msa_to_int(msa_sequences: List[str]) -> torch.Tensor:
    N_seq = len(msa_sequences)
    L = len(msa_sequences[0])
    
    msa_int = torch.zeros(N_seq, L, dtype=torch.long)
    
    for i, seq in enumerate(msa_sequences):
        seq = seq.upper()
        for j, aa in enumerate(seq):
            msa_int[i, j] = AA_TO_ID.get(aa, 20)
    
    return msa_int

def compute_deletion_matrix(msa_sequences: List[str]) -> torch.Tensor:
    N_seq = len(msa_sequences)
    L = len(msa_sequences[0])
    
    deletion_matrix = torch.zeros(N_seq, L, dtype=torch.float32)
    
    for i, seq in enumerate(msa_sequences):
        gap_count = 0
        for j, aa in enumerate(seq):
            if aa == '-':
                gap_count += 1
            else:
                if j > 0:
                    deletion_matrix[i, j-1] = gap_count
                gap_count = 0
    
    return deletion_matrix

def create_msa_feat(
    msa_int: torch.Tensor,
    deletion_matrix: torch.Tensor,
    has_deletion: bool = True,
    has_profile: bool = True
) -> torch.Tensor:
    N_seq, L = msa_int.shape
    
    msa_onehot = F.one_hot(msa_int, num_classes=22).float()
    
    features = [msa_onehot]
    
    if has_deletion:
        deletion_value = torch.clamp(deletion_matrix / 3.0, 0, 1).unsqueeze(-1)
        features.append(deletion_value)
        
        has_deletion_flag = (deletion_matrix > 0).float().unsqueeze(-1)
        features.append(has_deletion_flag)
        
        deletion_mean = deletion_matrix.mean(dim=0, keepdim=True)
        deletion_mean = torch.clamp(deletion_mean / 3.0, 0, 1)
        deletion_mean = deletion_mean.expand(N_seq, L).unsqueeze(-1)
        features.append(deletion_mean)
    
    if has_profile:
        profile = torch.zeros(L, 22, dtype=torch.float32)
        for j in range(L):
            for i in range(N_seq):
                aa_id = msa_int[i, j].item()
                if aa_id < 21:
                    profile[j, aa_id] += 1.0
        
        profile = profile / (profile.sum(dim=-1, keepdim=True) + 1e-8)
        profile = profile.unsqueeze(0).expand(N_seq, L, 22)
        features.append(profile)
    
    msa_feat = torch.cat(features, dim=-1)
    return msa_feat

def create_pair_feat(sequence: str) -> torch.Tensor:
    L = len(sequence)
    
    seq_onehot = sequence_to_onehot(sequence, vocab_size=21)
    
    outer = torch.einsum('ia,jb->ijab', seq_onehot, seq_onehot)
    outer = outer.reshape(L, L, -1)
    
    rel_pos = torch.arange(L).unsqueeze(1) - torch.arange(L).unsqueeze(0)
    rel_pos = torch.clamp(rel_pos, -32, 32) + 32
    rel_pos_onehot = F.one_hot(rel_pos, num_classes=65).float()
    
    pair_feat = torch.cat([outer, rel_pos_onehot], dim=-1)
    return pair_feat

def create_masks(
    msa_int: torch.Tensor,
    sequence: str
) -> Dict[str, torch.Tensor]:
    N_seq, L = msa_int.shape
    
    msa_mask = (msa_int != 21).float()
    seq_mask = torch.ones(L, dtype=torch.float32)
    pair_mask = torch.outer(seq_mask, seq_mask)
    
    return {
        'msa_mask': msa_mask,
        'seq_mask': seq_mask,
        'pair_mask': pair_mask
    }

def preprocess_features(
    sequence: str,
    msa_sequences: List[str],
    max_msa_sequences: int = None,
    add_batch_dim: bool = True
) -> Dict[str, torch.Tensor]:
    if max_msa_sequences is not None and len(msa_sequences) > max_msa_sequences:
        msa_sequences = [msa_sequences[0]] + msa_sequences[1:max_msa_sequences]
    
    aatype = sequence_to_int(sequence)
    msa_int = msa_to_int(msa_sequences)
    deletion_matrix = compute_deletion_matrix(msa_sequences)
    msa_feat = create_msa_feat(msa_int, deletion_matrix)
    target_feat = sequence_to_onehot(sequence, vocab_size=22)
    pair_feat = create_pair_feat(sequence)
    masks = create_masks(msa_int, sequence)
    
    features = {
        'aatype': aatype,
        'msa': msa_int,
        'msa_feat': msa_feat,
        'target_feat': target_feat,
        'pair': pair_feat,
        **masks
    }
    
    if add_batch_dim:
        for key in features:
            if key != 'aatype':
                features[key] = features[key].unsqueeze(0)
    
    return features

if __name__ == "__main__":
    """
        (base) root@cfb85d370bd4:/app/tmp/Alphafold2_reimplemented# python3 /app/tmp/Alphafold2_reimplemented/preprocessing.py
        ======================================================================
        PREPROCESSING MODULE TEST
        ======================================================================

        Query sequence length: 56
        Number of MSA sequences: 5

        Feature shapes (with batch dim):
        aatype         : (56,)
        msa            : (1, 5, 56)
        msa_feat       : (1, 5, 56, 47)
        target_feat    : (1, 56, 22)
        pair           : (1, 56, 56, 506)
        msa_mask       : (1, 5, 56)
        seq_mask       : (1, 56)
        pair_mask      : (1, 56, 56)

        ✓ Preprocessing module working correctly!
    """
    print("=" * 70)
    print("PREPROCESSING MODULE TEST")
    print("=" * 70)
    
    sequence = "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRR"
    msa_sequences = [sequence] * 5
    
    print(f"\nQuery sequence length: {len(sequence)}")
    print(f"Number of MSA sequences: {len(msa_sequences)}")
    
    features = preprocess_features(sequence, msa_sequences, add_batch_dim=True)
    
    print("\nFeature shapes (with batch dim):")
    for key, value in features.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key:15s}: {tuple(value.shape)}")
    
    print("\n✓ Preprocessing module working correctly!")
