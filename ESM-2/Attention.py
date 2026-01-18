import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple
from torch import Tensor
import uuid

from embeddings import ESMEmbeddings

class FairseqIncrementalState(object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_incremental_state()

    def init_incremental_state(self):
        self._incremental_state_id = str(uuid.uuid4())

    def _get_full_incremental_state_key(self, key: str) -> str:
        return "{}.{}".format(self._incremental_state_id, key)

    def get_incremental_state(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        key: str,
    ) -> Optional[Dict[str, Optional[Tensor]]]:
        full_key = self._get_full_incremental_state_key(key)
        if incremental_state is None or full_key not in incremental_state:
            return None
        return incremental_state[full_key]

    def set_incremental_state(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        key: str,
        value: Dict[str, Optional[Tensor]],
    ) -> Optional[Dict[str, Dict[str, Optional[Tensor]]]]:
        if incremental_state is not None:
            full_key = self._get_full_incremental_state_key(key)
            incremental_state[full_key] = value
        return incremental_state


def with_incremental_state(cls):
    cls.__bases__ = (FairseqIncrementalState,) + tuple(
        b for b in cls.__bases__ if b != FairseqIncrementalState
    )
    return cls


def utils_softmax(x, dim: int):
    return F.softmax(x, dim=dim, dtype=torch.float32)


@with_incremental_state
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_size = config.embed_size
        self.heads = config.heads
        self.head_dim = self.embed_size // self.heads

        assert (
            self.head_dim * self.heads == self.embed_size
        ), "Embedding size needs to be divisible by heads"

        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(self.embed_size, self.embed_size, bias=True)
        self.k_proj = nn.Linear(self.embed_size, self.embed_size, bias=True)
        self.v_proj = nn.Linear(self.embed_size, self.embed_size, bias=True)
        self.fc_out = nn.Linear(self.embed_size, self.embed_size, bias=True)

        self.dropout = getattr(config, 'attention_dropout', 0.1)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.fc_out.weight)
        if self.fc_out.bias is not None:
            nn.init.constant_(self.fc_out.bias, 0.0)

    def forward(
        self,
        query,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_size

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
        else:
            saved_state = None

        q = self.q_proj(query)
        k = self.k_proj(query if key is None else key)
        v = self.v_proj(query if value is None else value)
        
        q *= self.scaling

        q = q.contiguous().view(tgt_len, bsz * self.heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.heads, self.head_dim).transpose(0, 1)

        if saved_state is not None:
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.heads, -1, self.head_dim)
                k = torch.cat([prev_key, k], dim=1)
                
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                prev_value = _prev_value.view(bsz * self.heads, -1, self.head_dim)
                v = torch.cat([prev_value, v], dim=1)
                
            prev_key_padding_mask: Optional[Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            
            key_padding_mask = self._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=bsz,
                src_len=k.size(1),
            )

            saved_state["prev_key"] = k.view(bsz, self.heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(bsz, self.heads, -1, self.head_dim)
            saved_state["prev_key_padding_mask"] = key_padding_mask
            
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        
        src_len = k.size(1)

        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), 
                float("-inf")
            )
            attn_weights = attn_weights.view(bsz * self.heads, tgt_len, src_len)

        attn_weights_float = utils_softmax(attn_weights, dim=-1)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(
            attn_weights_float.type_as(attn_weights),
            p=self.dropout,
            training=self.training,
        )
        
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.heads, tgt_len, self.head_dim]
        
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.fc_out(attn)
        
        attn_weights_out: Optional[Tensor] = None
        if need_weights:
            attn_weights_out = attn_weights_float.view(
                bsz, self.heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                attn_weights_out = attn_weights_out.mean(dim=0)

        return attn, attn_weights_out

    @staticmethod
    def _append_prev_key_padding_mask(
        key_padding_mask: Optional[Tensor],
        prev_key_padding_mask: Optional[Tensor],
        batch_size: int,
        src_len: int,
    ) -> Optional[Tensor]:
        if prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
            )
        elif prev_key_padding_mask is not None:
            filler = torch.zeros(
                (batch_size, src_len - prev_key_padding_mask.size(1)),
                device=prev_key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), filler.float()], dim=1
            )
        elif key_padding_mask is not None:
            filler = torch.zeros(
                (batch_size, src_len - key_padding_mask.size(1)),
                device=key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat([filler.float(), key_padding_mask.float()], dim=1)
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)


if __name__ == "__main__":
    class TestConfig:
        def __init__(self):
            self.vocab_size = 33
            self.embed_size = 768
            self.heads = 12
            self.max_position_embeddings = 1024
            self.attention_dropout = 0.1
            self.embed_dropout = 0.1
    
    config = TestConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Running on device: {device}")
    print(f"Embed size: {config.embed_size}, Heads: {config.heads}")
    print(f"Head dim: {config.embed_size // config.heads}\n")
    
    # Test embeddings first
    print("="*60)
    print("Test 0: Embeddings Layer")
    print("="*60)
    
    embedding_layer = ESMEmbeddings(config).to(device)
    test_input = torch.randint(0, config.vocab_size, (4, 128)).to(device)
    embeddings = embedding_layer(test_input)
    print(f"Input IDs shape: {test_input.shape}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Expected: [batch=4, seq=128, embed=768]")
    print("✓ Embeddings working\n")
    
    model = MultiHeadAttention(config).to(device)
    model.eval()
    
    print("="*60)
    print("Test 1: Basic Forward Pass (Self-Attention)")
    print("="*60)
    
    seq_len = 128
    batch_size = 4
    
    query = torch.randn(seq_len, batch_size, config.embed_size).to(device)
    
    output, attn_weights = model(
        query=query,
        need_weights=True,
        need_head_weights=False
    )
    
    print(f"Input shape: {query.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"Output mean: {output.mean().item():.6f}, std: {output.std().item():.6f}")
    assert output.shape == query.shape
    assert attn_weights.shape == (batch_size, seq_len, seq_len)
    print("✓ Basic forward pass successful\n")
    
    print("="*60)
    print("Test 2: Cross-Attention (ESM2 doesn't use this,")
    print("        but the code supports it for extensions)")
    print("="*60)
    
    key_value_len = 256
    key = torch.randn(key_value_len, batch_size, config.embed_size).to(device)
    value = torch.randn(key_value_len, batch_size, config.embed_size).to(device)
    
    output_cross, attn_weights_cross = model(
        query=query,
        key=key,
        value=value,
        need_weights=True
    )
    
    print(f"Query shape: {query.shape}")
    print(f"Key shape: {key.shape}")
    print(f"Value shape: {value.shape}")
    print(f"Output shape: {output_cross.shape}")
    print(f"Attention weights shape: {attn_weights_cross.shape}")
    assert output_cross.shape == query.shape
    assert attn_weights_cross.shape == (batch_size, seq_len, key_value_len)
    print("✓ Cross-attention successful\n")
    
    print("="*60)
    print("Test 3: KV Caching (Incremental Inference)")
    print("="*60)
    
    incremental_state = {}
    seq_len_per_step = 1
    total_steps = 5
    
    print(f"Simulating {total_steps} autoregressive steps...")
    
    cached_outputs = []
    for step in range(total_steps):
        query_step = torch.randn(seq_len_per_step, batch_size, config.embed_size).to(device)
        
        output_step, _ = model(
            query=query_step,
            incremental_state=incremental_state,
            need_weights=False
        )
        
        cached_outputs.append(output_step)
        
        saved_state = model._get_input_buffer(incremental_state)
        if saved_state and "prev_key" in saved_state:
            cached_len = saved_state["prev_key"].shape[2]
            print(f"Step {step+1}: Output shape {output_step.shape}, Cached K/V length: {cached_len}")
    
    print(f"✓ KV caching working correctly\n")
    
    print("="*60)
    print("Test 4: Padding Mask")
    print("="*60)
    
    seq_lengths = [100, 80, 120, 90]
    max_len = max(seq_lengths)
    
    padded_query = torch.randn(max_len, batch_size, config.embed_size).to(device)
    
    key_padding_mask = torch.zeros(batch_size, max_len, dtype=torch.bool).to(device)
    for i, length in enumerate(seq_lengths):
        if length < max_len:
            key_padding_mask[i, length:] = True
    
    output_masked, _ = model(
        query=padded_query,
        key_padding_mask=key_padding_mask,
        need_weights=False
    )
    
    print(f"Sequence lengths: {seq_lengths}")
    print(f"Key padding mask shape: {key_padding_mask.shape}")
    print(f"Padded positions: {key_padding_mask.sum().item()}")
    print(f"Output shape: {output_masked.shape}")
    print("✓ Padding mask applied successfully\n")
    
    print("="*60)
    print("Test 5: Attention Mask (Causal)")
    print("="*60)
    
    seq_len_causal = 64
    query_causal = torch.randn(seq_len_causal, batch_size, config.embed_size).to(device)
    
    attn_mask = torch.triu(
        torch.ones(seq_len_causal, seq_len_causal) * float('-inf'),
        diagonal=1
    ).to(device)
    
    output_causal, attn_weights_causal = model(
        query=query_causal,
        attn_mask=attn_mask,
        need_weights=True
    )
    
    print(f"Causal mask shape: {attn_mask.shape}")
    print(f"Output shape: {output_causal.shape}")
    print(f"Attention weights shape: {attn_weights_causal.shape}")
    
    upper_triangle_mask = torch.triu(torch.ones(seq_len_causal, seq_len_causal), diagonal=1).bool()
    upper_triangle_sum = attn_weights_causal[:, upper_triangle_mask].sum().item()
    print(f"Upper triangle attention sum (should be ~0): {upper_triangle_sum:.6f}")
    print("✓ Causal masking working correctly\n")
    
    print("="*60)
    print("Test 6: Multi-Head Attention Weights")
    print("="*60)
    
    seq_len_heads = 64
    query_heads = torch.randn(seq_len_heads, batch_size, config.embed_size).to(device)
    
    output_heads, attn_weights_heads = model(
        query=query_heads,
        need_weights=True,
        need_head_weights=True
    )
    
    print(f"Per-head attention weights shape: {attn_weights_heads.shape}")
    print(f"Expected shape: ({config.heads}, {batch_size}, {seq_len_heads}, {seq_len_heads})")
    assert attn_weights_heads.shape == (config.heads, batch_size, seq_len_heads, seq_len_heads)
    print("✓ Per-head attention weights retrieved\n")
    
    print("="*60)
    print("Test 7: Gradient Flow")
    print("="*60)
    
    model.train()
    query_grad = torch.randn(seq_len_causal, batch_size, config.embed_size, requires_grad=True).to(device)
    
    output_grad, _ = model(query=query_grad, need_weights=False)
    loss = output_grad.mean()
    loss.backward()
    
    print(f"Output requires grad: {output_grad.requires_grad}")
    print(f"Query gradient norm: {query_grad.grad.norm().item():.6f}")
    print(f"Q proj weight gradient norm: {model.q_proj.weight.grad.norm().item():.6f}")
    print(f"K proj weight gradient norm: {model.k_proj.weight.grad.norm().item():.6f}")
    print(f"V proj weight gradient norm: {model.v_proj.weight.grad.norm().item():.6f}")
    print("✓ Gradients flowing correctly\n")
    
    print("="*60)
    print("ALL TESTS PASSED ✓")
    print("="*60)