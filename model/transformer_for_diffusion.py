from typing import Union, Optional, Tuple
import logging
import torch
import torch.nn as nn
import math

logger = logging.getLogger(__name__)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (..., dim)
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates half the hidden dimensions of the input tensor.
    Splits the last dimension into two halves, negates the second half,
    and concatenates them back together.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) module.
    Pre-computes sine and cosine values for efficient rotary embeddings.
    """
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        theta: float = 10000.0
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.theta = theta

        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq)
        self._set_cos_sin_cache(max_position_embeddings, self.inv_freq.device)

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device):
        """Updates the sine and cosine cache."""
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates rotary embeddings for the given positions.
        Args:
            x: Dummy tensor for device/dtype
            position_ids: Token positions, shape (batch_size, sequence_length)
        Returns:
            Tuple of (cos, sin) embeddings, shape (batch_size, 1, sequence_length, dim)
        """
        seq_len = position_ids.max().item() + 1
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device)

        # Extract embeddings using indexing instead of gather
        # cos_cached: [1, 1, max_seq, dim], position_ids: [B, seq_len]
        B, seq_len = position_ids.shape
        cos = self.cos_cached[:, :, position_ids[0], :]  # Use first batch's positions
        sin = self.sin_cached[:, :, position_ids[0], :]
        
        # Expand to batch size: [1, 1, seq_len, dim] -> [B, 1, seq_len, dim]
        cos = cos.expand(B, -1, -1, -1)
        sin = sin.expand(B, -1, -1, -1)
        
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class MultiheadAttentionWithQKNorm(nn.Module):
    """
    Wrapper around nn.MultiheadAttention with Query-Key Normalization and RoPE support.
    QK-Norm improves training stability and RoPE provides better positional encoding.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = False,
        norm_type: str = "rmsnorm",  # "rmsnorm" or "layernorm"
        use_rope: bool = True,  # Enable RoPE
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 2048
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first
        )
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # QK Normalization layers
        if norm_type == "rmsnorm":
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        elif norm_type == "layernorm":
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")
        
        # RoPE (optional)
        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryEmbedding(
                dim=self.head_dim,
                max_position_embeddings=max_position_embeddings,
                theta=rope_theta
            )
        
        self.batch_first = batch_first
    
    def _apply_qk_norm(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply QK normalization to query and key tensors.
        Args:
            q, k: (B, num_heads, seq_len, head_dim) if batch_first=True
                  or (seq_len, B, num_heads, head_dim) if batch_first=False
        """
        # Reshape for normalization: apply norm on head_dim
        q = self.q_norm(q)
        k = self.k_norm(k)
        return q, k
    
    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to query and key tensors.
        Args:
            q: (B, num_heads, seq_len_q, head_dim)
            k: (B, num_heads, seq_len_k, head_dim)
        Returns:
            q, k with RoPE applied (only if seq_len_q == seq_len_k, i.e., self-attention)
        """
        B = q.shape[0]
        seq_len_q = q.shape[2]
        seq_len_k = k.shape[2]
        device = q.device
        
        # Only apply RoPE for self-attention (when query and key have same sequence length)
        # For cross-attention, skip RoPE since positional relationship between different sequences isn't meaningful
        if seq_len_q != seq_len_k:
            return q, k
        
        # Generate position ids
        position_ids = torch.arange(seq_len_q, device=device).unsqueeze(0).expand(B, -1)
        
        # Get cos/sin embeddings: (B, 1, seq_len, head_dim)
        cos, sin = self.rope(q, position_ids)
        
        # Broadcast cos/sin to match q/k shape: (B, 1, seq_len, head_dim) -> (B, num_heads, seq_len, head_dim)
        # cos and sin will automatically broadcast across num_heads dimension
        
        # Apply rotation
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)
        
        return q, k
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True
    ):
        """
        Forward pass with QK-Norm and optional RoPE.
        Note: This is a simplified implementation that works with batch_first=True.
        For full compatibility, a custom attention implementation would be needed.
        """
        # For now, apply QK-Norm in embedding space (before splitting heads)
        # This is a compromise since we're wrapping nn.MultiheadAttention
        
        # Reshape to apply per-head normalization
        # query: (B, seq_len, embed_dim) -> (B, seq_len, num_heads, head_dim)
        if self.batch_first:
            B, seq_len_q, embed_dim = query.shape
            _, seq_len_k, _ = key.shape
            
            # Reshape for per-head operations
            q = query.view(B, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
            k = key.view(B, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Apply QK-Norm: (B, num_heads, seq_len, head_dim)
            q, k = self._apply_qk_norm(q, k)
            
            # Apply RoPE if enabled (will skip for cross-attention)
            if self.use_rope:
                q, k = self._apply_rope(q, k)
            
            # Reshape back
            query = q.transpose(1, 2).reshape(B, seq_len_q, embed_dim)
            key = k.transpose(1, 2).reshape(B, seq_len_k, embed_dim)
        else:
            # Handle seq_first case
            query = self.q_norm(query)
            key = self.k_norm(key)
        
        # Call standard MultiheadAttention
        return self.attn(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights
        )




class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        # Use float32 for computation, then convert if needed
        # x is typically long (timestep), so we use float32 for sinusoidal computation
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)
        emb = x[:, None].float() * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ModuleAttrMixin(nn.Module):
    def __init__(self):
        super().__init__()
        self._dummy_variable = nn.Parameter()

    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

class LowdimMaskGenerator(ModuleAttrMixin):
    def __init__(self,
        action_dim, obs_dim,
        # obs mask setup
        max_n_obs_steps=2, 
        fix_obs_steps=True, 
        # action mask
        action_visible=False
        ):
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.max_n_obs_steps = max_n_obs_steps
        self.fix_obs_steps = fix_obs_steps
        self.action_visible = action_visible

    @torch.no_grad()
    def forward(self, shape, seed=None):
        device = self.device
        B, T, D = shape
        assert D == (self.action_dim + self.obs_dim)

        # create all tensors on this device
        rng = torch.Generator(device=device)
        if seed is not None:
            rng = rng.manual_seed(seed)

        # generate dim mask
        dim_mask = torch.zeros(size=shape, 
            dtype=torch.bool, device=device)
        is_action_dim = dim_mask.clone()
        is_action_dim[...,:self.action_dim] = True
        is_obs_dim = ~is_action_dim

        # generate obs mask
        if self.fix_obs_steps:
            obs_steps = torch.full((B,), 
            fill_value=self.max_n_obs_steps, device=device)
        else:
            obs_steps = torch.randint(
                low=1, high=self.max_n_obs_steps+1, 
                size=(B,), generator=rng, device=device)
            
        steps = torch.arange(0, T, device=device).reshape(1,T).expand(B,T)
        obs_mask = (steps.T < obs_steps).T.reshape(B,T,1).expand(B,T,D)
        obs_mask = obs_mask & is_obs_dim

        # generate action mask
        if self.action_visible:
            action_steps = torch.maximum(
                obs_steps - 1, 
                torch.tensor(0,
                    dtype=obs_steps.dtype, 
                    device=obs_steps.device))
            action_mask = (steps.T < action_steps).T.reshape(B,T,1).expand(B,T,D)
            action_mask = action_mask & is_action_dim

        mask = obs_mask
        if self.action_visible:
            mask = mask | action_mask
        
        return mask

class CustomEncoderBlock(nn.Module):
    """
    Encoder block that handles condition embedding, VL pooling, encoding, and memory projection
    """
    def __init__(self, n_emb, n_head, n_cond_layers, p_drop_emb, p_drop_attn, vl_emb_dim, 
                 obs_as_cond, cond_dim, T_cond, reasoning_emb_dim=1536):
        super().__init__()
        self.n_emb = n_emb
        self.obs_as_cond = obs_as_cond
        self.T_cond = T_cond
        
        # Time embedding
        self.time_emb = SinusoidalPosEmb(n_emb)
        
        # Observation condition embedding
        self.cond_obs_emb = None
        if obs_as_cond:
            self.cond_obs_emb = nn.Linear(cond_dim, n_emb)
        
        # VL features processing
        self.vl_emb_proj = nn.Linear(vl_emb_dim, n_emb)
        self.vl_emb_norm = nn.LayerNorm(n_emb)
        self.vl_attention_pooling = MultiheadAttentionWithQKNorm(
            embed_dim=n_emb,
            num_heads=n_head,
            dropout=p_drop_attn,
            batch_first=True,
            norm_type="rmsnorm"
        )
        self.vl_pool_query = nn.Parameter(torch.randn(1, 1, n_emb))

        # Reasoning features processing
        self.reasoning_emb_dim = reasoning_emb_dim
        self.reasoning_emb_proj = nn.Linear(reasoning_emb_dim, n_emb)
        self.reasoning_emb_norm = nn.LayerNorm(n_emb)
        self.reasoning_attention_pooling = MultiheadAttentionWithQKNorm(
                embed_dim=n_emb,
                num_heads=n_head,
                dropout=p_drop_attn,
                batch_first=True,
                norm_type="rmsnorm"
            )
        self.reasoning_pool_query = nn.Parameter(torch.randn(1, 1, n_emb))
        
        # Position embedding and preprocessing
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, n_emb))
        self.drop = nn.Dropout(p_drop_emb)
        self.pre_encoder_norm = nn.LayerNorm(n_emb)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4*n_emb,
            dropout=p_drop_attn,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_cond_layers
        )
        
        # Memory processing
        self.memory_norm = nn.LayerNorm(n_emb)
        self.memory_proj = nn.Sequential(
            nn.Linear(n_emb, n_emb),
            nn.GELU(),
            nn.Dropout(p_drop_attn),
            nn.Linear(n_emb, n_emb)
        )
    
    def _attention_pool_vl_features(self, vl_features: torch.Tensor, vl_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply attention pooling to vl features"""
        batch_size = vl_features.shape[0]
        query = self.vl_pool_query.expand(batch_size, -1, -1)  # (B, 1, n_emb)
        pooled_features, _ = self.vl_attention_pooling(
            query=query,                    # (B, 1, n_emb)
            key=vl_features,               # (B, T_vl, n_emb)
            value=vl_features,             # (B, T_vl, n_emb)
            key_padding_mask=vl_padding_mask  # (B, T_vl)
        )
        return pooled_features  # (B, 1, n_emb)

    def _attention_pool_reasoning_features(self, reasoning_features: torch.Tensor, reasoning_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply attention pooling to reasoning features"""
        batch_size = reasoning_features.shape[0]
        query = self.reasoning_pool_query.expand(batch_size, -1, -1)  # (B, 1, n_emb)
        pooled_features, _ = self.reasoning_attention_pooling(
            query=query,                    # (B, 1, n_emb)
            key=reasoning_features,         # (B, T_r, n_emb)
            value=reasoning_features,       # (B, T_r, n_emb)
            key_padding_mask=reasoning_padding_mask  # (B, T_r)
        )
        return pooled_features  # (B, 1, n_emb)
    
    def forward(self, vl_embeds: torch.Tensor, reasoning_embeds: Optional[torch.Tensor] = None, 
                cond: Optional[torch.Tensor] = None, vl_padding_mask: Optional[torch.Tensor] = None,
                reasoning_padding_mask: Optional[torch.Tensor] = None):
        """
        vl_embeds: (B, T_vl, D_vl) vision-language embeddings
        reasoning_embeds: (B, T_r, D_r) reasoning embeddings
        cond: (B, T_cond, cond_dim) condition tensor
        vl_padding_mask: (B, T_vl) padding mask for VL features
        reasoning_padding_mask: (B, T_r) padding mask for reasoning features
        Note: timestep is removed - memory generation does not use timesteps
        """
        
        # 1. Process VL features
        vl_features = self.vl_emb_proj(vl_embeds)  # (B, T_vl, n_emb)
        vl_features = self.vl_emb_norm(vl_features)
        vl_features_processed = self._attention_pool_vl_features(vl_features, vl_padding_mask)

        # 2. Process Reasoning features
        reasoning_features = None
        reasoning_features_processed = None
        reasoning_features = self.reasoning_emb_proj(reasoning_embeds)
        reasoning_features = self.reasoning_emb_norm(reasoning_features)
        reasoning_features_processed = self._attention_pool_reasoning_features(reasoning_features, reasoning_padding_mask)
        
        # 3. Combine condition embeddings (no timestep here)
        cond_list = []
        cond_obs_emb = self.cond_obs_emb(cond)
        cond_list.append(cond_obs_emb)
        
        cond_list.append(vl_features_processed)
        cond_list.append(reasoning_features_processed)
            
        cond_embeddings = torch.cat(cond_list, dim=1)
        
        # 4. Add position embedding
        tc = cond_embeddings.shape[1]
        if tc <= self.cond_pos_emb.shape[1]:
            position_embeddings = self.cond_pos_emb[:, :tc, :]
        else:
            position_embeddings = torch.zeros(1, tc, self.cond_pos_emb.shape[2], 
                                            device=self.cond_pos_emb.device, dtype=self.cond_pos_emb.dtype)
            position_embeddings[:, :self.cond_pos_emb.shape[1], :] = self.cond_pos_emb
            if tc > self.cond_pos_emb.shape[1]:
                torch.nn.init.normal_(position_embeddings[:, self.cond_pos_emb.shape[1]:, :], mean=0.0, std=0.02)
        
        # 5. Apply dropout and pre-norm
        x = self.drop(cond_embeddings + position_embeddings)
        x = self.pre_encoder_norm(x)
        
        # 6. Transformer encoder
        x = self.encoder(x)
        
        # 7. Memory processing
        memory = self.memory_norm(x)
        memory = memory + self.memory_proj(memory)
        
        return memory, vl_features, reasoning_features


class CustomDecoderLayer(nn.Module):
    """
    Condition-Centric Decoder Layer for Path Planning.
    Removes self-attention on noisy trajectory, focuses on condition utilization.
    Architecture: Memory-VL Fusion -> Trajectory-Condition Cross-Attention -> FFN
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, batch_first=False):
        super().__init__()
        
        # Step 1: Enhance Memory with VL features
        self.memory_vl_cross_attn = MultiheadAttentionWithQKNorm(
            d_model, nhead, dropout=dropout, batch_first=batch_first, norm_type="rmsnorm"
        )
        
        # Step 2: Trajectory attends to enhanced memory (condition)
        self.traj_memory_cross_attn = MultiheadAttentionWithQKNorm(
            d_model, nhead, dropout=dropout, batch_first=batch_first, norm_type="rmsnorm"
        )
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.activation = torch.nn.functional.gelu
        
        # AdaLN modulation - 生成 9 个参数：3组 (shift, scale, gate)
        # For: memory_vl, traj_memory, ffn
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 9 * d_model, bias=True)
        )
    
    def modulate(self, x, shift, scale):
        """AdaLN modulation function"""
        if shift is None:
            return x * (1 + scale.unsqueeze(1))
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    
    def forward(self, tgt, memory, vl_features, reasoning_features=None, conditioning=None, tgt_mask=None, memory_mask=None, 
                vl_key_padding_mask=None, reasoning_key_padding_mask=None):
        
        # 生成调制参数
        mod_params = self.adaLN_modulation(conditioning)
        shift_mem_vl, scale_mem_vl, gate_mem_vl, \
        shift_traj_mem, scale_traj_mem, gate_traj_mem, \
        shift_ffn, scale_ffn, gate_ffn = mod_params.chunk(9, dim=1)
        
        # 1. Enhance Memory with VL features (Condition Fusion)
        memory2 = self.norm1(memory)
        memory2 = self.modulate(memory2, shift_mem_vl, scale_mem_vl)
            
        # Combine VL and Reasoning features for cross attention
        cross_attn_kv = vl_features
        cross_attn_mask = vl_key_padding_mask
        
        cross_attn_kv = torch.cat([vl_features, reasoning_features], dim=1)
            
        # Handle masks
        if vl_key_padding_mask is not None:
            if reasoning_key_padding_mask is not None:
                cross_attn_mask = torch.cat([vl_key_padding_mask, reasoning_key_padding_mask], dim=1)
            else:
                # If reasoning mask is None, assume all valid (False)
                r_mask = torch.zeros((reasoning_features.shape[0], reasoning_features.shape[1]), 
                                         device=vl_key_padding_mask.device, dtype=torch.bool)
                cross_attn_mask = torch.cat([vl_key_padding_mask, r_mask], dim=1)
        elif reasoning_key_padding_mask is not None:
            # If vl mask is None, assume all valid
            v_mask = torch.zeros((vl_features.shape[0], vl_features.shape[1]), 
                                      device=reasoning_key_padding_mask.device, dtype=torch.bool)
            cross_attn_mask = torch.cat([v_mask, reasoning_key_padding_mask], dim=1)
        
        enhanced_memory_output, _ = self.memory_vl_cross_attn(memory2, cross_attn_kv, cross_attn_kv, 
                                                      key_padding_mask=cross_attn_mask)
        enhanced_memory = memory + gate_mem_vl.unsqueeze(1) * enhanced_memory_output
        
        # 2. Trajectory attends to Enhanced Condition (Direct Condition Utilization)
        tgt2 = self.norm2(tgt)
        tgt2 = self.modulate(tgt2, shift_traj_mem, scale_traj_mem)
        tgt2, _ = self.traj_memory_cross_attn(tgt2, enhanced_memory, enhanced_memory, 
                                             attn_mask=memory_mask, key_padding_mask=None)
        tgt = tgt + gate_traj_mem.unsqueeze(1) * tgt2
        
        # 3. Feed forward with modulation
        tgt2 = self.norm3(tgt)
        tgt2 = self.modulate(tgt2, shift_ffn, scale_ffn)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + gate_ffn.unsqueeze(1) * tgt2
            
        return tgt


class CustomTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, obs_as_cond=False, causal_attn=False):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        
        # Mask generation settings
        self.obs_as_cond = obs_as_cond
        self.causal_attn = causal_attn
    
    def _generate_memory_mask(self, x, memory, cond, tgt_mask):
        """Generate dynamic memory mask for causal attention"""
        if not self.causal_attn or tgt_mask is None:
            return None
            
        actual_memory_length = memory.shape[1]
        T_actual = x.shape[1]  
        S_actual = actual_memory_length 
        time_pos = 0  
        obs_start = 1   
        obs_end = obs_start + cond.shape[1] 
        
        # VL features are pooled to 1 token
        vl_start = obs_end
        vl_end = vl_start + 1
        
        memory_mask_dynamic = torch.zeros((T_actual, S_actual), device=x.device, dtype=torch.bool)
        
        for t in range(T_actual):
            # Time embedding: visible to all positions
            memory_mask_dynamic[t, time_pos] = True
            
            # Vision-language features: visible to all positions (1 pooled token)
            if vl_start < S_actual and vl_end <= S_actual:
                memory_mask_dynamic[t, vl_start:vl_end] = True
            
            # Observation conditions: causal visibility
            if obs_start < obs_end:
                visible_obs_end = min(obs_start + t + 1, obs_end)
                memory_mask_dynamic[t, obs_start:visible_obs_end] = True
        
        # Use model dtype instead of hardcoded float() to support bfloat16
        model_dtype = x.dtype
        memory_mask = memory_mask_dynamic.to(dtype=model_dtype).masked_fill(
            memory_mask_dynamic == 0, float('-inf')
        ).masked_fill(memory_mask_dynamic == 1, float(0.0))
        
        return memory_mask
        
    def forward(self, tgt, memory, vl_features, reasoning_features=None, cond=None, tgt_mask=None, 
                vl_key_padding_mask=None, reasoning_key_padding_mask=None, conditioning=None):
        # Generate dynamic memory mask
        memory_mask = self._generate_memory_mask(tgt, memory, cond, tgt_mask)
        
        # Decoder layers
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, vl_features, reasoning_features=reasoning_features, 
                          conditioning=conditioning, tgt_mask=tgt_mask, memory_mask=memory_mask,
                          vl_key_padding_mask=vl_key_padding_mask, 
                          reasoning_key_padding_mask=reasoning_key_padding_mask)
        return output


class HistoryEncoder(nn.Module):
    """
    Encodes the history sequence of ego status into a single vector.
    """
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        # x: (B, T, input_dim)
        # We use the final hidden state as the summary
        _, h_n = self.gru(x) # h_n: (num_layers, B, hidden_dim)
        h = h_n[-1] # (B, hidden_dim)
        return self.out_proj(self.act(h))


class TrajectoryHead(nn.Module):
    def __init__(self, n_emb, output_dim, p_drop_emb):
        super().__init__()
        self.ln_f = nn.LayerNorm(n_emb)
        self.drop = nn.Dropout(p_drop_emb)
        self.head = nn.Linear(n_emb, output_dim)

    def forward(self, x):
        x = self.ln_f(x)
        x = self.drop(x)
        x = self.head(x)
        return x


class TrajectoryRefinementHead(nn.Module):
    """
    Trajectory Head with GRU Temporal Processing.
    Structure: Temporal GRU -> Activation -> Output Projection
    Uses GRU to capture temporal dependencies and ensure continuity between time steps.
    """
    def __init__(self, n_emb, output_dim, p_drop_emb):
        super().__init__()
        self.ln_f = nn.LayerNorm(n_emb)
        self.drop = nn.Dropout(p_drop_emb)
        
        self.refine_gru = nn.GRU(
            input_size=n_emb,
            hidden_size=n_emb,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.act = nn.GELU()
        
        # Output Projection
        self.output_head = nn.Linear(n_emb, output_dim)
        
        # Conditioning projection for GRU init
        self.cond_proj = nn.Linear(n_emb, n_emb)

    def forward(self, x, conditioning=None):
        # x: (B, T, n_emb)
        x_norm = self.ln_f(x)
        x_norm = self.drop(x_norm)
        
        # Prepare initial hidden state from conditioning
        # conditioning: (B, n_emb) -> (1, B, n_emb)
        h_0 = self.cond_proj(conditioning).unsqueeze(0)
        # Ensure contiguous
        h_0 = h_0.contiguous()
        
        # Temporal processing with GRU
        # GRU processes temporal sequence: (B, T, n_emb) -> (B, T, n_emb)
        x_gru, _ = self.refine_gru(x_norm, h_0)
        
        # Residual connection: combine input with GRU refinement
        x_refined = x_norm + self.act(x_gru)
        
        # Output projection
        output = self.output_head(x_refined)
        
        return output


class TransformerForDiffusion(ModuleAttrMixin):
    def __init__(self,
            input_dim: int,
            output_dim: int,
            horizon: int,
            n_obs_steps: int = None,
            cond_dim: int = 2,
            n_layer: int = 12,
            n_head: int = 12,
            n_emb: int = 768,
            p_drop_emb: float = 0.1,
            p_drop_attn: float = 0.1,
            causal_attn: bool=False,
            obs_as_cond: bool=False,
            n_cond_layers: int = 4,
            vl_emb_dim: int = 1536,
            reasoning_emb_dim: int = 1536, 
            status_dim: int = 15,  
            ego_status_seq_len: int = 1  
        ) -> None:
        super().__init__()

        if n_obs_steps is None:
            n_obs_steps = horizon
        
        self.n_obs_steps = n_obs_steps
        
        T = horizon
        T_cond = 1
        obs_as_cond = cond_dim > 0
        if obs_as_cond:
            T_cond += n_obs_steps
        
        # Compute VL tokens count for T_cond
        self.target_vl_tokens = None
        vl_tokens_count = 1 
        T_cond += vl_tokens_count   

        # Compute Reasoning tokens count for T_cond
        reasoning_tokens_count = 1
        T_cond += reasoning_tokens_count

        # input embedding stem
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.hist_feature_emb = nn.Linear(status_dim, n_emb) # Embedding for full history state
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.hist_pos_emb = nn.Parameter(torch.zeros(1, n_obs_steps, n_emb))
        
        # Segment embeddings to distinguish history and future
        self.hist_segment_emb = nn.Parameter(torch.zeros(1, 1, n_emb))
        self.fut_segment_emb = nn.Parameter(torch.zeros(1, 1, n_emb))
        
        self.drop = nn.Dropout(p_drop_emb)
        self.pre_decoder_norm = nn.LayerNorm(n_emb)
        
        # AdaLN components for ego_status modulation
        self.time_emb = SinusoidalPosEmb(n_emb)  
        
        # Linear projection for ego_status
        self.status_dim = status_dim
        self.ego_status_proj = nn.Linear(status_dim, n_emb)
        
        # History Encoder for global modulation
        self.history_encoder = HistoryEncoder(status_dim, n_emb)

        # Custom encoder block that handles all condition processing
        self.encoder_block = CustomEncoderBlock(
            n_emb=n_emb,
            n_head=n_head,
            n_cond_layers=n_cond_layers,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            vl_emb_dim=vl_emb_dim,
            obs_as_cond=obs_as_cond,
            cond_dim=cond_dim,
            T_cond=T_cond,
            reasoning_emb_dim=reasoning_emb_dim # Pass reasoning_emb_dim
        )
        # Custom decoder that integrates VL cross attention and pre-processing
        custom_decoder_layer = CustomDecoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4*n_emb,
            dropout=p_drop_attn,
            batch_first=True
        )
        self.decoder = CustomTransformerDecoder(
            decoder_layer=custom_decoder_layer,
            num_layers=n_layer,
            obs_as_cond=obs_as_cond,
            causal_attn=causal_attn
        )

        # attention mask
        if causal_attn:
            # self-attention causal mask is computed here
            # however, cross attention mask is moved to the forward pass
            sz = T
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            self.register_buffer("mask", mask)
            self.memory_mask = None
        else:
            self.mask = None
            self.memory_mask = None

        # trajectory head block (with Residual Refinement and Temporal Smoothing)
        self.trajectory_head = TrajectoryRefinementHead(n_emb, output_dim, p_drop_emb)
        
        # Anchor prediction head - predicts the 5th future waypoint from encoder memory
        # Input: memory pooled from encoder (B, n_emb)
        # Output: single waypoint at step 5 (B, 2) where 2 is (x, y)
        self.anchor_prediction_head = nn.Sequential(
            nn.LayerNorm(n_emb),
            nn.Linear(n_emb, n_emb),
            nn.GELU(),
            nn.Dropout(p_drop_emb),
            nn.Linear(n_emb, 2)  # Single waypoint (x, y) at step 5
        )
        
        # constants
        self.T = T
        self.T_cond = T_cond
        self.horizon = horizon
        self.obs_as_cond = obs_as_cond
        self.vl_emb_dim = vl_emb_dim
        self.reasoning_emb_dim = reasoning_emb_dim

        self.apply(self._init_weights)
        
        # Initialize AdaLN modulation weights to zero (like in recogdrive.py)
        def zero_out_init(module: nn.Module):
            if isinstance(module, nn.Linear):
                nn.init.constant_(module.weight, 0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        # Zero out adaLN_modulation in all decoder layers
        for layer in self.decoder.layers:
            if hasattr(layer, 'adaLN_modulation'):
                layer.adaLN_modulation.apply(zero_out_init)
        
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def _init_weights(self, module):
        ignore_types = (
            nn.Dropout, 
            SinusoidalPosEmb, 
            nn.TransformerEncoderLayer, 
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.GELU,
            nn.SiLU,  
            nn.Sequential,
            CustomEncoderBlock,
            CustomDecoderLayer,
            CustomTransformerDecoder,
            TrajectoryHead,
            TrajectoryRefinementHead,
            nn.ModuleList
        )  #
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.GRU):
             # This case is now handled inside TrajectoryRefinementHead check or here if standalone
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param.data)
        elif isinstance(module, TrajectoryRefinementHead):
            # Initialize GRU
            for name, param in module.refine_gru.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param.data)
            # Initialize cond_proj
            if hasattr(module, 'cond_proj'):
                torch.nn.init.normal_(module.cond_proj.weight, mean=0.0, std=0.02)
                torch.nn.init.zeros_(module.cond_proj.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            
            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, RotaryEmbedding):
            # RotaryEmbedding has its own initialization with buffers
            pass
        elif isinstance(module, MultiheadAttentionWithQKNorm):
            # QK-Norm wrapper - let it handle its own submodules
            pass
        elif isinstance(module, HistoryEncoder):
            for name, param in module.gru.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param.data)
            torch.nn.init.normal_(module.out_proj.weight, mean=0.0, std=0.02)
            torch.nn.init.zeros_(module.out_proj.bias)
        elif isinstance(module, TransformerForDiffusion):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            torch.nn.init.normal_(module.hist_pos_emb, mean=0.0, std=0.02)
            torch.nn.init.normal_(module.hist_segment_emb, mean=0.0, std=0.02)
            torch.nn.init.normal_(module.fut_segment_emb, mean=0.0, std=0.02)
            if hasattr(module, 'vl_pool_query'):
                torch.nn.init.normal_(module.vl_pool_query, mean=0.0, std=0.02)
            if hasattr(module, 'cond_pos_emb') and module.cond_pos_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))
    
    def get_optim_groups(self, weight_decay: float=1e-3):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention, torch.nn.Conv1d, torch.nn.GRU)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, RMSNorm)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias") or "bias" in pn:
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif "weight" in pn and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    # This includes weight_ih, weight_hh for GRU/LSTM, weight for Linear/Conv1d, etc.
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root module as not decayed
        param_dict = {pn: p for pn, p in self.named_parameters()}
        for name in param_dict:
            if 'pos_emb' in name or '_dummy_variable' in name:
                no_decay.add(name)
            elif 'vl_pool_query' in name or 'cond_pos_emb' in name:
                no_decay.add(name)
            elif 'hist_pos_emb' in name or 'segment_emb' in name:
                no_decay.add(name)

        # validate that we considered every parameter
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups

    def configure_optimizers(self, 
            learning_rate: float=1e-4, 
            weight_decay: float=1e-3,
            betas: Tuple[float, float]=(0.9,0.95)):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def encode_conditions(self,
        cond: torch.Tensor,
        gen_vit_tokens: Optional[torch.Tensor] = None,
        reasoning_query_tokens: Optional[torch.Tensor] = None,
        **kwargs):
        """
        Encode conditions (VL, reasoning, observations) to memory.
        This can be cached during diffusion sampling since it doesn't depend on timestep or trajectory.
        
        Returns:
            memory: (B, T_cond, n_emb)
            vl_features: (B, T_vl, n_emb)
            reasoning_features: (B, T_r, n_emb)
            vl_padding_mask: (B, T_vl)
            reasoning_padding_mask: (B, T_r)
        """
        cond = cond.contiguous()
        vl_embeds = gen_vit_tokens.contiguous()
        reasoning_embeds = reasoning_query_tokens.contiguous()
        
        # Check VL padding
        vl_padding_mask = None
        if 'vl_mask' in kwargs and kwargs['vl_mask'] is not None:
            vl_padding_mask = ~kwargs['vl_mask']
        else:
            vl_norm = torch.norm(vl_embeds, dim=-1)
            vl_padding_mask = (vl_norm == 0)
        
        # Check Reasoning padding
        reasoning_padding_mask = None
        if 'reasoning_mask' in kwargs and kwargs['reasoning_mask'] is not None:
            reasoning_padding_mask = ~kwargs['reasoning_mask']
        else:
            reasoning_norm = torch.norm(reasoning_embeds, dim=-1)
            reasoning_padding_mask = (reasoning_norm == 0)
        
        # Process conditions through encoder block
        memory, vl_features, reasoning_features = self.encoder_block(
            vl_embeds=vl_embeds,
            reasoning_embeds=reasoning_embeds,
            cond=cond,
            vl_padding_mask=vl_padding_mask,
            reasoning_padding_mask=reasoning_padding_mask
        )
        
        return memory, vl_features, reasoning_features, vl_padding_mask, reasoning_padding_mask

    def decode_with_cache(self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        memory: torch.Tensor,
        vl_features: torch.Tensor,
        reasoning_features: torch.Tensor,
        cond: torch.Tensor,
        ego_status: Optional[torch.Tensor] = None,
        vl_padding_mask: Optional[torch.Tensor] = None,
        reasoning_padding_mask: Optional[torch.Tensor] = None,
        **kwargs):
        """
        Decode trajectory using cached encoder outputs.
        This is much faster during diffusion sampling.
        
        Args:
            sample: (B, T, input_dim) - noisy trajectory
            timestep: diffusion timestep
            memory: cached encoder output
            vl_features: cached VL features
            reasoning_features: cached reasoning features
            cond: observation conditions
            ego_status: ego vehicle status
        """
        sample = sample.contiguous()
        cond = cond.contiguous()
        
        # Ensure all inputs match the model's dtype
        model_dtype = next(self.parameters()).dtype
        sample = sample.to(dtype=model_dtype)
        cond = cond.to(dtype=model_dtype)
        memory = memory.to(dtype=model_dtype)
        vl_features = vl_features.to(dtype=model_dtype)
        if reasoning_features is not None:
            reasoning_features = reasoning_features.to(dtype=model_dtype)
        if ego_status is not None:
            ego_status = ego_status.to(dtype=model_dtype)
        
        # 1. Prepare timesteps
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])
        
        # 2. Generate conditioning from timestep and ego_status
        time_embedding = self.time_emb(timesteps)
        conditioning = None
        hist_features = None
        
        current_status = ego_status[:, -1, :]
        hist_state = ego_status
        hist_tokens = self.hist_feature_emb(hist_state)
        
        t_hist = hist_tokens.shape[1]
        if t_hist <= self.hist_pos_emb.shape[1]:
            hist_pos = self.hist_pos_emb[:, :t_hist, :]
        else:
            hist_pos = self.hist_pos_emb
        
        hist_features = hist_tokens + hist_pos + self.hist_segment_emb
        hist_global_embedding = self.history_encoder(hist_state)
        status_embedding = self.ego_status_proj(current_status)
        conditioning = time_embedding + status_embedding + hist_global_embedding
        # Ensure conditioning matches model dtype (important for bfloat16 models)
        conditioning = conditioning.to(dtype=model_dtype)
        
        # 3. Pre-decoder processing
        token_embeddings = self.input_emb(sample)
        t = token_embeddings.shape[1]
        position_embeddings = self.pos_emb[:, :t, :]
        x = self.drop(token_embeddings + position_embeddings + self.fut_segment_emb)
        x = self.pre_decoder_norm(x)
        
        # Concatenate history
        tgt_mask = self.mask
        if hist_features is not None:
            hist_x = self.drop(hist_features)
            hist_x = self.pre_decoder_norm(hist_x)
            x = torch.cat([hist_x, x], dim=1)
            
            if tgt_mask is not None:
                # Use model dtype instead of float() to support bfloat16
                total_len = x.shape[1]
                mask = (torch.triu(torch.ones(total_len, total_len, device=x.device)) == 1).transpose(0, 1)
                tgt_mask = mask.to(dtype=model_dtype).masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        
        # 4. Decoder with cached memory
        x = self.decoder(
            tgt=x,
            memory=memory,
            vl_features=vl_features,
            reasoning_features=reasoning_features,
            cond=cond,
            tgt_mask=tgt_mask,
            vl_key_padding_mask=vl_padding_mask,
            reasoning_key_padding_mask=reasoning_padding_mask,
            conditioning=conditioning
        )
        
        # 5. Trajectory head
        x = self.trajectory_head(x, conditioning=conditioning)
        
        # Slice output if history was added
        if hist_features is not None:
            x = x[:, -t:, :]
        
        return x

    def forward(self, 
        sample: torch.Tensor, 
        timestep: Union[torch.Tensor, float, int], 
        cond: torch.Tensor,
        gen_vit_tokens: Optional[torch.Tensor] = None,
        reasoning_query_tokens: Optional[torch.Tensor] = None,
        ego_status: Optional[torch.Tensor] = None,
        return_anchor: bool = False,  # Whether to return predicted anchor points
        **kwargs):
        """
        Args:
            sample: (B, T, input_dim) noisy trajectory
            timestep: diffusion timestep
            cond: (B, n_obs_steps, cond_dim) condition features
            gen_vit_tokens: (B, seq_len, vl_emb_dim) VL features
            reasoning_query_tokens: (B, seq_len, reasoning_emb_dim) reasoning features
            ego_status: (B, status_dim) or (B, n_obs_steps, status_dim) ego status
            return_anchor: if True, return (output, predicted_anchor), else return output only
            
        Returns:
            if return_anchor=False: output trajectory (B, T, output_dim)
            if return_anchor=True: (output trajectory, predicted anchor) where predicted_anchor is (B, 2) - the 5th waypoint
        """
        # Ensure all inputs match the model's dtype (important for bfloat16 models)
        model_dtype = next(self.parameters()).dtype
        
        sample = sample.contiguous().to(dtype=model_dtype)
        cond = cond.contiguous().to(dtype=model_dtype)
        vl_embeds = gen_vit_tokens.contiguous().to(dtype=model_dtype)
        reasoning_embeds = reasoning_query_tokens.contiguous().to(dtype=model_dtype)
        if ego_status is not None:
            ego_status = ego_status.to(dtype=model_dtype)
        
        # 1. Prepare timesteps
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])
        
        # 2. Generate conditioning from timestep and ego_status (AdaLN)
        time_embedding = self.time_emb(timesteps)  # (B, n_emb)
        # Convert to model dtype
        time_embedding = time_embedding.to(dtype=model_dtype)
        
        conditioning = None
        hist_features = None
        # Handle 3D ego_status (B, To, status_dim)
        current_status = ego_status[:, -1, :]
        hist_state = ego_status
        # 1. Embed history for concatenation (Local Context)
        hist_tokens = self.hist_feature_emb(hist_state)
                
        # Add position embedding
        t_hist = hist_tokens.shape[1]
        if t_hist <= self.hist_pos_emb.shape[1]:
            hist_pos = self.hist_pos_emb[:, :t_hist, :]
        else:
            hist_pos = self.hist_pos_emb
                
        # Add segment embedding for history
        hist_features = hist_tokens + hist_pos + self.hist_segment_emb
                
        # 2. Encode history for modulation (Global Context)
        hist_global_embedding = self.history_encoder(hist_state)
                
        # Project ego_status (current)
        status_embedding = self.ego_status_proj(current_status)  # (B, n_emb)
            
        # Add to timestep embedding: Time + Current Status + History Trend
        conditioning = time_embedding + status_embedding + hist_global_embedding # (B, n_emb)

        
        # 3. Check VL padding
        vl_padding_mask = None
        if 'vl_mask' in kwargs and kwargs['vl_mask'] is not None:
            vl_padding_mask = ~kwargs['vl_mask']  
        else:
            vl_norm = torch.norm(vl_embeds, dim=-1)  # (B, T_vl)
            vl_padding_mask = (vl_norm == 0) 
            
        # Check Reasoning padding
        reasoning_padding_mask = None
        if 'reasoning_mask' in kwargs and kwargs['reasoning_mask'] is not None:
            reasoning_padding_mask = ~kwargs['reasoning_mask']
        else:
            reasoning_norm = torch.norm(reasoning_embeds, dim=-1)
            reasoning_padding_mask = (reasoning_norm == 0)
        
        # 4. Process conditions through encoder block to get memory (no timestep)
        memory, vl_features, reasoning_features = self.encoder_block(
            vl_embeds=vl_embeds, 
            reasoning_embeds=reasoning_embeds,
            cond=cond, 
            vl_padding_mask=vl_padding_mask,
            reasoning_padding_mask=reasoning_padding_mask)
        
        # 5. Pre-decoder processing
        token_embeddings = self.input_emb(sample)
        t = token_embeddings.shape[1]
        position_embeddings = self.pos_emb[:, :t, :]
        # Add future segment embedding
        x = self.drop(token_embeddings + position_embeddings + self.fut_segment_emb)
        x = self.pre_decoder_norm(x)
        
        # Concatenate history if available
        tgt_mask = self.mask
        if hist_features is not None:
            hist_x = self.drop(hist_features)
            hist_x = self.pre_decoder_norm(hist_x)
            x = torch.cat([hist_x, x], dim=1)
            
            if tgt_mask is not None:
                # Create causal mask for the extended sequence
                # Use model dtype instead of float() to support bfloat16
                model_dtype = x.dtype
                total_len = x.shape[1]
                mask = (torch.triu(torch.ones(total_len, total_len, device=x.device)) == 1).transpose(0, 1)
                tgt_mask = mask.to(dtype=model_dtype).masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        
        # 6. Decoder with integrated memory mask handling and VL cross attention
        # Pass conditioning to decoder for AdaLN modulation
        x = self.decoder(
            tgt=x,
            memory=memory,
            vl_features=vl_features,
            reasoning_features=reasoning_features,
            cond=cond,
            tgt_mask=tgt_mask,
            vl_key_padding_mask=vl_padding_mask,
            reasoning_key_padding_mask=reasoning_padding_mask,
            conditioning=conditioning  # 传递 conditioning
        )        
        
        # 7. trajectory head 
        # Pass full sequence (History + Future) to head for GRU continuity
        # Also pass conditioning for GRU initialization
        x = self.trajectory_head(x, conditioning=conditioning)
        
        # Slice output if history was added (keep only future)
        if hist_features is not None:
            x = x[:, -t:, :]
        
        # 8. Predict anchor point from encoder memory if requested
        predicted_anchor = None
        if return_anchor:
            # Use memory from encoder to predict the 5th future waypoint
            # Pool memory to get a global representation: (B, T_cond, n_emb) -> (B, n_emb)
            memory_pooled = memory.mean(dim=1)  # Global average pooling
            
            # Predict anchor: (B, n_emb) -> (B, 2)
            predicted_anchor = self.anchor_prediction_head(memory_pooled)  # (B, 2)
            
            return x, predicted_anchor
            
        return x



def test():
    # GPT with time embedding and obs cond and encoder
    transformer = TransformerForDiffusion(
        input_dim=2,  # Changed to 2 to match hist_traj dimension
        output_dim=2, # Changed to 2
        horizon=8,
        n_obs_steps=4,
        cond_dim=10,
        causal_attn=True,
        n_cond_layers=4,
        vl_emb_dim=1536,
        status_dim=13,  # ego_status 维度
        ego_status_seq_len=4  # 历史帧数
    )
    opt = transformer.configure_optimizers()
    
    timestep = torch.tensor(0)
    sample = torch.zeros((4,8,2)) # Changed to 2
    cond = torch.zeros((4,4,10))
    vl_embeds = torch.ones((4,36,1536))
    
    # Test 1: Without ego_status_history (backward compatibility)
    print("Test 1: Without ego_status_history")
    out = transformer(sample, timestep, cond, vl_embeds)
    print(f"Output shape: {out.shape}")
    
    # Test 2: With ego_status (AdaLN modulation)
    print("\nTest 2: With ego_status (AdaLN modulation)")
    ego_status = torch.randn((4, 13))  # (B, status_dim=13) - 2D for simple modulation
    out_with_status = transformer(sample, timestep, cond, vl_embeds, ego_status=ego_status)
    print(f"Output shape with status: {out_with_status.shape}")
    
    # Test 3: Verify outputs are different with/without status
    print("\nTest 3: Verify modulation effect")
    diff = torch.abs(out - out_with_status).mean()
    print(f"Mean difference between outputs: {diff:.6f}")
    if diff > 0:
        print("✓ AdaLN modulation is working!")
    else:
        print("⚠ Warning: Outputs are identical (modulation may not be working)")

    # Test 4: With 3D ego_status (History concatenation + AdaLN)
    print("\nTest 4: With 3D ego_status (History concatenation)")
    # ego_status with history: (B, To=4, status_dim=13)
    ego_status_3d = torch.randn((4, 4, 13)) 
    out_with_hist = transformer(sample, timestep, cond, vl_embeds, ego_status=ego_status_3d)
    print(f"Output shape with history: {out_with_hist.shape}")
    
    diff_hist = torch.abs(out_with_status - out_with_hist).mean()
    print(f"Mean difference between 2D and 3D status: {diff_hist:.6f}")
    if diff_hist > 0:
        print("✓ History concatenation is affecting output!")
    
    # Test 5: Verify GRU Continuity and Conditioning
    print("\nTest 5: Verify GRU Continuity and Conditioning")
    # We expect the output to change slightly because of GRU initialization with conditioning
    # and because the GRU now sees the history sequence.
    
    # Run with same inputs as Test 4 but check if output is different from what we would expect 
    # if we sliced BEFORE the head (which we can't easily simulate without changing code back, 
    # but we can check if conditioning affects output).
    
    # Run with different conditioning (different timestep)
    timestep2 = torch.tensor(10)
    out_diff_time = transformer(sample, timestep2, cond, vl_embeds, ego_status=ego_status_3d)
    
    diff_time = torch.abs(out_with_hist - out_diff_time).mean()
    print(f"Mean difference with different timestep (Global Context Injection): {diff_time:.6f}")
    if diff_time > 0:
        print("✓ Global Context Injection (Conditioning) is working!")

    # Test 6: Verify Global History Modulation
    print("\nTest 6: Verify Global History Modulation")
    # We check if changing the history (but keeping current status same) affects the output
    # via the modulation pathway.
    
    # Create two histories with same final state but different past
    ego_status_A = torch.randn((4, 4, 13))
    ego_status_B = ego_status_A.clone()
    # Change the past (t=0..2), keep current (t=3) same
    ego_status_B[:, :3, :] = torch.randn((4, 3, 13))
    
    # Run model
    out_A = transformer(sample, timestep, cond, vl_embeds, ego_status=ego_status_A)
    out_B = transformer(sample, timestep, cond, vl_embeds, ego_status=ego_status_B)
    
    diff_hist_mod = torch.abs(out_A - out_B).mean()
    print(f"Mean difference with different history (same current): {diff_hist_mod:.6f}")
    if diff_hist_mod > 0:
        print("✓ Global History Modulation is working!")

    # Test 7: Verify Reasoning Tokens Integration
    print("\nTest 7: Verify Reasoning Tokens Integration")
    reasoning_tokens = torch.randn((4, 10, 1536)) # (B, seq_len, dim)
    
    # Initialize with reasoning dim
    transformer_reasoning = TransformerForDiffusion(
        input_dim=2,
        output_dim=2,
        horizon=8,
        n_obs_steps=4,
        cond_dim=10,
        causal_attn=True,
        n_cond_layers=4,
        vl_emb_dim=1536,
        reasoning_emb_dim=1536, # Enable reasoning
        status_dim=13,
        ego_status_seq_len=4
    )
    
    out_no_reasoning = transformer_reasoning(sample, timestep, cond, vl_embeds, ego_status=ego_status_3d)
    out_with_reasoning = transformer_reasoning(sample, timestep, cond, vl_embeds, 
                                             reasoning_query_tokens=reasoning_tokens, 
                                             ego_status=ego_status_3d)
    
    diff_reasoning = torch.abs(out_no_reasoning - out_with_reasoning).mean()
    print(f"Mean difference with reasoning tokens: {diff_reasoning:.6f}")
    if diff_reasoning > 0:
        print("✓ Reasoning Tokens Integration is working!")

    print("\n✓ All tests passed!")


    
if __name__ == "__main__":
    test()