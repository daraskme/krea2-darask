from __future__ import annotations


class Sage2KreaAttnProcessor:
    """Per-model SageAttention 2 processor with exact padding compaction and GQA expansion."""

    def __init__(self):
        try:
            from sageattention import sageattn
        except ImportError as exc:
            raise RuntimeError("SageAttention 2 is not installed in this environment") from exc
        self._sageattn = sageattn
        self.fallback_count = 0

    def __call__(self, attn, hidden_states, attention_mask=None, image_rotary_emb=None):
        import torch
        from diffusers.models.embeddings import apply_rotary_emb

        query = attn.to_q(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))
        key = attn.to_k(hidden_states).unflatten(-1, (attn.num_kv_heads, attn.head_dim))
        value = attn.to_v(hidden_states).unflatten(-1, (attn.num_kv_heads, attn.head_dim))
        gate = attn.to_gate(hidden_states)
        query, key = attn.norm_q(query), attn.norm_k(key)
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
        groups = attn.num_heads // attn.num_kv_heads
        if groups > 1:
            key = key.repeat_interleave(groups, dim=2)
            value = value.repeat_interleave(groups, dim=2)

        if attention_mask is None:
            result = self._run(query, key, value)
        elif not (
            attention_mask.dtype == torch.bool
            and attention_mask.ndim == 4
            and attention_mask.shape[1:3] == (1, 1)
            and attention_mask.shape[-1] == key.shape[1]
        ):
            import torch.nn.functional as F

            self.fallback_count += 1
            result = F.scaled_dot_product_attention(
                query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
                attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
            ).transpose(1, 2)
        else:
            valid_keys = attention_mask[:, 0, 0, :].bool()
            rows = []
            for batch in range(query.shape[0]):
                valid = valid_keys[batch]
                if not valid.any():
                    import torch.nn.functional as F

                    self.fallback_count += 1
                    rows.append(F.scaled_dot_product_attention(
                        query[batch : batch + 1].transpose(1, 2),
                        key[batch : batch + 1].transpose(1, 2),
                        value[batch : batch + 1].transpose(1, 2),
                        attn_mask=attention_mask[batch : batch + 1], dropout_p=0.0,
                    ).transpose(1, 2))
                else:
                    rows.append(self._run(query[batch : batch + 1], key[batch : batch + 1, valid], value[batch : batch + 1, valid]))
            result = torch.cat(rows, dim=0)
        result = result.flatten(2, 3) * torch.sigmoid(gate)
        return attn.to_out[0](result)

    def _run(self, query, key, value):
        # Krea tensors are B,N,H,D; Sage HND expects B,H,N,D.
        q, k, v = query.transpose(1, 2).contiguous(), key.transpose(1, 2).contiguous(), value.transpose(1, 2).contiguous()
        try:
            result = self._sageattn(q, k, v, tensor_layout="HND", is_causal=False)
        except (RuntimeError, ValueError, NotImplementedError):
            import torch.nn.functional as F

            self.fallback_count += 1
            result = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        return result.transpose(1, 2)
