import copy

import torch
import torch.nn.functional as F

from nats.chunk_utils import get_pad_size
import math

from nats.components.cache.cache_update import (
    update_valid_tokens,
    merge_local_to_global_caches,
    reduce_local_cache_with_cache_info,
    CacheUpdateInfo,
    extend_cache
)
from nats.components.cache.triton.update_one_step import post_update_one_step
from nats.components.masks.utils import generate_hard_seq_masks


class NAtSLayerCache:
    def __init__(self, bsz: int, n_msks: int, sliding_window_size: int,
                 layer_id: int = 0,
                 update_chunk_size: int = 16,
                 device: torch.device = torch.device('cuda')):
        """
        Cache for each layer from segment transformer. Currently, we have three types of tokens: the global tokens, the
        local tokens, and sliding window tokens (with fixed length). The size of sliding window tokens with is always
        fixed, therefore, we could consider them as part of global tokens and always put them in the beginning of the
        global tokens. We note that the head of global tokens only store the `past` sliding window tokens, so their
        size should be sliding_window - 1
        """
        # the sliding window only store the `past` sliding
        # sliding_window_size = sliding_window_size - 1
        self.layer_id = layer_id

        self.size_local_kv = torch.zeros(bsz, n_msks, device=device, dtype=torch.long)
        self.size_global_kv = torch.full([bsz, n_msks], fill_value=sliding_window_size, device=device, dtype=torch.long)
        # since we restrict the maximal local size, we store the local cache as a queue, this value indicates the
        # tail of the local cache.
        self.sliding_window_size = sliding_window_size
        # in principle, all the values in the same batch should share the same tails
        self.sliding_queue_tail = 0
        self.sliding_queue_tail_last = 0

        # this value is only used for  recoding the most recent sliding window info
        self._tokens_info = None
        self._cache_k_new = None
        self._cache_v_new = None
        self._n_global_min = torch.min(self.size_global_kv)
        self._sliding_window_tokens = None

        self.new_token_states = None

        self.n_ctx_local = torch.max(self.size_local_kv)
        # self.n_ctx_global = torch.max(self.size_global_kv)

        self.local_tokens = None

        self.n_ctx_data = torch.max(self.size_global_kv + self.size_local_kv)

        self._end_seqs_local = None
        self.valid_tokens = None

        self.cache_k = None
        self.cache_v = None

        self.bsz = bsz
        self.n_msks = n_msks

        self._seen_tokens = 0
        # This value is used to check how often we would like to do the update and post_update
        self.update_chunk_size = update_chunk_size
        self._token_states_info_in_chunk = None
        self.chunk_fill_size = 0
        self.chunk_fill_size_last = 0

        if sliding_window_size < update_chunk_size:
            raise NotImplementedError('The sliding window size must be greater or equal to the update chunk size!')

        self.device = device

    def split(self, split_size: list[int]) -> tuple['NAtSLayerCache']:
        """
        Split the cache into two seperated ones to reduce some computation overheads
        """
        new_caches = tuple(
            NAtSLayerCache(
                bsz=self.bsz,
                n_msks=n_msk,
                sliding_window_size=self.sliding_window_size,
                layer_id=self.layer_id, device=self.device
            ) for n_msk in split_size)

        size_local_kv_split = torch.split(self.size_local_kv, split_size, dim=1)
        size_global_kv_split = torch.split(self.size_global_kv, split_size, dim=1)
        valid_tokens_split = torch.split(self.valid_tokens, split_size, dim=1)
        cache_k_split = torch.split(self.cache_k, split_size, dim=1)
        cache_v_split = torch.split(self.cache_v, split_size, dim=1)

        for s_l_kv, s_g_kv, v_t_, cache_k, cache_v, new_cache in zip(
                size_local_kv_split, size_global_kv_split, valid_tokens_split, cache_k_split, cache_v_split, new_caches
        ):
            new_cache._seen_tokens = self._seen_tokens
            new_cache.sliding_queue_tail = self.sliding_queue_tail
            new_cache.size_local_kv = s_l_kv
            new_cache.size_global_kv = s_g_kv

            new_cache.n_ctx_local = s_l_kv
            new_cache._n_global_min = torch.min(s_g_kv)
            new_cache.n_ctx_data = torch.max(s_l_kv + s_g_kv)
            new_cache.valid_tokens = v_t_
            new_cache.cache_k = cache_k[:, :, :new_cache.n_ctx_data]
            new_cache.cache_v = cache_v[:, :, :new_cache.n_ctx_data]
        return new_caches

    def reset_cache(self, ):
        self.size_local_kv.zero_()
        self.size_global_kv.fill_(self.sliding_window_size)
        self.sliding_queue_tail = 0
        self.sliding_queue_tail_last = 0

        self.n_ctx_local = torch.max(self.size_local_kv)
        # self.n_ctx_global = torch.max(self.size_global_kv)
        self.new_token_states = None
        self.chunk_fill_size = 0
        self.chunk_fill_size_last = 0

        self.n_ctx_data = torch.max(self.size_global_kv + self.size_local_kv)
        self._tokens_info = None
        self._end_seqs_local = None
        self._n_global_min = torch.min(self.size_global_kv)
        self.valid_tokens = None

        self.cache_k = None
        self.cache_v = None

        self._cache_k_new = None
        self._cache_v_new = None
        self._sliding_window_tokens = None

        self._seen_tokens = 0

    def update_seqs_size_info(self, end_seqs_info: torch.Tensor):
        bsz, nheads, x_len = end_seqs_info.shape
        self.n_ctx_local = torch.max(self.size_local_kv) + x_len
        # self.n_ctx_global = torch.max(self.size_global_kv) + x_len

        # self._end_seqs_local = end_seqs_info

    def get_local_fill_idx(self, x_len: int):
        local_fill_idx = torch.arange(x_len, device=self.size_local_kv.device).view(1, 1, -1)
        local_fill_idx = self.size_local_kv.unsqueeze(-1) + local_fill_idx
        return local_fill_idx

    def generate_mask(self, x_len: int):
        mask = self._generate_mask(
            size_local_kv=self.size_local_kv,
            size_global_kv=self.size_global_kv,
            start_idx=self.chunk_fill_size - x_len,  # type: ignore
            x_len=x_len,
        )
        return mask

    def get_seq_length(self):
        return self._seen_tokens

    def update(self,
               key_states: torch.Tensor,
               value_states: torch.Tensor,
               token_states_info: torch.Tensor,
               ):
        """
        During update, we first consider the case where only one new token is provided. In this case, the new sliding
        tokens are added to the head of the global caches, and we can gradually remove the oldest tokens in the queue.
        However, if the number of tokens is greater than one, we first consider both local tokens and the sliding window
        tokens as local tokens to construct the attention maps and compute the attention output, we then move all the
        surviving sliding window tokens to the head of the global caches and do merging for local and sliding window
        tokens
        Args:
            key_states: torch.Tensor with shape [B, H, N_CTX, N_HEADS]
             new key states
            value_states: torch.Tensor with shape [B, H, N_CTX, N_HEADS]
             new value states
            token_states_info: torch.Tensor with shape [B, H, N_CTX, 3]
             token information, each element is a one hot encoded model. 0 indicates global tokens, 1 indicates local
             tokens, 2 indicates sliding window tokens.

        Returns:

        """
        bsz, nheads, x_len, head_dim = key_states.shape

        # cache_idx = torch.ones_like(key_states) * (
        #        torch.arange(key_states.shape[2]).view(1, 1, -1, 1).to(key_states) + 1 + self._seen_tokens)

        if self._token_states_info_in_chunk is None:
            self._token_states_info_in_chunk = torch.empty(
                [bsz, nheads, self.update_chunk_size, token_states_info.shape[-1]],
                device=token_states_info.device, dtype=token_states_info.dtype
            )
        if self.update_chunk_size == 1:
            self._cache_k_new = key_states[:, :, -self.sliding_window_size:]
            self._cache_v_new = value_states[:, :, -self.sliding_window_size:]
        if token_states_info.shape[2] == 0:
            return self.cache_k[:, :, :self.n_ctx_data], self.cache_v[:, :, :self.n_ctx_data]

        self.chunk_fill_size += x_len
        #"""
        if self.chunk_fill_size > self.update_chunk_size:
            # in this case, we need to create a larger token states info to cover all the past information
            # token info in chunk will be cleaned after wards anyway so we do not update them
            self._tokens_info = torch.cat(
                [self._token_states_info_in_chunk[:, :, :self.chunk_fill_size_last], token_states_info],
                dim=2
            )

        else:
            self._token_states_info_in_chunk[:, :, self.chunk_fill_size_last: self.chunk_fill_size] = token_states_info
            self._tokens_info = self._token_states_info_in_chunk[:, :, :self.chunk_fill_size]
            # self._tokens_info = self._token_states_info_in_chunk[:, :, :self.chunk_fill_size]
            # self._tokens_info[:, :, self.chunk_fill_size_last:] = token_states_info
        #"""
        #self._tokens_info = token_states_info

        # used for moving these values to the head of tokens in the end
        # """
        # self._tokens_info = token_states_info
        # if x_len > 1:
        if x_len > 1:
            self._sliding_window_tokens = self._tokens_info[..., 2] == 1
            self._end_seqs_local = self._tokens_info[..., 0].clone()

        # now we would like to move the end_seqs_local
        # bsz, nheads, x_len, head_dim = key_states.shape

        if self.cache_k is None:
            # self._end_seqs_local = token_states_info[..., 0].clone()
            sliding_window_tokens = torch.zeros(
                [*key_states.shape[:2], self.sliding_window_size, key_states.shape[-1]],
                device=key_states.device, dtype=key_states.dtype
            )
            pad_size = get_pad_size(x_len, self.update_chunk_size)
            pad_token = torch.zeros(
                [*key_states.shape[:2], pad_size, key_states.shape[-1]],
                device=key_states.device, dtype=key_states.dtype
            )
            self.cache_k = torch.cat([sliding_window_tokens, key_states, pad_token], dim=-2)
            self.cache_v = torch.cat([sliding_window_tokens, value_states, pad_token], dim=-2)

            # self.cache_idx = torch.cat([sliding_window_tokens, cache_idx, pad_token], dim=-2)

            self.valid_tokens = torch.zeros(
                [*key_states.shape[:2], self.sliding_window_size + x_len + pad_size],
                device=token_states_info.device, dtype=torch.bool,
            )
            self.n_ctx_local = key_states.shape[2]
            # self.valid_tokens = F.pad(self.valid_tokens, (0, x_len), value=1)
            self.valid_tokens[:, :, self.sliding_window_size: self.sliding_window_size + x_len] = True
            return key_states, value_states
        # Otherwise, we attach key values to self.cache_key
        self.update_seqs_size_info(token_states_info[..., 0])

        # in this case, we do a quick forward pass

        fill_start = self.n_ctx_data
        cache_size = self.cache_k.shape[2]
        score_size = fill_start + x_len
        n_next_chunk = math.ceil(score_size / self.update_chunk_size) * self.update_chunk_size
        pad_size = n_next_chunk - cache_size

        if pad_size > 0:
            self.cache_k = F.pad(self.cache_k, (0, 0, 0, pad_size))
            self.cache_v = F.pad(self.cache_v, (0, 0, 0, pad_size))

            self.valid_tokens = F.pad(self.valid_tokens, (0, pad_size), value=0)

        self.cache_k[:, :, self.n_ctx_data:score_size] = key_states.to(self.cache_k)
        self.cache_v[:, :, self.n_ctx_data:score_size] = value_states.to(self.cache_v)


        self.valid_tokens[:, :, self.n_ctx_data:score_size] = 1

        return self.cache_k[:, :, :score_size], self.cache_v[:, :, :score_size]

    def post_update(self, x_len: int):
        # with torch.cuda.stream(mem_stream):
        self._post_update(x_len)

    def _post_update(self, x_len):
        self._seen_tokens = self._seen_tokens + x_len

        if x_len == 0:
            return

        self.n_ctx_data += x_len

        if self.chunk_fill_size < self.update_chunk_size:
            # we need to mask out the past sliding window tokens
            if self._seen_tokens < self.update_chunk_size:
                fill_indices = self.chunk_fill_size
            else:
                fill_indices = (self.chunk_fill_size + self.sliding_queue_tail + 1) % self.sliding_window_size
            self.valid_tokens[:, :, fill_indices] = False

            self.chunk_fill_size_last = self.chunk_fill_size

            return

        # """

        # return

        if self._tokens_info[..., 2].all():
            self.move_sliding_window_tokens_to_head(self.chunk_fill_size, )

            # In this case, all the tokens are updated to the sw tokens, we simply remove all the past token infos
            self.chunk_fill_size = 0
            self.chunk_fill_size_last = 0
            self._tokens_info = None
            self._cache_k_new = None
            self._cache_v_new = None
            self._sliding_window_tokens = None
            self._end_seqs_local = None
            return
        if x_len == 1 and self.update_chunk_size == 1:
            new_k = self._cache_k_new
            new_v = self._cache_v_new

            is_global_tokens = (self._tokens_info[..., 0, 0] == 1.)
            # is_global_tokens = (self._tokens_info[..., 0, 0] == 1.).to(self.size_global_kv)

            new_idx = torch.where(
                is_global_tokens, self.size_global_kv, self.size_global_kv + self.size_local_kv
            ).unsqueeze(-1)

            if self._seen_tokens == x_len:
                fill_indices = self.chunk_fill_size
                self.sliding_queue_tail = 0
            else:
                fill_indices = (self.chunk_fill_size + self.sliding_queue_tail + 1) % self.sliding_window_size
                self.sliding_queue_tail = (self.sliding_queue_tail + 1) % self.sliding_window_size

            self.valid_tokens[:, :, [fill_indices]] = (self._tokens_info[..., 2] == 1).to(self.valid_tokens)


            new_idx = torch.cat([new_idx, torch.full_like(new_idx, fill_indices)], dim=-1)

            new_idx = self.expand_idx_for_cache_values(new_idx, head_dim=new_k.shape[-1])

            self.cache_k.scatter_(2, new_idx, new_k.expand(-1,-1,2,-1))
            self.cache_v.scatter_(2, new_idx, new_v.expand(-1,-1,2,-1))

            size_update_local = torch.where(is_global_tokens, self.size_local_kv + 1, 0)
            size_update_local = size_update_local + self._tokens_info[
                ..., -1, 2].long()  # for sliding window tokens, this value is 1

            # size_update_global = torch.where(is_global_tokens, 1, 0)
            size_update_global = is_global_tokens
            self.size_global_kv += size_update_global
            self.size_local_kv -= size_update_local

        else:
            self.move_sliding_window_tokens_to_head(self.chunk_fill_size, )

            updated_cache_info = self.get_updated_cache_info(
                tokens_info=self._tokens_info,
                x_len=self.chunk_fill_size,
                size_local=self.size_local_kv,
                size_global=self.size_global_kv
            )

            self.cache_k, self.cache_v, = merge_local_to_global_caches(
                [self.cache_k, self.cache_v, ],
                updated_cache_info,
            )
            size_update_global = 0
            if updated_cache_info is not None:
                size_update_global = updated_cache_info.size_update_global

                self.size_global_kv += size_update_global
                self.size_local_kv -= updated_cache_info.size_update_local
        self.size_local_kv += self.chunk_fill_size
        self.n_ctx_data = torch.max(self.size_global_kv + self.size_local_kv)

        self.valid_tokens = self.mask_invalid_tokens(
            self.valid_tokens, self.n_ctx_data,
            self.size_global_kv,
            self.size_local_kv,
            size_update_global
        )
        self._n_global_min = torch.min(self.size_global_kv)

        if self._seen_tokens == x_len:
            # this helps to reduce the unnecessary memory within pre-filling
            self.cache_k = self.cache_k[:, :, :self.n_ctx_data]
            self.cache_v = self.cache_v[:, :, :self.n_ctx_data]
            self.valid_tokens = self.valid_tokens[:, :, :self.n_ctx_data]

            torch.cuda.empty_cache()

        self.chunk_fill_size = 0
        self.chunk_fill_size_last = 0
        self._tokens_info = None
        self._cache_k_new = None
        self._cache_v_new = None
        self._sliding_window_tokens = None
        self._end_seqs_local = None

        # size_cache = torch.max(self.size_global_kv + self.size_local_kv)
        # self.cache_k = self.cache_k[:,:,:size_cache]
        # self.cache_v = self.cache_v[:,:,:size_cache]

    def move_sliding_window_tokens_to_head(self, x_len: int):
        # we first move all the sliding windows related tokens towards the head of the global tokens:
        """
        if x_len == 1:
            self.sliding_queue_tail += 1
            if self.sliding_queue_tail == self.sliding_window_size:
                self.sliding_queue_tail = 0
            self.valid_tokens[:, :, [self.sliding_queue_tail]] = (self._tokens_info[..., 2] == 1).to(self.valid_tokens)
            return
            self.cache_k[:, :, [self.sliding_queue_tail]] = self._cache_k_new
            self.cache_v[:, :, [self.sliding_queue_tail]] = self._cache_v_new
            return
        #"""
        x_sliding_window_tokens = min(x_len, self.sliding_window_size)

        if x_len == self._seen_tokens:
            fill_indices = torch.arange(x_sliding_window_tokens, device=self.device)
            self.sliding_queue_tail = x_sliding_window_tokens - 1
        else:
            fill_indices = torch.arange(x_sliding_window_tokens, device=self.device) + self.sliding_queue_tail + 1
            fill_indices = fill_indices % self.sliding_window_size
            self.sliding_queue_tail = (self.sliding_queue_tail + x_sliding_window_tokens) % self.sliding_window_size
            # in this case, the tail does not change

        self.valid_tokens[:, :, fill_indices] = (self._tokens_info[..., -x_sliding_window_tokens:, 2] == 1).to(
            self.valid_tokens)

        self.cache_k[:, :, fill_indices] = self.cache_k[:, :,
                                           self.n_ctx_data - x_sliding_window_tokens: self.n_ctx_data]
        self.cache_v[:, :, fill_indices] = self.cache_v[:, :,
                                           self.n_ctx_data - x_sliding_window_tokens: self.n_ctx_data]

        return

    def expand_idx_for_cache_values(self, idx_tensor: torch.Tensor, head_dim: int, ):
        idx_tensor = idx_tensor.unsqueeze(-1)
        idx_tensor = idx_tensor.expand(-1, -1, -1, head_dim)
        return idx_tensor

    def _generate_mask(self,
                       size_local_kv: torch.Tensor,
                       size_global_kv: torch.Tensor,
                       start_idx: int,
                       x_len: int,
                       ):
        """
        Generate masks based on the end_seqs_local and valid_tokens information
        """
        if x_len == 1:
            return self._generate_mask_one_step(size_local_kv + size_global_kv)

        msk_new_tokens, idx_sub_seq = self._generate_local_mask(size_local_kv=size_local_kv.unsqueeze(-1),
                                                                start_idx=start_idx,
                                                                x_len=self.chunk_fill_size)
        if self._seen_tokens == 0:
            return msk_new_tokens

        if self.valid_tokens.shape[-1] == 0:
            return msk_new_tokens
        # masks for tokens in the current chunks and new tokens are generated through funciton above
        n_data_ctx_last_update = self.n_ctx_data - self.chunk_fill_size_last
        # now we want to construct mask for cache values. This is composed of two parts, we first fit the sliding
        # window tokens
        msk_cache = self.valid_tokens[:, :, : n_data_ctx_last_update].unsqueeze(-2)

        msk_cache = msk_cache.repeat(1, 1, x_len, 1)
        msk_cache = self._fit_sliding_queue_masks(msk_cache, x_len, start_idx=start_idx, )

        # we now fit the local tokens. These local tokens might only survive until the first global token appear.
        # Assuming that the first global token appears at step 3 and we have x_len=5, then only 3 out of 5 tokens are
        # valid tokens
        msk_local = torch.arange(x_len, device=idx_sub_seq.device, dtype=idx_sub_seq.dtype).view(1, 1, -1)
        msk_local = (msk_local <= idx_sub_seq[:, :, [0]]).unsqueeze(-1)  # [bsz, nheads, x_len, 1]

        is_local_tokens = torch.arange(n_data_ctx_last_update - self._n_global_min, device=msk_local.device).view(1, 1,
                                                                                                                  1, -1)

        local_tokens_upper = (self.size_local_kv + self.size_global_kv - self._n_global_min)[..., None, None]
        local_token_lower = (self.size_global_kv - self._n_global_min)[..., None, None]

        # is_local_tokens = (local_token_lower <= is_local_tokens < local_tokens_upper).unsqueeze(-2)
        msk_local = torch.where(local_token_lower <= is_local_tokens, msk_local, True)
        msk_local = torch.where(is_local_tokens < local_tokens_upper, msk_local, False)

        # This is only applied to local masks
        msk_cache[:, :, :, self._n_global_min:n_data_ctx_last_update] = msk_local.to(msk_cache)
        return torch.cat([msk_cache, msk_new_tokens], dim=-1)

    def _generate_local_mask(self, size_local_kv: torch.Tensor, start_idx: int, x_len: int = 0) -> tuple[
        torch.Tensor, torch.Tensor]:
        # end_seqs_local = F.pad(self._end_seqs_local, (n_ctx_local, 0), mode='constant')

        # msk = generate_hard_seq_masks(end_seqs_local, start_idx=n_ctx_local)
        msk, idx_sub_seq = generate_hard_seq_masks(end_seqs_hard=self._end_seqs_local, start_idx=start_idx)

        # we now fill sliding window masks
        if self._sliding_window_tokens.any():
            n_ranges = torch.arange(x_len, device=self.device)
            sliding_window = n_ranges[start_idx:].unsqueeze(1) - n_ranges.unsqueeze(0)
            sliding_window_msk = (sliding_window <= self.sliding_window_size)

            msk[:, :, :, -x_len:] = torch.where(
                self._sliding_window_tokens.unsqueeze(-2),
                sliding_window_msk.view(1, 1, x_len - start_idx, x_len),
                msk[:, :, :, -x_len:]
            )

        msk = msk.tril_(diagonal=start_idx)

        # seq_mask_ = msk
        # msk = torch.where(seq_mask_, 0.0, -torch.inf)
        return msk, idx_sub_seq

    def _generate_mask_one_step(self,
                                size_cache: torch.Tensor):
        """
        if we only need to generate mask for one single step, since we already know that all the cached values will
        contribute to the new output generation, we could directly check the size of the valid tokens
        """
        if self._seen_tokens == 0:
            return torch.ones([1], device=size_cache.device, dtype=torch.bool).view(1, 1, 1, 1)
        # mask = F.pad(self.valid_tokens, (0, 1)).unsqueeze(-2)
        mask = self.valid_tokens[:, :, :self.n_ctx_data + 1].unsqueeze(-2)
        return mask

    @torch.compile
    def mask_invalid_tokens(self, valid_tokens: torch.Tensor, n_ctx_data: int, size_global_kv: torch.Tensor,
                            size_local_kv: torch.Tensor, size_update_local: torch.Tensor | int):
        size_global_min = self._n_global_min
        if n_ctx_data == size_global_min:
            return valid_tokens

        involved_idx_range = (torch.arange(n_ctx_data - size_global_min, device=valid_tokens.device)).view(1, 1, -1)
        n_last = (size_global_kv + size_local_kv - size_global_min).unsqueeze(-1)
        fill_value = involved_idx_range < n_last
        valid_tokens[:, :, size_global_min:n_ctx_data] = fill_value

        return valid_tokens

    @torch.compile
    def mask_invalid_tokens_compile(self, valid_tokens: torch.Tensor, n_ctx_data: int, size_global_kv: torch.Tensor,
                            size_local_kv: torch.Tensor, size_update_local: torch.Tensor | int):
        size_data = size_global_kv + size_local_kv
        for b in range(valid_tokens.shape[0]):
            for h in range(valid_tokens.shape[1]):
                valid_tokens[b,h, size_data[b,h]:] = 0
        return valid_tokens


        size_global_min = self._n_global_min
        if n_ctx_data == size_global_min:
            return valid_tokens

        involved_idx_range = (torch.arange(n_ctx_data - size_global_min, device=valid_tokens.device)).view(1, 1, -1)
        n_last = (size_global_kv + size_local_kv - size_global_min).unsqueeze(-1)
        fill_value = involved_idx_range < n_last
        valid_tokens[:, :, size_global_min:n_ctx_data] = fill_value

        return valid_tokens

    def _fit_sliding_queue_masks(self,
                                 mask_global: torch.Tensor,
                                 x_len: int,
                                 start_idx: int = 0,
                                 ):
        # we also need to check the maks for sliding windows, in this case, only the first self.sliding_window_size
        # have the valid masks
        # in this case, the sliding window queue is still not full: the head of the queue is 0
        x_len_range = torch.arange(x_len, device=self.device).unsqueeze(1)
        if self._seen_tokens < self.sliding_window_size:
            queue_range = torch.arange(self._seen_tokens, device=self.device).unsqueeze(0)
            mask_global[..., :self._seen_tokens] = ((x_len_range + queue_range) < self.sliding_window_size).view(1, 1,
                                                                                                                 x_len,
                                                                                                                 self._seen_tokens)
        else:
            # in other case, the queue is full, the head of the queue is (tail+1) % sliding_window_size
            queue_range = torch.arange(self.sliding_window_size, device=self.device)
            queue_range = (
                                  queue_range + self.sliding_window_size - 1 - self.sliding_queue_tail) % self.sliding_window_size
            if x_len <= self.sliding_window_size:
                msk_global_update = torch.ones(
                    (x_len, self.sliding_window_size), device=mask_global.device, dtype=mask_global.dtype
                ).triu(0)
                mask_global_update = msk_global_update[:, queue_range]
            else:
                msk_global_update = torch.ones(
                    (self.sliding_window_size, self.sliding_window_size), device=mask_global.device,
                    dtype=mask_global.dtype
                ).triu(0)
                msk_global_update = msk_global_update[:, queue_range]
                mask_global_update = F.pad(msk_global_update, (0, 0, 0, x_len - self.sliding_window_size), )
            # for each new tokens, we need to remove the first corresponding values in the token
            mask_global[..., :self.sliding_window_size] &= mask_global_update.view(1, 1, x_len,
                                                                                   self.sliding_window_size)

        return mask_global

    def get_updated_cache_info(self,
                               tokens_info: torch.Tensor,
                               size_local: torch.Tensor,
                               size_global: torch.Tensor,
                               x_len: int,
                               ) -> CacheUpdateInfo | None:
        """
        This function is provided to check if any information from the new sequence (including the cached local
        sequence and the new sequence) can be compressed into the global sequence.
        For the others, we simply put the corresponding values to the tail of each sequences

        Given that every item from end_seqs_local must contain at least one sequence with only 0. we could easily check
        which item in the batch will be updated by comparing the concatenated new sequence and the old local sequence.
        Once we have found which sample in the batch needs to be updated, we could extract the tokens that should be
        placed under the global caches.


        Args:
            tokens_info (torch.Tensor) torch Tensor with size (bsz, nh, x_len,n_states), the states of the new tokens
            size_local (torch.Tensor) torch Tensor with size (bsz, nh), size of the existing local tensors, this
                value is applied to compute the required reduced size for local kv models
            x_len (torch.Tensor): length of the new input sequences

        Returns:
            updated_cache_info (dict[int, torch.Tensor]): an updated cache information. This item is used to show
                which token in the local and new sequences need to be moved to the global sequence.

        """
        # the existing local end_seqs must contain at least one sequence that only contain 0
        # Hence, if the new end_seqs need to be merged to the global cache, then none of its subsequence should contain
        # a pure 0 sequence
        if tokens_info[..., 1].all():
            # all the tokens are local tokens, there is no need to do the update
            return None
        if tokens_info[..., 2].all():
            # all the tokens are sliding window tokens, we only need to take care of the sliding window tokens
            return CacheUpdateInfo(b_idx=[],
                                   h_idx=[],
                                   update_idx=[],
                                   shift_idx=[],
                                   seq_len=x_len,
                                   valid_value_mask=None,
                                   size_update_global=torch.zeros_like(size_global),
                                   size_update_local=torch.ones_like(size_global) * x_len,
                                   global_token_masks=None,
                                   )

        if x_len == 1:
            return self.get_updated_cache_info_one_step(tokens_info, size_local, size_global)
        # TODO rewrite this with triton???
        global_tokens = tokens_info[..., 0]
        bsz, n_heads, seq_len = global_tokens.shape

        # now we need to determine the end tokens that we would like to put into the global tokens. Since the remaining
        # local tokens must contain a sequence with only 0. Hence, we check the tokens that result in a shortest seq
        # that only contains 0. i.e., the position of the token where the last True in that sequence appears
        # value_ranges = torch.where(end_seqs_local[idx2update], torch.arange(end_seqs_local.shape[-2]).unsqueeze(-1), 0.0
        # ).max(1)[0].min(-1)[0].long()

        # the number of tokens added to the global tokens
        size_update_global = global_tokens.long().sum(-1)

        global_token_indices = torch.where(
            global_tokens > 0.,
            torch.arange(seq_len, device=global_tokens.device).view(1, 1, -1),
            -1
        )
        last_global_idx: torch.Tensor = global_token_indices.max(-1)[0]

        msk_removed_tokens = torch.arange(seq_len, device=tokens_info.device).view(1, 1, -1)
        msk_removed_tokens = msk_removed_tokens > last_global_idx.unsqueeze(-1)

        size_update_local = torch.where(last_global_idx > -1, last_global_idx + size_local, 0).long()

        remaining_local_tokens = torch.where(msk_removed_tokens, tokens_info[..., 1], 0).long()

        n_sliding_window_tokens = seq_len - F.relu(last_global_idx) - remaining_local_tokens.sum(-1)
        # sliding window tokens will be removed anyway
        size_update_local += n_sliding_window_tokens

        # these are the tokens that need to be preserved in the cache
        remaining_tokens = global_tokens + remaining_local_tokens

        non_zero_idx = torch.nonzero(remaining_tokens)
        # provide indexing information for each element
        non_zero_seqs = non_zero_idx[:, 0] * n_heads + non_zero_idx[:, 1]
        # since non_zero_seqs is already sorted, we could check the number of each elements
        counts_each_seq = torch.unique_consecutive(non_zero_seqs, return_counts=True)[1]

        max_update_size = counts_each_seq.max()
        # the following codes is equivalent to torch.cat([torch.arange(x) for x in max_update_size])
        # TODO check which function is faster!
        max_update_size_ranges = torch.arange(max_update_size, device=counts_each_seq.device)
        mask = (max_update_size_ranges.unsqueeze(0) < counts_each_seq.unsqueeze(1))

        shift_idx = max_update_size_ranges.repeat(len(counts_each_seq))[mask.flatten()]

        sizes_all = size_global + size_local
        new_model_base = torch.where(last_global_idx == -1, sizes_all, size_global)

        b_idx = non_zero_idx[:, 0]
        h_idx = non_zero_idx[:, 1]

        # we also need to know which tokens are global tokens and which not
        global_token_masks = non_zero_idx[:, 2] <= last_global_idx[b_idx, h_idx]

        shift_idx += new_model_base[b_idx, h_idx]
        # update_idx = non_zero_idx[:, 2] + sizes_all[b_idx, h_idx]
        update_idx = non_zero_idx[:, 2] + self.n_ctx_data - self.chunk_fill_size

        return CacheUpdateInfo(b_idx=b_idx,
                               h_idx=h_idx,
                               update_idx=update_idx,
                               shift_idx=shift_idx,
                               seq_len=seq_len,
                               valid_value_mask=None,
                               size_update_global=size_update_global,
                               size_update_local=size_update_local,
                               global_token_masks=global_token_masks,
                               )

    def get_updated_cache_info_one_step(self,
                                        tokens_info: torch.Tensor,
                                        size_local: torch.Tensor,
                                        size_global: torch.Tensor,
                                        ) -> CacheUpdateInfo | None:
        """
        If only one new value is updated each time, we can ensure that each sequence in end_seqs_local only contains one
        True value, therefore, we only need to check if end_seqs_local contains a True value and collect those
        """
        seq_len = 1

        b_idx, h_idx, update_idx = torch.nonzero(tokens_info[..., 0], as_tuple=True)
        size_update_global = torch.zeros_like(size_local)
        size_update_global[b_idx, h_idx] = 1

        size_update_local = torch.zeros_like(size_local)

        size_local_involved = size_local[b_idx, h_idx]
        size_global_involved = size_global[b_idx, h_idx]

        size_update_local[b_idx, h_idx] = update_idx + 1 + size_local_involved
        # we also need to remove the sliding window tokens
        size_update_local += tokens_info[:, :, -1, 2].long()

        return CacheUpdateInfo(
            b_idx=b_idx,
            h_idx=h_idx,
            update_idx=update_idx + size_local_involved + size_global_involved,
            shift_idx=size_global_involved,
            seq_len=seq_len,
            valid_value_mask=None,
            size_update_global=size_update_global,
            size_update_local=size_update_local,
            global_token_masks=None
        )
