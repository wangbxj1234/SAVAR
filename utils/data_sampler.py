import numpy as np
import torch
from torch.utils.data.sampler import Sampler


class EvalDistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas, rank):
        seps = np.linspace(0, len(dataset), num_replicas+1, dtype=int)
        beg, end = seps[:-1], seps[1:]
        beg, end = beg[rank], end[rank]
        self.indices = tuple(range(beg, end))
    
    def __iter__(self):
        return iter(self.indices)
    
    def __len__(self) -> int:
        return len(self.indices)


class InfiniteBatchSampler(Sampler):
    def __init__(self, dataset_len, batch_size, seed_for_all_rank=0, fill_last=False, shuffle=True, drop_last=False, start_ep=0, start_it=0):
        self.dataset_len = dataset_len
        self.batch_size = batch_size
        self.iters_per_ep = dataset_len // batch_size if drop_last else (dataset_len + batch_size - 1) // batch_size
        self.max_p = self.iters_per_ep * batch_size
        self.fill_last = fill_last
        self.shuffle = shuffle
        self.epoch = start_ep
        self.same_seed_for_all_ranks = seed_for_all_rank
        self.indices = self.gener_indices()
        self.start_ep, self.start_it = start_ep, start_it
    
    def gener_indices(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch + self.same_seed_for_all_ranks)
            indices = torch.randperm(self.dataset_len, generator=g).numpy()
        else:
            indices = torch.arange(self.dataset_len).numpy()
        
        tails = self.batch_size - (self.dataset_len % self.batch_size)
        if tails != self.batch_size and self.fill_last:
            tails = indices[:tails]
            np.random.shuffle(indices)
            indices = np.concatenate((indices, tails))
        
        # built-in list/tuple is faster than np.ndarray (when collating the data via a for-loop)
        # noinspection PyTypeChecker
        return tuple(indices.tolist())
    
    def __iter__(self):
        # start_ep/start_it are for resume; we must begin exactly at that point
        self.epoch = self.start_ep
        while True:
            p = (self.start_it * self.batch_size) if self.epoch == self.start_ep else 0
            while p < self.max_p:
                q = p + self.batch_size
                yield self.indices[p:q]
                p = q
            self.epoch += 1
            if self.shuffle:
                self.indices = self.gener_indices()
    
    def __len__(self):
        return self.iters_per_ep


class DistInfiniteBatchSampler(InfiniteBatchSampler):
    def __init__(self, world_size, rank, dataset_len, glb_batch_size, same_seed_for_all_ranks=0, repeated_aug=0, fill_last=False, shuffle=True, start_ep=0, start_it=0):
        assert glb_batch_size % world_size == 0
        self.world_size, self.rank = world_size, rank
        self.dataset_len = dataset_len
        self.glb_batch_size = glb_batch_size
        self.batch_size = glb_batch_size // world_size
        
        self.iters_per_ep = (dataset_len + glb_batch_size - 1) // glb_batch_size
        self.fill_last = fill_last
        self.shuffle = shuffle
        self.repeated_aug = repeated_aug
        self.epoch = start_ep
        self.same_seed_for_all_ranks = same_seed_for_all_ranks
        self.indices = self.gener_indices()
        self.start_ep, self.start_it = start_ep, start_it
    
    def gener_indices(self):
        global_max_p = self.iters_per_ep * self.glb_batch_size
        if self.shuffle:
            g = torch.Generator()
            # 使用连续的种子让数据顺序更平滑，而不是每个epoch完全重新shuffle
            # 原来: self.epoch + self.same_seed_for_all_ranks
            # 修改为: 基于总iteration数而非epoch数
            seed = (self.epoch * self.iters_per_ep) + self.same_seed_for_all_ranks
            g.manual_seed(seed)
            global_indices = torch.randperm(self.dataset_len, generator=g)
            if self.repeated_aug > 1:
                global_indices = global_indices[:(self.dataset_len + self.repeated_aug - 1) // self.repeated_aug].repeat_interleave(self.repeated_aug, dim=0)
        else:
            global_indices = torch.arange(self.dataset_len)
        
        # 确保 global_indices 长度至少为 global_max_p
        if global_indices.shape[0] < global_max_p:
            filling = global_max_p - global_indices.shape[0]
            global_indices = torch.cat((global_indices, global_indices[:filling]))
        
        # 严格截断到 global_max_p，防止不同 epoch 之间数据重叠或多余
        global_indices = global_indices[:global_max_p]
        
        # 均匀分配给各个 rank
        local_chunk_size = global_max_p // self.world_size
        local_indices = global_indices[self.rank * local_chunk_size : (self.rank + 1) * local_chunk_size].tolist()
        self.max_p = len(local_indices)
        return local_indices
