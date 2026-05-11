import math
from typing import List, Optional, Tuple, Union

import torch


class NullCtx:
    def __enter__(self):
        pass
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class AmpOptimizer:
    def __init__(
        self,
        mixed_precision: int,
        optimizer: torch.optim.Optimizer, names: List[str], paras: List[torch.nn.Parameter],
        grad_clip: float, n_gradient_accumulation: int = 1,
    ):
        self.enable_amp = mixed_precision > 0
        self.using_fp16_rather_bf16 = mixed_precision == 1
        
        if self.enable_amp:
            self.amp_ctx = torch.autocast('cuda', enabled=True, dtype=torch.float16 if self.using_fp16_rather_bf16 else torch.bfloat16, cache_enabled=True)
            self.scaler = torch.cuda.amp.GradScaler(init_scale=2. ** 11, growth_interval=1000) if self.using_fp16_rather_bf16 else None # only fp16 needs a scaler
        else:
            self.amp_ctx = NullCtx()
            self.scaler = None
        
        self.optimizer, self.names, self.paras = optimizer, names, paras   # paras have been filtered so everyone requires grad
        self.grad_clip = grad_clip
        self.early_clipping = self.grad_clip > 0 and not hasattr(optimizer, 'global_grad_norm')
        self.late_clipping = self.grad_clip > 0 and hasattr(optimizer, 'global_grad_norm')
        
        self.r_accu = 1 / n_gradient_accumulation   # r_accu == 1.0 / n_gradient_accumulation
        self.last_grad_norm = None
        self.last_clipped = None
    
    def backward_clip_step(
        self, stepping: bool, loss: torch.Tensor,
    ) -> Tuple[Optional[Union[torch.Tensor, float]], Optional[float]]:
        # backward
        loss = loss.mul(self.r_accu)   # r_accu == 1.0 / n_gradient_accumulation
        orig_norm = scaler_sc = None
        if self.scaler is not None:
            self.scaler.scale(loss).backward(retain_graph=False, create_graph=False)
        else:
            loss.backward(retain_graph=False, create_graph=False)
        
        if stepping:
            if self.scaler is not None: self.scaler.unscale_(self.optimizer)
            if self.early_clipping:
                orig_norm = torch.nn.utils.clip_grad_norm_(self.paras, self.grad_clip)
            
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                scaler_sc: float = self.scaler.get_scale()
                if scaler_sc > 32768.: # fp16 will overflow when >65536, so multiply 32768 could be dangerous
                    self.scaler.update(new_scale=32768.)
                else:
                    self.scaler.update()
                try:
                    scaler_sc = float(math.log2(scaler_sc))
                except Exception as e:
                    print(f'[scaler_sc = {scaler_sc}]\n' * 15, flush=True)
                    raise e
            else:
                self.optimizer.step()
            
            if self.late_clipping:
                orig_norm = self.optimizer.global_grad_norm
            
            self.optimizer.zero_grad(set_to_none=True)
        
        if orig_norm is not None and self.grad_clip > 0:
            try:
                norm_val = orig_norm.item() if hasattr(orig_norm, 'item') else float(orig_norm)
            except Exception:
                norm_val = None
            self.last_grad_norm = norm_val
            self.last_clipped = float(norm_val is not None and norm_val > self.grad_clip)
        else:
            self.last_grad_norm = None
            self.last_clipped = None
        
        return orig_norm, scaler_sc
    
    def state_dict(self):
        return {
            'optimizer': self.optimizer.state_dict()
        } if self.scaler is None else {
            'scaler': self.scaler.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
    
    def load_state_dict(self, state, strict=True):
        if self.scaler is not None:
            try: 
                self.scaler.load_state_dict(state['scaler'])
                scale = self.scaler.get_scale()
                print(f'[AmpOptimizer.load_state_dict] scaler loaded successfully, current scale: {scale:.2f} (log2: {math.log2(scale):.2f})', flush=True)
            except Exception as e: 
                print(f'[AmpOptimizer.load_state_dict] ERROR loading scaler: {e}', flush=True)
                import traceback
                traceback.print_exc()
        try:
            # 检查 checkpoint 中的 optimizer 状态
            ckpt_state = state['optimizer'].get('state', {})
            ckpt_state_keys = len(ckpt_state)
            ckpt_param_groups = len(state['optimizer'].get('param_groups', []))
            
            # 加载 optimizer 状态
            self.optimizer.load_state_dict(state['optimizer'])
            
            # 检查加载后的 optimizer 状态
            loaded_state = self.optimizer.state
            loaded_state_keys = len(loaded_state)
            loaded_param_groups = len(self.optimizer.param_groups)
            
            # 检查是否有参数的状态丢失
            total_params = sum(len(pg['params']) for pg in self.optimizer.param_groups)
            # 使用 id() 来比较参数对象，避免 tensor 布尔值转换错误
            all_params_set = {id(param) for pg in self.optimizer.param_groups for param in pg['params']}
            params_with_state = sum(1 for p in loaded_state.keys() if id(p) in all_params_set)
            
            print(f'[AmpOptimizer.load_state_dict] optimizer loaded: ckpt_state_keys={ckpt_state_keys}, loaded_state_keys={loaded_state_keys}, '
                  f'ckpt_param_groups={ckpt_param_groups}, loaded_param_groups={loaded_param_groups}, '
                  f'total_params={total_params}, params_with_state={params_with_state}', flush=True)
            
            if ckpt_state_keys > 0 and loaded_state_keys == 0:
                print(f'[AmpOptimizer.load_state_dict] CRITICAL WARNING: optimizer state was lost during loading!', flush=True)
            elif ckpt_state_keys != loaded_state_keys:
                print(f'[AmpOptimizer.load_state_dict] WARNING: optimizer state count mismatch: ckpt={ckpt_state_keys}, loaded={loaded_state_keys}', flush=True)
        except Exception as e:
            print(f'[AmpOptimizer.load_state_dict] ERROR loading optimizer: {e}', flush=True)
            import traceback
            traceback.print_exc()
            raise
