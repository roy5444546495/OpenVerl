import torch

def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.

    Args:
        loss_mat: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_agg_mode: (str) choices:
            method to aggregate the loss matrix into a scalar.
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
  
    loss = masked_mean(loss_mat, loss_mask)


    return loss

def masked_sum(values, mask, axis=None):
    """Compute mean of tensor with a masked values."""
    # If NaNs exist out of mask, replace NaNs in values with a value that
    # won't affect the sum (e.g., 0 for masked regions)
    # valid_values = torch.where(mask.bool(), values, 0.0)
    valid_values = torch.where(mask.bool(), values, torch.tensor(0.0, dtype=values.dtype, device=values.device))
    if axis is None:
        return (valid_values * mask).sum()
    else:
        return (valid_values * mask).sum(dim=axis)

def masked_mean(values, mask, axis=None):
    """
    Compute the mean of `values` over elements selected by `mask`.

    Args:
        values (Tensor): Input tensor.
        mask (Tensor): Boolean or numeric mask of the same shape as `values`.
        axis (int or tuple of int, optional): Dimension(s) along which to compute the mean.
            Defaults to None (over all elements).

    Returns:
        Tensor: Masked mean, with shape equal to `values` reduced over `axis`.
    """
    s = masked_sum(values, mask, axis)
    if axis is None:
        return s / (mask.sum() + 1e-8)
    else:
        return s / (mask.sum(dim=axis) + 1e-8)

def compute_policy_loss_clip_cov(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    rollout_log_probs: torch.Tensor = None,
    current_step: int = 0,
    total_steps: int = 1000000,
):
    clip_cov_ratio = 0.0002
    cliprange = 0.2
    cliprange_low = 0.2
    cliprange_high = 0.2
    clip_cov_ub = 5.0
    clip_cov_lb = 1.0

    assert clip_cov_ratio > 0, "clip_ratio should be larger than 0."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio

    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    corr = torch.ones_like(advantages)
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_by_origin = (pg_losses2 > pg_losses1) & (response_mask > 0)

    cov_all = (advantages - masked_mean(advantages, response_mask)) * (
        log_prob - masked_mean(log_prob.detach(), response_mask)
    )
    cov_all[response_mask == 0] = -torch.inf
    cov_all[clip_by_origin] = -torch.inf
    print("check")
    from pprint import pprint
    pprint(cov_all)
    
    clip_num = max(int(clip_cov_ratio * response_mask.sum().item()), 1)
    print("clip num", clip_num, response_mask.sum())
    top_k_idx = (cov_all < clip_cov_ub) & (cov_all > clip_cov_lb) & (response_mask > 0)
    print("topk", top_k_idx)
    top_k_idx = torch.nonzero(top_k_idx)
    print("topk", top_k_idx)

    print("len", len(top_k_idx))
    if len(top_k_idx) > 0:
        perm = torch.randperm(len(top_k_idx))
        top_k_idx = top_k_idx[perm[: min(clip_num, len(top_k_idx))]]
    else:
        top_k_idx = torch.empty((0, 2), device=cov_all.device, dtype=torch.long)

    corr[top_k_idx[:, 0], top_k_idx[:, 1]] = 0

    pg_clipfrac = masked_mean((corr == 0).float(), response_mask)

    pg_losses = torch.maximum(pg_losses1, pg_losses2) * corr
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, torch.tensor(0.0)

import math
import torch.nn.functional as F

def compute_policy_loss_clip_cov_enhance(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    rollout_log_probs: torch.Tensor = None,
    current_step: int = 0,
    total_steps: int = 1000000,
):
    clip_cov_ratio = 0.0002
    cliprange = 0.2
    cliprange_low = 0.2
    cliprange_high = 0.2
    clip_cov_ub = 5.0
    clip_cov_lb = 1.0
    
    # 获取自适应阈值参数（新增）
    clip_cov_ub_min = 3.0
    clip_cov_ub_max = 8.0
    clip_cov_lb_min = 0.5
    clip_cov_lb_max = 3.0
    adaptive_decay = 0.1  # 自适应衰减系数
    
    assert clip_cov_ratio > 0, "clip_ratio should be larger than 0."

    # 计算自适应阈值（新增）
    progress = current_step / total_steps
    # 使用指数衰减调整阈值
    clip_cov_ub = clip_cov_ub_min + (clip_cov_ub_max - clip_cov_ub_min) * math.exp(-adaptive_decay * progress)
    clip_cov_lb = clip_cov_lb_min + (clip_cov_lb_max - clip_cov_lb_min) * (1 - math.exp(-adaptive_decay * progress))
    clip_cov_ub = 5.0
    clip_cov_lb = 1.0
    
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio

    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    corr = torch.ones_like(advantages)
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_by_origin = (pg_losses2 > pg_losses1) & (response_mask > 0)

    cov_all = (advantages - masked_mean(advantages, response_mask)) * (
        log_prob - masked_mean(log_prob.detach(), response_mask)
    )
    cov_all[response_mask == 0] = -torch.inf
    cov_all[clip_by_origin] = -torch.inf

    clip_num = max(int(clip_cov_ratio * response_mask.sum().item()), 1)
    top_k_idx = (cov_all < clip_cov_ub) & (cov_all > clip_cov_lb) & (response_mask > 0)

    top_k_idx = torch.nonzero(top_k_idx)
    
    # 计算每个位置的裁剪概率（新增：重要性加权）
    if len(top_k_idx) > 0:
        # [clip_cov_lb, clip_cov_ub]范围内的协方差
        cov_values = torch.abs(cov_all[top_k_idx[:, 0], top_k_idx[:, 1]])
        print("cov_v", cov_values)

        clip_probs = F.softmax(cov_values, dim=-1)
        print("probs", clip_probs)
        sampled_indices = torch.multinomial(clip_probs, clip_num, replacement=False)

        # 获取采样结果
        sampled_top_k = top_k_idx[sampled_indices]
        print("result", sampled_top_k)
     
    
        
        # 应用裁剪
        if len(sampled_top_k) > 0:
            corr[sampled_top_k[:, 0], sampled_top_k[:, 1]] = 0
    
    # 计算裁剪比例
    pg_clipfrac = masked_mean((corr == 0).float(), response_mask)

    pg_losses = torch.maximum(pg_losses1, pg_losses2) * corr
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, torch.tensor(0.0)

def test_clip_cov():
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    
    # 模拟配置
    config_dict = {
        "clip_ratio": 0.2,
        "clip_ratio_low": 0.1,
        "clip_ratio_high": 0.3,
        "policy_loss": {
            "clip_cov_ratio": 0.0005,
            "clip_cov_ub_min": 2.0,
            "clip_cov_ub_max": 6.0,
            "clip_cov_lb_min": 1.0,
            "clip_cov_lb_max": 3.0,
            "adaptive_decay": 0.2
        }
    }
    
    # 模拟输入数据 - 确保所有张量都是float32类型
    batch_size, response_length = 4, 10
    
    # 模拟 old_log_prob 和 log_prob (策略概率)
    old_log_prob = torch.randn(batch_size, response_length, dtype=torch.float32)
    log_prob = old_log_prob + torch.randn(batch_size, response_length, dtype=torch.float32) * 0.1  # 稍微改变
    
    # 模拟 advantages (优势函数)
    advantages = torch.randn(batch_size, response_length, dtype=torch.float32)
    
    # 模拟 response_mask (响应掩码)
    response_mask = torch.ones(batch_size, response_length, dtype=torch.float32)
    # 随机将一些位置设为0
    mask_zero_indices = torch.rand(batch_size, response_length) < 0.2
    response_mask[mask_zero_indices] = 0
    
    # 模拟训练步数
    current_step = 5000
    total_steps = 100000
    
    print("输入数据形状:")
    print(f"old_log_prob: {old_log_prob.shape}")
    print(f"log_prob: {log_prob.shape}")
    print(f"advantages: {advantages.shape}")
    print(f"response_mask: {response_mask.shape}")
    print(f"非零掩码数量: {response_mask.sum().item()}")
    
    # 调用函数
    pg_loss, pg_clipfrac, ppo_kl, _ = compute_policy_loss_clip_cov(
        old_log_prob, 
        log_prob, 
        advantages, 
        response_mask,
        "token-mean",
        None,
        current_step, 
        total_steps
    )
    
    print("\n输出结果:")
    print(f"策略损失 (pg_loss): {pg_loss.item()}")
    print(f"裁剪比例 (pg_clipfrac): {pg_clipfrac.item()}")
    print(f"近似KL散度 (ppo_kl): {ppo_kl.item()}")
    
    # 验证输出类型和形状
    assert isinstance(pg_loss, torch.Tensor)
    assert isinstance(pg_clipfrac, torch.Tensor)
    assert isinstance(ppo_kl, torch.Tensor)
    assert pg_loss.dim() == 0  # 标量
    assert pg_clipfrac.dim() == 0  # 标量
    assert ppo_kl.dim() == 0  # 标量
    
    print("\n测试通过! 所有输出都是标量张量。")

def test_clip_cov_enhance():
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
        
    # 模拟输入数据 - 确保所有张量都是float32类型
    batch_size, response_length = 4, 10
    
    # 模拟 old_log_prob 和 log_prob (策略概率)
    old_log_prob = torch.randn(batch_size, response_length, dtype=torch.float32)
    log_prob = old_log_prob + torch.randn(batch_size, response_length, dtype=torch.float32) * 0.1  # 稍微改变
    
    # 模拟 advantages (优势函数)
    advantages = torch.randn(batch_size, response_length, dtype=torch.float32)
    
    # 模拟 response_mask (响应掩码)
    response_mask = torch.ones(batch_size, response_length, dtype=torch.float32)
    # 随机将一些位置设为0
    mask_zero_indices = torch.rand(batch_size, response_length) < 0.2
    response_mask[mask_zero_indices] = 0
    
    # 模拟训练步数
    current_step = 5000
    total_steps = 100000
    
    print("输入数据形状:")
    print(f"old_log_prob: {old_log_prob.shape}")
    print(f"log_prob: {log_prob.shape}")
    print(f"advantages: {advantages.shape}")
    print(f"response_mask: {response_mask.shape}")
    print(f"非零掩码数量: {response_mask.sum().item()}")
    
    # 调用函数
    pg_loss, pg_clipfrac, ppo_kl, _ = compute_policy_loss_clip_cov_enhance(
        old_log_prob, 
        log_prob, 
        advantages, 
        response_mask,
        "token-mean",
        None,
        current_step, 
        total_steps
    )
    
    print("\n输出结果:")
    print(f"策略损失 (pg_loss): {pg_loss.item()}")
    print(f"裁剪比例 (pg_clipfrac): {pg_clipfrac.item()}")
    print(f"近似KL散度 (ppo_kl): {ppo_kl.item()}")
    
    # 验证输出类型和形状
    assert isinstance(pg_loss, torch.Tensor)
    assert isinstance(pg_clipfrac, torch.Tensor)
    assert isinstance(ppo_kl, torch.Tensor)
    assert pg_loss.dim() == 0  # 标量
    assert pg_clipfrac.dim() == 0  # 标量
    assert ppo_kl.dim() == 0  # 标量
    
    print("\n测试通过! 所有输出都是标量张量。")

# 运行测试
if __name__ == "__main__":
    test_clip_cov()
    test_clip_cov_enhance()