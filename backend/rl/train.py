# backend/rl/train.py
# -*- coding: utf-8 -*-
"""
Go / Gomoku AlphaZero 风格自对弈强化学习（带 LLM 先验、轻量 PVNet、MCTS+PUCT）
- 轻量 PVNet: PyTorch 推理与训练（策略+价值）
- MCTS: 根处注入 Dirichlet 噪声，子树展开时继续融合 LLM 先验
- LLM 先验: 可选，从 HuggingFace 本地目录加载（SFT 结果），按 prior_mix 融合
- 训练过程：每局自对弈 → 累积(s, π, z) → 小批量训练 → 周期性保存 ckpt → 最后保存 final

依赖：
- Go   : backend/mcts/dlgo/*
- Gomoku: backend/gomoku/dlgo/*
"""

import os
import time
import math
import random
import importlib
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================
# 统一的棋种 API 选择层
# =========================================================
@dataclass
class GameAPI:
    GameState: Any
    Move: Any
    Point: Any
    Player: Any
    compute_game_result: Any

    def is_black(self, p) -> bool:
        try:
            return p == getattr(self.Player, "black")
        except Exception:
            return getattr(p, "name", "").lower() == "black"

    def is_white(self, p) -> bool:
        try:
            return p == getattr(self.Player, "white")
        except Exception:
            return getattr(p, "name", "").lower() == "white"


def get_game_api(game: str) -> GameAPI:
    """
    根据棋种返回统一访问接口。
      - Go      -> backend/mcts/dlgo/*
      - Gomoku  -> backend/gomoku/dlgo/*
    """
    if game == "gomoku":
        from ..gomoku.dlgo.goboard import GameState, Move
        from ..gomoku.dlgo.gotypes import Player, Point
        from ..gomoku.dlgo.scoring import compute_game_result
        return GameAPI(GameState=GameState, Move=Move, Point=Point,
                       Player=Player, compute_game_result=compute_game_result)
    else:  # 默认 Go
        from ..mcts.dlgo.goboard import GameState, Move
        from ..mcts.dlgo.gotypes import Player, Point
        from ..mcts.dlgo.scoring import compute_game_result
        return GameAPI(GameState=GameState, Move=Move, Point=Point,
                       Player=Player, compute_game_result=compute_game_result)


# =========================================================
# 设备/AMP
# =========================================================
def _get_device_and_amp(device: str):
    use_cuda = (device == "cuda" and torch.cuda.is_available())
    torch_device = torch.device("cuda" if use_cuda else "cpu")
    amp_autocast = torch.cuda.amp.autocast if use_cuda else nullcontext
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)
    return torch_device, amp_autocast, scaler, use_cuda


# =========================================================
# 轻量 PVNet（PyTorch）
# =========================================================
class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_ch, pw_ch, kernel_size=3, padding=1, alpha=0.01):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding,
                            groups=in_ch, bias=False)
        self.act1 = nn.LeakyReLU(negative_slope=alpha, inplace=True)
        self.pw = nn.Conv2d(in_ch, pw_ch, kernel_size=1, padding=0, bias=False)
        self.act2 = nn.LeakyReLU(negative_slope=alpha, inplace=True)

    def forward(self, x):
        x = self.dw(x)
        x = self.act1(x)
        x = self.pw(x)
        x = self.act2(x)
        return x


class LightweightPVNet(nn.Module):
    @staticmethod
    def _maybe_pool2x2(x, pool_layer: nn.Module):
        if x.size(-2) >= 2 and x.size(-1) >= 2:
            return pool_layer(x)
        return x

    def __init__(self, in_ch: int, board_size: int):
        super().__init__()
        self.board_size = board_size
        N = board_size * board_size

        self.conv1 = nn.Conv2d(in_ch, 32, kernel_size=3, padding=1, bias=False)
        self.act1 = nn.LeakyReLU(0.01, inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.block2 = DepthwiseSeparableBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.block3 = DepthwiseSeparableBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.block4 = DepthwiseSeparableBlock(128, 256)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.block5 = DepthwiseSeparableBlock(256, 512)
        self.pool5 = nn.AdaptiveAvgPool2d((2, 2))

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * 2 * 2, 512)
        self.fc_act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

        self.policy_head = nn.Linear(512, N)
        self.value_head = nn.Linear(512, 1)

    def forward(self, x):
        x = self.conv1(x); x = self.act1(x); x = self._maybe_pool2x2(x, self.pool1)
        x = self.block2(x); x = self._maybe_pool2x2(x, self.pool2)
        x = self.block3(x); x = self._maybe_pool2x2(x, self.pool3)
        x = self.block4(x); x = self._maybe_pool2x2(x, self.pool4)
        x = self.block5(x); x = self.pool5(x)
        x = self.flatten(x)
        x = self.fc(x); x = self.fc_act(x); x = self.dropout(x)
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value


# =========================================================
# 编码/动作映射（与 state 模块对齐）
# =========================================================
def _default_go_api() -> GameAPI:
    from ..mcts.dlgo.goboard import GameState, Move
    from ..mcts.dlgo.gotypes import Player, Point
    from ..mcts.dlgo.scoring import compute_game_result
    return GameAPI(GameState=GameState, Move=Move, Point=Point,
                   Player=Player, compute_game_result=compute_game_result)

def _resolve_classes(state: Any, api: Optional[GameAPI] = None):
    """
    优先从 state 所在模块拿 Move/Point；拿不到再用 api；api 为空时回退 Go 的默认实现。
    """
    try:
        mod = importlib.import_module(state.__class__.__module__)
        MoveCls = getattr(mod, "Move")
        PointCls = getattr(mod, "Point")
        return MoveCls, PointCls
    except Exception:
        if api is not None:
            return api.Move, api.Point
        else:
            d = _default_go_api()
            return d.Move, d.Point

def encode_state_planes(state: Any, board_size: int, api: Optional[GameAPI] = None) -> torch.Tensor:
    api = api or _default_go_api()
    planes = torch.zeros(3, board_size, board_size, dtype=torch.float32)
    for r in range(1, board_size + 1):
        for c in range(1, board_size + 1):
            p = api.Point(r, c)
            stone = state.board.get(p)
            if stone is None:
                continue
            if api.is_black(stone):
                planes[0, r - 1, c - 1] = 1.0
            else:
                planes[1, r - 1, c - 1] = 1.0
    planes[2, :, :] = 1.0 if api.is_black(state.next_player) else 0.0
    return planes

def move_index(r: int, c: int, board_size: int) -> int:
    return (r - 1) * board_size + (c - 1)

def index_to_move(idx: int, board_size: int,
                  api: Optional[GameAPI] = None,
                  state: Any = None):
    r = (idx // board_size) + 1
    c = (idx % board_size) + 1

    if state is not None:
        MoveCls, PointCls = _resolve_classes(state, api)
    elif api is not None:
        MoveCls, PointCls = api.Move, api.Point
    else:
        d = _default_go_api()
        MoveCls, PointCls = d.Move, d.Point

    return MoveCls.play(PointCls(r, c))

def legal_moves_and_mask(state: Any, board_size: int, api: GameAPI) -> Tuple[List[Any], torch.Tensor, List[int]]:
    MoveCls, PointCls = _resolve_classes(state, api)
    legals: List[Any] = []
    idxs: List[int] = []
    for r in range(1, board_size + 1):
        for c in range(1, board_size + 1):
            mv = MoveCls.play(PointCls(r, c))
            if state.is_valid_move(mv):
                legals.append(mv)
                idxs.append(move_index(r, c, board_size))
    mask = torch.zeros(board_size * board_size, dtype=torch.bool)
    if idxs:
        mask[torch.tensor(idxs, dtype=torch.long)] = True
    return legals, mask, idxs

def move_to_token_go(move: Any, player: Any, board_size: int, api: GameAPI) -> int:
    idx = move_index(move.point.row, move.point.col, board_size)
    return idx if api.is_black(player) else idx + board_size * board_size

def move_to_token_gomoku(move: Any, player: Any, board_size: int, api: GameAPI) -> int:
    idx = move_index(move.point.row, move.point.col, board_size)
    return idx if api.is_black(player) else idx + board_size * board_size

def move_to_token_for_game(game: str, move: Any, player: Any, board_size: int, api: GameAPI) -> int:
    if game == "gomoku":
        return move_to_token_gomoku(move, player, board_size, api)
    else:
        return move_to_token_go(move, player, board_size, api)


# =========================================================
# LLM 先验
# =========================================================
def top_k_logits(logits: torch.Tensor, k: int = 32) -> torch.Tensor:
    values, _ = torch.topk(logits, min(k, logits.size(-1)))
    thresh = values[:, -1, None]
    return torch.where(logits < thresh,
                       torch.tensor(float("-inf"), device=logits.device), logits)


# =========================================================
# MCTS 结点与 TD(λ) 增强
# =========================================================
class TreeNode:
    __slots__ = ["state", "parent", "player", "P", "N", "W", "Q",
                 "children", "move_from_parent", "history_ids", "E"]

    def __init__(self, state, parent, prior: torch.Tensor,
                 legal_idxs: List[int], board_size: int, move_from_parent,
                 history_ids: Optional[List[int]] = None):
        self.state = state
        self.parent = parent
        self.player = state.next_player
        self.P: Dict[int, float] = {idx: float(prior[idx]) for idx in legal_idxs}
        self.N: Dict[int, int] = {idx: 0 for idx in legal_idxs}
        self.W: Dict[int, float] = {idx: 0.0 for idx in legal_idxs}
        self.Q: Dict[int, float] = {idx: 0.0 for idx in legal_idxs}
        self.E: Dict[int, float] = {idx: 0.0 for idx in legal_idxs}
        self.children: Dict[int, "TreeNode"] = {}
        self.move_from_parent = move_from_parent
        self.history_ids = [] if history_ids is None else list(history_ids)

    def is_expanded(self):
        return len(self.P) > 0

    def best_child(self, c_puct: float) -> int:
        total_N = sum(self.N.values()) + 1
        best_score = -1e9
        best_idx = None
        for idx in self.P.keys():
            q = self.Q[idx]
            u = c_puct * self.P[idx] * math.sqrt(total_N) / (1 + self.N[idx])
            score = q + u
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx


def td_update_path_sarsa_lambda(
    path: List[Tuple["TreeNode", int]],
    gamma: float,
    lam: float,
    leaf_value: float,
    terminal: bool
):
    if not path:
        return
    v = leaf_value
    lam_pow = 1.0
    for (node, idx) in reversed(path):
        q_old = node.Q.get(idx, 0.0)
        td_err = v - q_old
        node.E[idx] = node.E.get(idx, 0.0) + lam_pow
        alpha = 0.1
        node.Q[idx] = q_old + alpha * td_err * node.E[idx]
        node.E[idx] *= gamma * lam
        v = -v
        lam_pow *= lam

    if terminal:
        node_T, idx_T = path[-1]
        child = node_T.children.get(idx_T)
        max_next_q = 0.0
        if child is not None and child.Q:
            max_next_q = max(child.Q.values())
        target = leaf_value + gamma * max_next_q
        alpha_T = 0.2
        node_T.Q[idx_T] = (1 - alpha_T) * node_T.Q[idx_T] + alpha_T * target


# =========================================================
# 单次模拟 & 搜索
# =========================================================
def mcts_search_once(
    root: "TreeNode",
    pvnet: LightweightPVNet,
    torch_device: torch.device,
    amp_autocast,
    prior_mix: float,
    llm, tok, api: GameAPI, game: str, board_size: int,
    c_puct: float = 2.0,
    gamma: float = 0.99,
    lam: float = 0.90,
):
    # Selection
    node = root
    path: List[Tuple["TreeNode", int]] = []
    while True:
        if node.state.is_over():
            break
        if not node.children:
            break
        idx = node.best_child(c_puct)
        path.append((node, idx))
        node = node.children[idx]

    # Expansion & Evaluation
    if not node.state.is_over():
        planes = encode_state_planes(node.state, board_size, api).unsqueeze(0).to(torch_device)
        with amp_autocast():
            policy_logits, value = pvnet(planes)
            pv_policy = F.softmax(policy_logits[0], dim=-1)  # (N,)

        legals, mask, idxs = legal_moves_and_mask(node.state, board_size, api)
        pv_prior = torch.where(mask.to(pv_policy.device), pv_policy, torch.zeros_like(pv_policy))
        if pv_prior.sum() > 0:
            pv_prior = pv_prior / pv_prior.sum()

        # 子树融合 LLM 先验
        if llm is not None and tok is not None:
            try:
                input_ids = torch.tensor(
                    [node.history_ids] if node.history_ids else
                    [[tok.bos_token_id if getattr(tok, "bos_token_id", None) is not None else 0]],
                    dtype=torch.long,
                    device=pv_policy.device
                )
                with torch.no_grad():
                    l_logits = llm(input_ids).logits[:, -1, :]
                    l_logits = top_k_logits(l_logits, k=64)
                    l_probs = F.softmax(l_logits, dim=-1)[0]
                N = board_size * board_size
                llm_prior_full = l_probs[:N] if api.is_black(node.state.next_player) else l_probs[N:2 * N]
                llm_prior_full = torch.where(mask.to(llm_prior_full.device), llm_prior_full,
                                             torch.zeros_like(llm_prior_full))
                if llm_prior_full.sum() > 0:
                    llm_prior_full = llm_prior_full / llm_prior_full.sum()
                blended_prior = prior_mix * llm_prior_full + (1 - prior_mix) * pv_prior
            except Exception:
                blended_prior = pv_prior
        else:
            blended_prior = pv_prior

        if blended_prior.sum() > 0:
            blended_prior = blended_prior / blended_prior.sum()

        for idx in idxs:
            mv = index_to_move(idx, board_size, api, state=node.state)
            child_state = node.state.apply_move(mv)
            _, child_mask, child_idxs = legal_moves_and_mask(child_state, board_size, api)
            child_prior = torch.where(child_mask.to(blended_prior.device), blended_prior,
                                      torch.zeros_like(blended_prior))

            child_token = move_to_token_for_game(game, mv, node.state.next_player, board_size, api)
            child_history = node.history_ids + [child_token]

            node.children[idx] = TreeNode(
                state=child_state,
                parent=node,
                prior=child_prior,
                legal_idxs=child_idxs,
                board_size=board_size,
                move_from_parent=mv,
                history_ids=child_history,
            )

        leaf_value = float(value.item())
    else:
        # 终局：从当前节点 player 视角
        result = api.compute_game_result(node.state)
        if result.winner is None:
            leaf_value = 0.0
        elif api.is_black(result.winner):
            leaf_value = 1.0 if api.is_black(node.player) else -1.0
        else:
            leaf_value = 1.0 if api.is_white(node.player) else -1.0

    # Backup
    v = leaf_value
    for parent, idx in reversed(path):
        parent.N[idx] += 1
        parent.W[idx] += v
        parent.Q[idx] = parent.W[idx] / parent.N[idx]
        v = -v

    # TD(λ) 增强
    td_update_path_sarsa_lambda(
        path=path, gamma=gamma, lam=lam,
        leaf_value=leaf_value, terminal=node.state.is_over()
    )


def run_mcts(
    state: Any,
    pvnet: LightweightPVNet,
    torch_device: torch.device,
    amp_autocast,
    blended_root_prior: torch.Tensor,
    board_size: int,
    llm, tok, api: GameAPI, game: str,
    gamma: float,
    lam: float,
    num_simulations: int = 300,
    c_puct: float = 2.0,
    root_dirichlet_alpha: float = 0.3,
    root_explore_frac: float = 0.25,
    history_ids: Optional[List[int]] = None,
    prior_mix: float = 0.0,   # <--- 新增：子树阶段也融合 LLM 先验的系数（0~1）
) -> Tuple["TreeNode", Dict[int, int]]:

    # 1) 取根节点合法动作及其索引
    legals, mask, idxs = legal_moves_and_mask(state, board_size, api)

    # 2) 根节点注入 Dirichlet 噪声
    prior = blended_root_prior.clone()
    if len(idxs) > 0:
        import numpy as np
        noise = np.random.dirichlet([root_dirichlet_alpha] * len(idxs)).astype("float32")
        noise_vec = torch.zeros_like(prior)
        idxs_tensor = torch.tensor(idxs, dtype=torch.long, device=prior.device)
        noise_vec[idxs_tensor] = torch.from_numpy(noise).to(prior.device)
        prior = (1.0 - root_explore_frac) * prior + root_explore_frac * noise_vec
        s = prior.sum()
        if float(s.item()) > 0:
            prior = prior / s

    # 3) 创建根节点
    root = TreeNode(
        state=state,
        parent=None,
        prior=prior,
        legal_idxs=idxs,
        board_size=board_size,
        move_from_parent=None,
        history_ids=(history_ids or [])
    )

    # 4) 多次模拟
    for _ in range(num_simulations):
        mcts_search_once(
            root=root,
            pvnet=pvnet,
            torch_device=torch_device,
            amp_autocast=amp_autocast,
            prior_mix=prior_mix,     # <--- 关键：将 prior_mix 传入，让子树也融合 LLM 先验
            llm=llm,
            tok=tok,
            api=api,
            game=game,
            board_size=board_size,
            c_puct=c_puct,
            gamma=gamma,
            lam=lam
        )

    # 5) 汇总根节点的访问计数
    visit_counts = {idx: root.N.get(idx, 0) for idx in idxs}
    return root, visit_counts



# =========================================================
# Loss
# =========================================================
def policy_value_loss(
    policy_logits: torch.Tensor,  # (B, N)
    target_pi: torch.Tensor,      # (B, N)
    pred_value: torch.Tensor,     # (B, 1)
    target_value: torch.Tensor,   # (B, 1)
    l2_reg: float = 0.0,
    model: Optional[nn.Module] = None
):
    logp = F.log_softmax(policy_logits, dim=-1)
    policy_loss = -(target_pi * logp).sum(dim=-1).mean()
    value_loss = F.mse_loss(pred_value, target_value)
    loss = policy_loss + value_loss
    if l2_reg > 0.0 and model is not None:
        l2 = sum((p**2).sum() for n, p in model.named_parameters() if "bias" not in n)
        loss = loss + l2_reg * l2
    return loss, policy_loss.item(), value_loss.item()


# =========================================================
# 主入口（生成器：持续输出进度）
# =========================================================
def run_rl(
    game: str,
    board_size: int,
    episodes: int,
    sims: int,
    gamma: float,
    lam: float,
    note: str,
    model_root: str,
    llm_subdir: Optional[str],
    prior_mix: float,
    device: str,
    save_root: str,
):
    if game not in ("go", "gomoku"):
        yield "[RL] 暂仅支持 Go / Gomoku。"
        return

    api = get_game_api(game)
    torch_device, amp_autocast, scaler, use_cuda = _get_device_and_amp(device)

    tag = f"{game}-pvnet-sims{sims}-epi{episodes}-{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = os.path.join(save_root, tag)
    os.makedirs(out_dir, exist_ok=True)
    yield f"[RL] 输出目录：{out_dir}"
    yield f"[RL] 配置：game={game}, size={board_size}, episodes={episodes}, sims={sims}, device={torch_device.type}, prior_mix={prior_mix}"

    # 加载 SFT 先验（可选）
    llm, tok = None, None
    if llm_subdir:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        try:
            sft_path = llm_subdir
            if not os.path.isabs(sft_path):
                sft_path = os.path.join(model_root, "sft", sft_path)
            sft_path = os.path.normpath(os.path.abspath(sft_path))

            def _is_hf_dir(p):
                if not os.path.isdir(p): return False
                has_cfg = os.path.isfile(os.path.join(p, "config.json"))
                has_pt  = any(os.path.isfile(os.path.join(p, fn)) for fn in ["pytorch_model.bin", "model.safetensors"])
                has_tf  = os.path.isfile(os.path.join(p, "tf_model.h5"))
                return has_cfg and (has_pt or has_tf)

            if not _is_hf_dir(sft_path):
                raise RuntimeError(f"未找到有效HF目录：{sft_path}")

            tok = AutoTokenizer.from_pretrained(sft_path, local_files_only=True)
            try:
                llm = AutoModelForCausalLM.from_pretrained(sft_path, local_files_only=True)
            except OSError:
                # 仅有 TensorFlow 权重时回退
                llm = AutoModelForCausalLM.from_pretrained(sft_path, local_files_only=True, from_tf=True)

            llm.resize_token_embeddings(len(tok))
            llm.to(torch_device).eval()
            yield f"[RL] 已加载 SFT 先验：{os.path.basename(sft_path)}"
        except Exception as e:
            yield f"[RL] 警告：加载 SFT 失败，关闭 LLM 先验 —— {e}"
            llm, tok = None, None
    else:
        yield "[RL] 未选择 SFT 模型，LLM 先验关闭。"

    # 构建 PVNet
    pvnet = LightweightPVNet(in_ch=3, board_size=board_size).to(torch_device)
    optimizer = torch.optim.AdamW(pvnet.parameters(), lr=1e-3, weight_decay=1e-4)

    CKPT_EVERY = max(5, min(50, episodes // 10))
    BATCH_SIZE = 64
    TEMP_CHOICE = 1.0

    for ep in range(1, episodes + 1):
        ep_t0 = time.time()
        state = api.GameState.new_game(board_size)
        history_ids: List[int] = []
        replay_states: List[torch.Tensor] = []
        replay_pis: List[torch.Tensor] = []
        players: List[Any] = []

        steps = 0
        max_steps = board_size * board_size
        report_every = max(1, max_steps // 10)

        yield f"[RL][Ep {ep}] 自对弈开始。"

        while not state.is_over() and steps < max_steps:
            steps += 1

            planes = encode_state_planes(state, board_size, api).unsqueeze(0).to(torch_device)
            with torch.no_grad():
                pv_logits, _ = pvnet(planes)
                pv_policy = F.softmax(pv_logits[0], dim=-1)

            _, mask_root, _ = legal_moves_and_mask(state, board_size, api)
            blended_root_prior = torch.where(mask_root.to(pv_policy.device),
                                             pv_policy, torch.zeros_like(pv_policy))

            # 根融合 LLM
            if llm is not None and tok is not None:
                N = board_size * board_size
                input_ids = torch.tensor(
                    [history_ids] if history_ids else
                    [[getattr(tok, "bos_token_id", 0)]],
                    dtype=torch.long, device=torch_device
                )
                with torch.no_grad():
                    logits = llm(input_ids).logits[:, -1, :]
                    logits = top_k_logits(logits, k=64)
                    probs_voc = F.softmax(logits, dim=-1)[0]

                if probs_voc.numel() >= 2 * N:
                    llm_prior_full = probs_voc[:N] if api.is_black(state.next_player) else probs_voc[N:2 * N]
                    llm_prior_full = torch.where(mask_root.to(llm_prior_full.device),
                                                 llm_prior_full, torch.zeros_like(llm_prior_full))
                    if llm_prior_full.sum() > 0:
                        llm_prior_full = llm_prior_full / llm_prior_full.sum()
                    blended_root_prior = prior_mix * llm_prior_full.to(pv_policy.device) + \
                        (1 - prior_mix) * blended_root_prior

            if blended_root_prior.sum() > 0:
                blended_root_prior = blended_root_prior / blended_root_prior.sum()

            root, visit_counts = run_mcts(
                state=state, pvnet=pvnet, torch_device=torch_device, amp_autocast=amp_autocast,
                blended_root_prior=blended_root_prior, board_size=board_size,
                llm=llm, tok=tok, api=api, game=game, gamma=gamma, lam=lam,
                num_simulations=sims, c_puct=2.0, history_ids=history_ids,
                prior_mix=prior_mix
            )

            N_vec = torch.zeros(board_size * board_size, dtype=torch.float32, device=torch_device)
            for idx, v in visit_counts.items():
                N_vec[idx] = float(v)
            if N_vec.sum() > 0:
                pi = (N_vec ** (1.0 / TEMP_CHOICE)); pi = pi / pi.sum()
            else:
                pi = torch.where(mask_root.to(N_vec.device), torch.ones_like(N_vec), torch.zeros_like(N_vec))
                pi = pi / pi.sum()

            replay_states.append(planes[0].detach().cpu())
            replay_pis.append(pi.detach().cpu())
            players.append(state.next_player)

            idx = torch.multinomial(pi, num_samples=1).item()
            mv = index_to_move(idx, board_size, api, state=state)
            if not state.is_valid_move(mv):
                _, _, idxs = legal_moves_and_mask(state, board_size, api)
                if idxs:
                    idx = random.choice(idxs)
                    mv = index_to_move(idx, board_size, api, state=state)
                else:
                    break

            token = move_to_token_for_game(game, mv, state.next_player, board_size, api)
            history_ids.append(token)

            state = state.apply_move(mv)

            if steps % report_every == 0:
                yield f"[RL][Ep {ep}] 自对弈进度：{steps}/{max_steps} (~{int(100*steps/max_steps)}%)"

        # 终局值
        result = api.compute_game_result(state)
        if getattr(result, "winner", None) is None:
            z_final = 0.0
        else:
            z_final = 1.0

        targets_v = []
        for p in players:
            if getattr(result, "winner", None) is None:
                z = 0.0
            else:
                z = z_final if ((api.is_black(p) and api.is_black(result.winner)) or
                                (api.is_white(p) and api.is_white(result.winner))) else -z_final
            targets_v.append([z])
        targets_v = torch.tensor(targets_v, dtype=torch.float32)

        X = torch.stack(replay_states, dim=0).to(torch_device)
        Y_pi = torch.stack(replay_pis, dim=0).to(torch_device)
        Y_v = targets_v.to(torch_device)

        pvnet.train()
        total_loss = total_pl = total_vl = 0.0
        batches = 0
        B = X.size(0)
        for i in range(0, B, BATCH_SIZE):
            xb = X[i:i + BATCH_SIZE]
            pib = Y_pi[i:i + BATCH_SIZE]
            vb = Y_v[i:i + BATCH_SIZE]

            optimizer.zero_grad(set_to_none=True)
            with amp_autocast():
                logits, pred_v = pvnet(xb)
                loss, pl, vl = policy_value_loss(
                    policy_logits=logits, target_pi=pib, pred_value=pred_v, target_value=vb,
                    l2_reg=1e-4, model=pvnet
                )
            if use_cuda:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += float(loss.detach().cpu().item())
            total_pl += pl; total_vl += vl; batches += 1

        ep_time = time.time() - ep_t0
        if batches == 0:
            yield f"[RL][Ep {ep}] 无样本（可能提前终局），跳过训练。用时 {ep_time:.1f}s"
        else:
            yield f"[RL][Ep {ep}] 完成：steps={steps}, loss={total_loss/batches:.4f}, pl={total_pl/batches:.4f}, vl={total_vl/batches:.4f}, time={ep_time:.1f}s, winner={getattr(result, 'winner', None)}"

        if ep % CKPT_EVERY == 0:
            ckpt_dir = os.path.join(out_dir, f"ckpt_ep{ep}")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(pvnet.state_dict(), os.path.join(ckpt_dir, "pvnet.pt"))
            yield f"[RL] 已保存检查点：{ckpt_dir}"

    final_dir = os.path.join(out_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    torch.save(pvnet.state_dict(), os.path.join(final_dir, "pvnet.pt"))
    yield f"[RL] 训练完成 ✅ 最终 PVNet 已保存：{final_dir}"
