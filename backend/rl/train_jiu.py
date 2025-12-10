# backend/rl/train_jiu.py
# -*- coding: utf-8 -*-
import os
import time
import math
import random
import re
from typing import List, Tuple, Optional, Dict
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

# —— 你项目里的 Jiu 规则实现 —— #
from ..jiu.dlgo.goboard import GameState, Move
from ..jiu.dlgo.gotypes import Player, Point
from ..jiu.dlgo.scoring import compute_game_result


# =========================
# 设备/AMP
# =========================
def _get_device_and_amp(device: str):
    use_cuda = (device == "cuda" and torch.cuda.is_available())
    torch_device = torch.device("cuda" if use_cuda else "cpu")
    amp_autocast = torch.cuda.amp.autocast if use_cuda else nullcontext
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)
    return torch_device, amp_autocast, scaler, use_cuda


# =========================
# 轻量策略-价值网络（久棋）
# - 策略：棋盘热力图 (H×W)
# - 价值：标量
# - 安全池化，避免 1×1 → 0×0
# =========================
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


class JiuLightweightPVNet(nn.Module):
    def __init__(self, in_ch: int, board_size: int):
        super().__init__()
        self.board_size = board_size

        # Block 1
        self.conv1 = nn.Conv2d(in_ch, 32, kernel_size=3, padding=1, bias=False)
        self.act1 = nn.LeakyReLU(0.01, inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Block 2
        self.block2 = DepthwiseSeparableBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Block 3
        self.block3 = DepthwiseSeparableBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Block 4
        self.block4 = DepthwiseSeparableBlock(128, 256)
        self.pool4 = nn.MaxPool2d(2, 2)  # forward 里会按需调用

        # Block 5
        self.block5 = DepthwiseSeparableBlock(256, 512)

        # 最终自适应到 2x2
        self.final_pool = nn.AdaptiveAvgPool2d((2, 2))

        # Head
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * 2 * 2, 512)
        self.fc_act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

        # 双头
        self.value_head = nn.Linear(512, 1)
        self.policy_head = nn.Linear(512, board_size * board_size)  # 棋盘热力图 logits

    @staticmethod
    def _maybe_pool2x2(x, pool_layer: nn.Module):
        if x.size(-2) >= 2 and x.size(-1) >= 2:
            return pool_layer(x)
        return x

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.conv1(x); x = self.act1(x); x = self._maybe_pool2x2(x, self.pool1)
        x = self.block2(x); x = self._maybe_pool2x2(x, self.pool2)
        x = self.block3(x); x = self._maybe_pool2x2(x, self.pool3)
        x = self.block4(x); x = self._maybe_pool2x2(x, self.pool4)
        x = self.block5(x)
        x = self.final_pool(x)
        x = self.flatten(x)
        x = self.fc(x); x = self.fc_act(x); x = self.dropout(x)
        v = self.value_head(x)                                            # (B,1)
        p = self.policy_head(x).view(-1, self.board_size, self.board_size)  # (B,H,W)
        return p, v


# =========================
# 编码：3 平面（黑/白/当前手）
# =========================
def encode_state_planes(state: GameState, board_size: int) -> torch.Tensor:
    planes = torch.zeros(3, board_size, board_size, dtype=torch.float32)
    for r in range(1, board_size + 1):
        for c in range(1, board_size + 1):
            p = Point(r, c)
            stone = state.board.get(p)
            if stone is None:
                continue
            if stone == Player.black:
                planes[0, r - 1, c - 1] = 1.0
            else:
                planes[1, r - 1, c - 1] = 1.0
    planes[2, :, :] = 1.0 if state.next_player == Player.black else 0.0
    return planes


# =========================
# Jiu 的复合动作 token 编解码（与 SFT 对齐）
# =========================
def move_to_sgf(current_player: Player, moves):
    # 单步或复合列表
    if not isinstance(moves, list):
        moves = [moves]
    if not moves:
        return ""

    try:
        first_move = moves[0]
        if getattr(first_move, "from_point", None) is None:
            return ""

        player = "B" if current_player == Player.black else "W"
        parts = []
        # 起点
        src_col = chr(ord('A') + first_move.from_point.col - 1)
        src_row = str(first_move.from_point.row)
        parts.append(f"{player}({src_col},{src_row})")

        # 途经点
        if getattr(first_move, "to_points", None):
            jumps = [f"-O({chr(ord('A') + p.col - 1)},{p.row})" for p in first_move.to_points]
            parts.append("".join(jumps))

        # 终点
        if getattr(first_move, "to_point", None):
            final_col = chr(ord('A') + first_move.to_point.col - 1)
            final_row = str(first_move.to_point.row)
            parts.append(f"-O({final_col},{final_row})")

        tc, fc = [], []
        total_jump_count = len(first_move.to_points) if getattr(first_move, "to_points", None) else 0
        for idx, mv in enumerate(moves[1:]):
            if getattr(mv, "is_capture", False) and getattr(mv, "point", None) is not None:
                cap_col = chr(ord('A') + mv.point.col - 1)
                cap_row = str(mv.point.row)
                captured_str = f"{'W' if player == 'B' else 'B'}({cap_col},{cap_row})"
                if idx < total_jump_count:
                    tc.append(captured_str)
                else:
                    fc.append(captured_str)
        if tc:
            parts.append(f",TC:{','.join(tc)}")
        if fc:
            parts.append(f",FC:{','.join(fc)}")
        return "".join(parts)
    except Exception:
        return ""


def sgf_to_tokens(sgf_string: str):
    tokens = []
    action_pattern = re.compile(
        r'([WB])\(([A-Za-z]+),(\d+)\)'
        r'((-O\([A-Za-z]+,\d+\))*)'
        r'(?:,TC:((?:[WB]\([A-Za-z]+,\d+\)(?:,[WB]\([A-Za-z]+,\d+\))*)))?'
        r'(?:,FC:((?:[WB]\([A-Za-z]+,\d+\)(?:,[WB]\([A-Za-z]+,\d+\))*)))?'
    )
    for m in action_pattern.finditer(sgf_string):
        player = m.group(1)
        src_col = m.group(2)
        src_row = m.group(3)
        src_token = f"{player}_{src_col}{src_row}"
        raw_moves = m.group(4) or ""
        dst_moves = [f"{k.group(1)}{k.group(2)}" for k in re.finditer(r'\(([A-Za-z]+),(\d+)\)', raw_moves)]
        dst_token = "-O_" + "-O_".join(dst_moves) if dst_moves else ""
        tc_part = m.group(5); fc_part = m.group(6)
        eat_tc = []
        if tc_part:
            eat_tc = [f"EAT_TC_{col}{row}" for _, col, row in re.findall(r'([WB])\(([A-Za-z]+),(\d+)\)', tc_part)]
        eat_fc = []
        if fc_part:
            eat_fc = [f"EAT_FC_{col}{row}" for _, col, row in re.findall(r'([WB])\(([A-Za-z]+),(\d+)\)', fc_part)]
        tok = src_token + (f"-{dst_token}" if dst_token else "")
        if eat_tc:
            tok += "-" + "-".join(eat_tc)
        if eat_fc:
            tok += "-" + "-".join(eat_fc)
        tokens.append(tok)
    return tokens


# =========================
# 把棋盘策略图投影到“复合动作集合”的 logits
# =========================
def action_logits_from_policy_map(policy_map: torch.Tensor,  # (H,W)
                                  legal_moves: List[List[Move]]) -> torch.Tensor:
    H, W = policy_map.shape
    out = []
    for mv_list in legal_moves:
        pts = []
        m0 = mv_list[0]
        if getattr(m0, "from_point", None):
            pts.append(m0.from_point)
        if getattr(m0, "to_points", None):
            pts.extend(m0.to_points)
        if getattr(m0, "to_point", None):
            pts.append(m0.to_point)
        for m in mv_list[1:]:
            if getattr(m, "point", None):
                pts.append(m.point)

        if not pts:
            out.append(torch.tensor(0.0, device=policy_map.device))
            continue

        vals = []
        for pt in pts:
            r, c = pt.row - 1, pt.col - 1
            if 0 <= r < H and 0 <= c < W:
                vals.append(policy_map[r, c])
        if len(vals) == 0:
            out.append(torch.tensor(0.0, device=policy_map.device))
        else:
            out.append(torch.stack(vals).mean())
    return torch.stack(out) if out else torch.zeros(0, device=policy_map.device)


# =========================
# LLM 先验（针对复合动作集合）
# =========================
def llm_prior_for_jiu_state(llm, tok, history_ids: List[int],
                            legal_moves: List[List[Move]],
                            current_player: Player) -> torch.Tensor:
    """
    返回 (L,) 的向量，对应 L 个合法复合动作概率。
    """
    device = next(llm.parameters()).device
    if len(history_ids) == 0:
        input_ids = torch.tensor([[tok.bos_token_id if tok.bos_token_id is not None else 0]], device=device)
    else:
        input_ids = torch.tensor([history_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = llm(input_ids).logits[:, -1, :]  # (1,V)
        probs = F.softmax(logits, dim=-1)[0]      # (V,)

    L = len(legal_moves)
    prior = torch.zeros(L, dtype=torch.float32, device=device)

    for i, mv_list in enumerate(legal_moves):
        sgf = move_to_sgf(current_player, mv_list)
        if not sgf:
            continue
        toks = sgf_to_tokens(sgf)
        if not toks:
            continue
        tid = tok.convert_tokens_to_ids([toks[0]])[0]
        if tid == tok.unk_token_id:
            continue
        prior[i] = probs[tid].float()

    s = prior.sum()
    if s > 0:
        prior = prior / s
    else:
        if L > 0:
            prior[:] = 1.0 / L
    return prior


# =========================
# MCTS 节点（久棋，索引空间：0..L-1）
# =========================
class JiuNode:
    __slots__ = [
        "state", "parent", "player",
        "P", "N", "W", "Q",
        "children", "move_from_parent", "E",
        "_legal_moves_cache", "history_ids",   # 缓存合法动作 + 历史 token
    ]

    def __init__(self, state: GameState, parent: Optional["JiuNode"],
                 legal_moves: List[List[Move]], prior: torch.Tensor,
                 history_ids: Optional[List[int]] = None):
        self.state = state
        self.parent = parent
        self.player = state.next_player

        # 缓存本节点合法复合动作（索引空间：0..L-1）
        self._legal_moves_cache: List[List[Move]] = list(legal_moves) if legal_moves is not None else []
        L = len(self._legal_moves_cache)

        # 若先验长度与 L 不一致，做一次兜底
        if isinstance(prior, torch.Tensor):
            if prior.numel() != L:
                prior = torch.full((L,), 1.0 / max(1, L), dtype=torch.float32,
                                   device=prior.device if prior.numel() > 0 else torch.device("cpu"))
        else:
            prior = torch.full((L,), 1.0 / max(1, L), dtype=torch.float32)

        self.P: Dict[int, float] = {i: float(prior[i].item() if isinstance(prior, torch.Tensor) else prior[i])
                                    for i in range(L)}
        self.N: Dict[int, int] = {i: 0 for i in range(L)}
        self.W: Dict[int, float] = {i: 0.0 for i in range(L)}
        self.Q: Dict[int, float] = {i: 0.0 for i in range(L)}
        self.children: Dict[int, "JiuNode"] = {}
        self.move_from_parent = None
        self.E: Dict[int, float] = {i: 0.0 for i in range(L)}  # SARSA(λ)
        self.history_ids = [] if history_ids is None else list(history_ids)

    def get_legal_moves(self) -> List[List[Move]]:
        return self._legal_moves_cache

    def moves(self):
        return list(self.P.keys())

    def best_child(self, c_puct: float):
        if not self.P:
            return None
        total_N = sum(self.N.values()) + 1
        best_score, best_idx = -1e9, None
        for idx in self.P.keys():
            q = self.Q.get(idx, 0.0)
            n = self.N.get(idx, 0)
            p = self.P.get(idx, 0.0)
            u = c_puct * p * math.sqrt(total_N) / (1 + n)
            score = q + u
            if score > best_score:
                best_score, best_idx = score, idx
        return best_idx


# =========================
# TD 更新（SARSA(λ) + 终局 Q-learning）
# =========================
def td_update_path_sarsa_lambda(path: List[Tuple[JiuNode, int]], gamma: float = 0.99, lam: float = 0.8):
    if not path:
        return
    # 清零资格迹
    for node, _ in path:
        for a in node.E:
            node.E[a] = 0.0

    # 逐步累积
    for t in range(len(path) - 1):
        node_t, a_t = path[t]
        node_tp1, a_tp1 = path[t + 1]
        q_next = node_tp1.Q.get(a_tp1, 0.0)
        delta = (-node_t.Q.get(a_t, 0.0)) + gamma * q_next  # 轮换对手视角
        node_t.E[a_t] = node_t.E.get(a_t, 0.0) + 1.0
        for j in range(t + 1):
            nj, aj = path[j]
            nj.Q[aj] = nj.Q.get(aj, 0.0) + delta * (gamma ** (t - j)) * (lam ** (t - j)) * nj.E.get(aj, 0.0)


def td_backup_terminal_qlearning(path: List[Tuple[JiuNode, int]],
                                 result_winner: Optional[Player],
                                 leaf_player: Player,
                                 gamma: float = 0.99):
    if result_winner is None:
        z = 0.0
    else:
        z = 1.0 if result_winner == leaf_player else -1.0

    v = z
    for node, a in reversed(path):
        if a is None or a not in node.N:
            continue
        node.N[a] += 1
        node.W[a] += v
        node.Q[a] = node.W[a] / node.N[a]
        v = -v * gamma


# =========================
# 工具：把合法动作统一成 List[List[Move]]
# =========================
def _listify_legal(state: GameState, board_size: int) -> List[List[Move]]:
    legal_moves: List[List[Move]] = []
    if hasattr(state, "get_legal_moves"):
        legal = state.get_legal_moves()
        for mv in legal:
            legal_moves.append(mv if isinstance(mv, list) else [mv])
    else:
        # 兜底（一般到不了这里）
        for r in range(1, board_size + 1):
            for c in range(1, board_size + 1):
                m = Move.play(Point(r, c))
                if state.is_valid_move(m):
                    legal_moves.append([m])
    return legal_moves


# =========================
# MCTS 一次搜索（久棋）
# - 当前叶子：用 PV 投影 + （可选）LLM 先验融合
# - 子节点：创建时为其计算 LLM 先验（基于 child_history + child_legal）
# =========================
def mcts_search_once_jiu(
    root: JiuNode,
    pvnet: JiuLightweightPVNet,
    torch_device: torch.device,
    amp_autocast,
    board_size: int,
    c_puct: float = 2.0,
    llm=None, tok=None, prior_mix: float = 0.5,
):
    # 1) Selection
    node = root
    path: List[Tuple[JiuNode, int]] = []

    while True:
        # 终局或还没展开子节点，停止下行
        if node.state.determine_winner() or not node.children:
            break
        ai = node.best_child(c_puct)
        if ai is None or ai not in node.children:
            break
        path.append((node, ai))
        node = node.children[ai]

    # 2) Expansion & Evaluation
    if not node.state.determine_winner():
        # 当前节点的合法复合动作
        legal_moves = node.get_legal_moves()
        if not legal_moves:
            legal_moves = _listify_legal(node.state, board_size)
            L = len(legal_moves)
            if L == 0:
                result = compute_game_result(node.state)
                td_backup_terminal_qlearning(path, result.winner, node.player)
                return
            node.P = {i: 1.0 / L for i in range(L)}
            node.N = {i: 0 for i in range(L)}
            node.W = {i: 0.0 for i in range(L)}
            node.Q = {i: 0.0 for i in range(L)}
            node.E = {i: 0.0 for i in range(L)}
            node._legal_moves_cache = legal_moves

        L = len(legal_moves)
        if L == 0:
            result = compute_game_result(node.state)
            td_backup_terminal_qlearning(path, result.winner, node.player)
            return

        # 估值 + 当前节点的 PV 投影先验
        planes = encode_state_planes(node.state, board_size).unsqueeze(0).to(torch_device)
        with amp_autocast():
            p_map, v_pred = pvnet(planes)  # p_map:(1,H,W), v:(1,1)
        leaf_value = float(v_pred.item())

        pv_logits = action_logits_from_policy_map(p_map[0], legal_moves)  # (L,)
        if pv_logits.numel() > 0:
            pv_prior = F.softmax(pv_logits, dim=0)
        else:
            pv_prior = torch.full((L,), 1.0 / L, dtype=torch.float32, device=planes.device)

        # 当前节点再融合一次 LLM 先验
        if llm is not None and tok is not None:
            try:
                llm_prior = llm_prior_for_jiu_state(llm, tok, node.history_ids, legal_moves, node.state.next_player)
                blended_prior = prior_mix * llm_prior + (1 - prior_mix) * pv_prior
            except Exception:
                blended_prior = pv_prior
        else:
            blended_prior = pv_prior

        if blended_prior.sum() > 0:
            blended_prior = blended_prior / blended_prior.sum()
        # 用融合先验覆盖本节点 P
        node.P = {i: float(blended_prior[i].item()) for i in range(L)}

        # 展开子节点，并为每个子节点计算 LLM 先验（仅 LLM，PV 留到它变叶子时再算）
        for i, mv_list in enumerate(legal_moves):
            st = node.state
            for m in mv_list:
                st = st.apply_move(m)

            child_legal = _listify_legal(st, board_size)
            Lc = len(child_legal)

            # 子历史：追加复合动作 token
            child_hist = node.history_ids
            if tok is not None and llm is not None:
                sgf = move_to_sgf(node.state.next_player, mv_list)
                toks = sgf_to_tokens(sgf)
                if toks:
                    tid = tok.convert_tokens_to_ids([toks[0]])[0]
                    if tid != tok.unk_token_id:
                        child_hist = child_hist + [tid]

            # 子先验：LLM（若不可用则均匀）
            if llm is not None and tok is not None and Lc > 0:
                try:
                    child_prior = llm_prior_for_jiu_state(llm, tok, child_hist, child_legal, st.next_player)
                except Exception:
                    child_prior = torch.full((Lc,), 1.0 / Lc, dtype=torch.float32, device=planes.device)
            else:
                child_prior = torch.full((Lc,), 1.0 / max(1, Lc), dtype=torch.float32, device=planes.device)

            child = JiuNode(st, node, child_legal, child_prior, history_ids=child_hist)
            node.children[i] = child

        # SARSA(λ) 路径 TD
        td_update_path_sarsa_lambda(path, gamma=0.99, lam=0.8)

        # 备份 v 值
        v = leaf_value
        for pn, ai in reversed(path):
            if ai is None or ai not in pn.N:
                continue
            pn.N[ai] += 1
            pn.W[ai] += v
            pn.Q[ai] = pn.W[ai] / pn.N[ai]
            v = -v
    else:
        # 终局：Q-learning 回传
        result = compute_game_result(node.state)
        td_backup_terminal_qlearning(path, result.winner, node.player)


# =========================
# 运行 MCTS（久棋）
# =========================
def run_mcts_jiu(
    state: GameState,
    root_prior: torch.Tensor,              # (L,)
    legal_root_moves: List[List[Move]],
    pvnet: JiuLightweightPVNet,
    torch_device: torch.device,
    amp_autocast,
    board_size: int,
    num_simulations: int = 300,
    c_puct: float = 2.0,
    llm=None, tok=None, prior_mix: float = 0.5,
    history_ids: Optional[List[int]] = None,
) -> Tuple[JiuNode, Dict[int, int]]:

    # 根节点先验兜底
    if root_prior.numel() == 0 or float(root_prior.sum().item()) <= 0:
        L = len(legal_root_moves)
        if L == 0:
            root = JiuNode(state, None, [], torch.zeros(0, device=torch_device), history_ids=(history_ids or []))
            return root, {}
        root_prior = torch.full((L,), 1.0 / L, dtype=torch.float32, device=torch_device)

    root = JiuNode(state, None, legal_root_moves, root_prior, history_ids=(history_ids or []))

    for _ in range(num_simulations):
        mcts_search_once_jiu(
            root, pvnet, torch_device, amp_autocast, board_size, c_puct,
            llm=llm, tok=tok, prior_mix=prior_mix
        )

    visit_counts = {i: root.N.get(i, 0) for i in root.P.keys()}
    return root, visit_counts


# =========================
# 主入口（供 UI 调用）
# =========================
def run_rl_jiu(
    game: str,                 # "jiu"
    board_size: int,           # 通常 14
    episodes: int,
    sims: int,
    gamma: float,
    lam: float,
    note: str,
    model_root: str,           # models/
    llm_subdir: Optional[str], # models/sft/<subdir>
    prior_mix: float,          # LLM 与 PV 融合比例
    device: str,
    save_root: str,            # models/rl
):
    assert game == "jiu", "run_rl_jiu 仅支持久棋"

    torch_device, amp_autocast, scaler, use_cuda = _get_device_and_amp(device)

    tag = f"jiu-rl-sims{sims}-epi{episodes}-{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = os.path.join(save_root, tag)
    os.makedirs(out_dir, exist_ok=True)
    yield f"[RL/Jiu] 输出目录：{out_dir}"
    yield f"[RL/Jiu] 配置：size={board_size}, episodes={episodes}, sims={sims}, device={torch_device.type}, prior_mix={prior_mix}"

    # 加载 LLM（久棋 SFT，含自定义动作 token）
    llm, tok = None, None
    if llm_subdir:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        candidate = llm_subdir.strip()
        if not os.path.isabs(candidate):
            candidate = os.path.join(model_root, "sft", candidate)
        sft_path = os.path.normpath(os.path.abspath(candidate))
        try:
            tok = AutoTokenizer.from_pretrained(sft_path, local_files_only=True)
            try:
                llm = AutoModelForCausalLM.from_pretrained(sft_path, local_files_only=True)
                loaded_from_tf = False
            except Exception:
                llm = AutoModelForCausalLM.from_pretrained(sft_path, local_files_only=True, from_tf=True)
                loaded_from_tf = True

            if tok.pad_token is None:
                if tok.eos_token is not None:
                    tok.pad_token = tok.eos_token
                elif tok.unk_token is not None:
                    tok.pad_token = tok.unk_token
                else:
                    tok.add_special_tokens({"pad_token": "[PAD]"})
            if getattr(llm.config, "pad_token_id", None) is None:
                llm.config.pad_token_id = tok.pad_token_id

            llm.resize_token_embeddings(len(tok))
            llm.to(torch_device).eval()
            src = "TensorFlow(tf_model.h5)" if loaded_from_tf else "PyTorch(pytorch_model.bin/model.safetensors)"
            yield f"[RL/Jiu] 已加载 SFT 模型先验：{os.path.basename(sft_path)}（{src}）"
        except Exception as e:
            yield f"[RL/Jiu] 警告：加载久棋 SFT 失败，先验关闭 —— {e}"
            llm, tok = None, None
    else:
        yield "[RL/Jiu] 未选择 SFT 模型，先验关闭。"

    # 策略-价值网络
    pvnet = JiuLightweightPVNet(in_ch=3, board_size=board_size).to(torch_device)
    optim = torch.optim.AdamW(pvnet.parameters(), lr=1e-3, weight_decay=1e-4)

    BATCH_SIZE = 64
    CKPT_EVERY = max(5, min(50, episodes // 10))

    for ep in range(1, episodes + 1):
        t0 = time.time()
        state = GameState.new_game(board_size)
        history_ids: List[int] = []                        # LLM 历史
        replay_states: List[torch.Tensor] = []             # (T, 3, H, W)
        replay_players: List[Player] = []                  # (T,)
        replay_legal_sets: List[List[List[Move]]] = []     # (T, L_t, ...)
        replay_pi: List[torch.Tensor] = []                 # list[(L_t,)]

        steps = 0
        max_steps = board_size * board_size * 2
        report_every = max(1, max_steps // 10)

        while (not state.determine_winner()) and steps < max_steps:
            steps += 1
            # 根合法复合动作
            legal_root = _listify_legal(state, board_size)
            L = len(legal_root)
            if L == 0:
                break

            # PV 投影到动作集合（根）
            planes = encode_state_planes(state, board_size).unsqueeze(0).to(torch_device)
            with amp_autocast():
                p_map, _ = pvnet(planes)
            pv_logits = action_logits_from_policy_map(p_map[0], legal_root)  # (L,)
            pv_prior = F.softmax(pv_logits, dim=0) if pv_logits.numel() > 0 else torch.full((L,), 1.0 / L, dtype=torch.float32, device=torch_device)

            # LLM 先验（根）
            if llm is not None and tok is not None:
                llm_prior = llm_prior_for_jiu_state(llm, tok, history_ids, legal_root, state.next_player)
                prior = prior_mix * llm_prior + (1 - prior_mix) * pv_prior
            else:
                prior = pv_prior
            if prior.sum() > 0:
                prior = prior / prior.sum()

            # 运行 MCTS（带上 llm/tok/prior_mix/history_ids）
            root, visit_counts = run_mcts_jiu(
                state=state,
                root_prior=prior,
                legal_root_moves=legal_root,
                pvnet=pvnet,
                torch_device=torch_device,
                amp_autocast=amp_autocast,
                board_size=board_size,
                num_simulations=sims,
                c_puct=2.0,
                llm=llm, tok=tok, prior_mix=prior_mix,
                history_ids=history_ids,
            )

            # 根据访问计数生成 π
            N_vec = torch.tensor([visit_counts.get(i, 0) for i in range(L)],
                                 dtype=torch.float32, device=torch_device)
            if N_vec.sum() > 0:
                pi = N_vec / N_vec.sum()
            else:
                pi = torch.full((L,), 1.0 / L, dtype=torch.float32, device=torch_device)

            # 记录样本（策略+价值）
            replay_states.append(encode_state_planes(state, board_size))
            replay_players.append(state.next_player)
            replay_legal_sets.append(legal_root)
            replay_pi.append(pi.detach().cpu())

            # 抽样执行复合动作
            a_idx = torch.multinomial(pi, 1).item()
            mv_list = legal_root[a_idx]

            st_next = state
            for m in mv_list:
                st_next = st_next.apply_move(m)
            state = st_next

            # 更新 LLM 历史
            if llm is not None and tok is not None:
                sgf = move_to_sgf(replay_players[-1], mv_list)
                toks = sgf_to_tokens(sgf)
                if toks:
                    tid = tok.convert_tokens_to_ids([toks[0]])[0]
                    if tid != tok.unk_token_id:
                        history_ids.append(tid)

            if steps % report_every == 0:
                yield f"[RL/Jiu][Ep {ep}] 自对弈进度：{steps}/{max_steps} (~{int(100 * steps / max_steps)}%)"

        # 终局 -> value targets
        result = compute_game_result(state)
        if result.winner is None:
            z_final = 0.0
        else:
            z_final = 1.0

        targets_v: List[List[float]] = []
        for p in replay_players:
            if result.winner is None:
                z = 0.0
            else:
                z = 1.0 if p == result.winner else -1.0
            targets_v.append([z])

        if len(replay_states) == 0:
            yield f"[RL/Jiu][Ep {ep}] 无样本，跳过。"
            continue

        X = torch.stack(replay_states, dim=0).to(torch_device)          # (T, 3, H, W)
        Y_v = torch.tensor(targets_v, dtype=torch.float32, device=torch_device)  # (T, 1)
        Pis = [p.to(torch_device) for p in replay_pi]                   # list[(L_t,)]

        # 训练：policy + value
        pvnet.train()
        total_pl, total_vl, total_loss, batches = 0.0, 0.0, 0.0, 0
        for i in range(0, X.size(0), BATCH_SIZE):
            xb = X[i:i + BATCH_SIZE]
            vb = Y_v[i:i + BATCH_SIZE]
            legal_batch = replay_legal_sets[i:i + BATCH_SIZE]
            pi_batch = Pis[i:i + BATCH_SIZE]

            optim.zero_grad(set_to_none=True)
            with amp_autocast():
                p_maps, v_pred = pvnet(xb)  # p_maps:(B,H,W), v_pred:(B,1)

                # policy：对每个样本把 p_map 投影到其动作集合
                plosses = []
                for b in range(p_maps.size(0)):
                    logits_b = action_logits_from_policy_map(p_maps[b], legal_batch[b])  # (L_b,)
                    if logits_b.numel() == 0:
                        continue
                    logp_b = F.log_softmax(logits_b, dim=0)
                    plosses.append(-(pi_batch[b] * logp_b).sum())
                policy_loss = torch.stack(plosses).mean() if plosses else torch.tensor(0.0, device=torch_device)

                value_loss = F.mse_loss(v_pred, vb)
                loss = policy_loss + value_loss

            loss.backward()
            optim.step()
            total_pl += float(policy_loss.detach().cpu().item())
            total_vl += float(value_loss.detach().cpu().item())
            total_loss += float(loss.detach().cpu().item())
            batches += 1

        elapsed = time.time() - t0
        yield (f"[RL/Jiu][Ep {ep}] 完成：steps={steps}, "
               f"loss={total_loss / max(1, batches):.4f}, "
               f"pl={total_pl / max(1, batches):.4f}, "
               f"vl={total_vl / max(1, batches):.4f}, "
               f"time={elapsed:.1f}s, winner={result.winner}")

        if ep % CKPT_EVERY == 0:
            ckpt = os.path.join(out_dir, f"ckpt_ep{ep}")
            os.makedirs(ckpt, exist_ok=True)
            torch.save(pvnet.state_dict(), os.path.join(ckpt, "pvnet.pt"))
            yield f"[RL/Jiu] 已保存检查点：{ckpt}"

    final_dir = os.path.join(out_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    torch.save(pvnet.state_dict(), os.path.join(final_dir, "pvnet.pt"))
    yield f"[RL/Jiu] 训练完成 ✅ 已保存：{final_dir}"
