# backend/arena/jiu_agents.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import os
import math
import torch
import torch.nn.functional as F

# —— 复用通用 AgentOutput 定义，保持和现有 UI/后端一致 —— #
try:
    from .agents import AgentOutput
except Exception:
    @dataclass
    class AgentOutput:
        move: Optional[Any] = None   # 对于久棋是 Move 或复合动作的“首步”落点
        value: Optional[float] = None
        policy: Optional[List[Tuple[Tuple[int, int], float]]] = None  # ((r,c), prob)
        topk: Optional[List[Tuple[str, float, int]]] = None           # (coord_str, prob, idx)

# —— 久棋：复用你已有的 dlgo 规则 —— #
from ..jiu.dlgo.goboard import GameState, Move
from ..jiu.dlgo.gotypes import Player, Point

# —— 复用你写好的 RL 侧实现（编码/先验/MCTS/网络等） —— #
from ..rl.train_jiu import (
    encode_state_planes,
    llm_prior_for_jiu_state,
    run_mcts_jiu,
    JiuLightweightPVNet,
    _get_device_and_amp
)

# 用于列合法“复合动作”列表
def _listify_legal(state: GameState, board_size: int) -> List[List[Move]]:
    legal_moves: List[List[Move]] = []
    if hasattr(state, "get_legal_moves"):
        legal = state.get_legal_moves()
        for mv in legal:
            legal_moves.append(mv if isinstance(mv, list) else [mv])
    else:
        # 兜底：尝试将所有点作为“单步”——通常不会走到
        for r in range(1, board_size + 1):
            for c in range(1, board_size + 1):
                m = Move.play(Point(r, c))
                try:
                    if state.is_valid_move(m):
                        legal_moves.append([m])
                except Exception:
                    pass
    return legal_moves


# -------------------------
# Agent 基类（久棋）
# -------------------------
class BaseJiuAgent:
    def __init__(self, board_size: int, device: str = "cpu", sims: int = 300, prior_mix: float = 0.7):
        self.board_size = int(board_size)
        self.device_str = device
        self.sims = int(sims)
        self.prior_mix = float(prior_mix)
        self._hist_ids: List[int] = []  # 若你要做基于 token 的历史，可以在 propose 内维护

        self.torch_device, self.amp_autocast, self.scaler, self.use_cuda = _get_device_and_amp(self.device_str)

    def on_move_applied(self, move: Move, new_state: GameState):
        """可用于维护 LLM 历史；目前久棋 token 历史在 rl.train_jiu 中实现，这里按需扩展。"""
        return

    def propose(self, state: GameState) -> AgentOutput:
        raise NotImplementedError


# -------------------------
# 1) 纯随机 Agent（作为后备）
# -------------------------
class JiuRandomAgent(BaseJiuAgent):
    def propose(self, state: GameState) -> AgentOutput:
        import random
        legal = _listify_legal(state, self.board_size)
        if not legal:
            return AgentOutput(move=None, value=0.0, policy=[], topk=[])
        mv_list = random.choice(legal)
        mv = mv_list[0] if mv_list else None
        pol_pairs = []
        if legal:
            p = 1.0 / len(legal)
            for ml in legal[: min(10, len(legal))]:
                if not ml:
                    continue
                m0 = ml[0]
                pol_pairs.append(((m0.point.row, m0.point.col), p))
        return AgentOutput(move=mv, value=0.0, policy=pol_pairs, topk=[
            (f"({r},{c})", 1.0 / len(legal), -1) for ((r, c), _) in pol_pairs
        ])


# -------------------------
# 2) SFT(LM) 先验 + 价值网络 + MCTS
# - 如果没有 RL 的 value net checkpoint，也能工作（用随机初始化 vnet）
# - 如果有 RL 的 value net（valuenet.pt），可通过 kind= "rl:<dir>" 指向
# -------------------------
class JiuMCTSAgent(BaseJiuAgent):
    def __init__(self, board_size: int, device: str, sims: int, prior_mix: float,
                 llm_tok=None, llm_model=None, value_net_path: Optional[str] = None):
        super().__init__(board_size=board_size, device=device, sims=sims, prior_mix=prior_mix)
        self.tok = llm_tok
        self.llm = llm_model
        # 价值网络
        self.vnet = JiuLightweightPVNet(in_ch=3, board_size=board_size).to(self.torch_device)
        if value_net_path:
            try:
                sd = torch.load(value_net_path, map_location=self.torch_device)
                self.vnet.load_state_dict(sd, strict=False)
            except Exception as e:
                print(f"[JiuAgent] 加载 value net 失败：{e}（将使用随机初始化）")
        self.vnet.eval()

    def _root_prior(self, state: GameState, legal_root: List[List[Move]]) -> torch.Tensor:
        L = len(legal_root)
        if L == 0:
            return torch.zeros(0, dtype=torch.float32, device=self.torch_device)
        if self.llm is None or self.tok is None:
            return torch.full((L,), 1.0 / L, dtype=torch.float32, device=self.torch_device)
        try:
            prior = llm_prior_for_jiu_state(self.llm, self.tok, [], legal_root, state.next_player)
            if prior.numel() != L:
                return torch.full((L,), 1.0 / L, dtype=torch.float32, device=self.torch_device)
            if float(prior.sum().item()) <= 0:
                return torch.full((L,), 1.0 / L, dtype=torch.float32, device=self.torch_device)
            return prior.to(self.torch_device)
        except Exception:
            return torch.full((L,), 1.0 / L, dtype=torch.float32, device=self.torch_device)

    @torch.no_grad()
    def propose(self, state: GameState) -> AgentOutput:
        # 根合法复合动作
        legal_root = _listify_legal(state, self.board_size)
        L = len(legal_root)
        if L == 0:
            return AgentOutput(move=None, value=0.0, policy=[], topk=[])

        # LLM 先验
        root_prior = self._root_prior(state, legal_root)  # (L,)

        # MCTS
        root, visit_counts = run_mcts_jiu(
            state=state,
            root_prior=root_prior,
            legal_root_moves=legal_root,
            pvnet=self.vnet,
            torch_device=self.torch_device,
            amp_autocast=self.amp_autocast,
            board_size=self.board_size,
            num_simulations=self.sims,
            c_puct=2.0
        )
        # 选择动作（最大访问计数）
        vc_list = [(i, visit_counts.get(i, 0)) for i in range(L)]
        vc_list.sort(key=lambda x: x[1], reverse=True)
        best_idx = vc_list[0][0]
        mv_list = legal_root[best_idx]
        mv = mv_list[0] if mv_list else None

        # 估值
        planes = encode_state_planes(state, self.board_size).unsqueeze(0).to(self.torch_device)
        v = float(self.vnet(planes).item())

        # top-k 展示（根的访问计数）
        denom = sum(v for _, v in vc_list) or 1
        topk = []
        for i, cnt in vc_list[:10]:
            m0 = legal_root[i][0]
            prob = cnt / denom
            topk.append((f"({m0.point.row},{m0.point.col})", float(prob), i))

        # policy（给 UI 字段）
        policy_pairs = []
        for i, cnt in vc_list[:10]:
            m0 = legal_root[i][0]
            prob = cnt / denom
            policy_pairs.append(((m0.point.row, m0.point.col), float(prob)))

        return AgentOutput(move=mv, value=v, policy=policy_pairs, topk=topk)


# -------------------------
# 3) 工厂：把 "none" / "sft:<name>" / "rl:<dir>" 解析为久棋 Agent
# -------------------------
def build_jiu_agent(kind: str, model_root: str, board_size: int, device: str = "cpu",
                    sims: int = 300, prior_mix: float = 0.7) -> Optional[BaseJiuAgent]:
    """
    kind:
      - "none"                     -> 返回 None（表示人类）
      - "sft:<subdir>"             -> 使用 models/sft/<subdir> 的 LM 先验 + 随机初始化 ValueNet + MCTS
      - "rl:<run_dir>"             -> 使用 <run_dir>/valuenet.pt，如果能找到 SFT 目录也会结合 LLM 先验
    """
    if not kind or kind == "none":
        return None

    if kind.startswith("sft:"):
        sub = kind[len("sft:"):]
        sft_path = sub if os.path.isabs(sub) else os.path.join(model_root, "sft", sub)
        tok = llm = None
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            tok = AutoTokenizer.from_pretrained(sft_path, local_files_only=True)
            try:
                llm = AutoModelForCausalLM.from_pretrained(sft_path, local_files_only=True)
            except Exception:
                llm = AutoModelForCausalLM.from_pretrained(sft_path, local_files_only=True, from_tf=True)
            # pad_token 兜底
            if tok.pad_token is None:
                if tok.eos_token is not None:
                    tok.pad_token = tok.eos_token
                else:
                    tok.add_special_tokens({"pad_token": "[PAD]"})
            if getattr(llm.config, "pad_token_id", None) is None:
                llm.config.pad_token_id = tok.pad_token_id
            llm.resize_token_embeddings(len(tok))
        except Exception as e:
            print(f"[JiuAgent] 加载 SFT 失败：{e}（将无先验）")
            tok = llm = None
        return JiuMCTSAgent(board_size=board_size, device=device, sims=sims, prior_mix=prior_mix,
                            llm_tok=tok, llm_model=llm, value_net_path=None)

    if kind.startswith("rl:"):
        run_dir = kind[len("rl:"):]
        run_dir = run_dir if os.path.isabs(run_dir) else os.path.join(model_root, run_dir)
        # 价值网络权重
        vnet_path = None
        for cand in ["valuenet.pt", os.path.join("final", "valuenet.pt")]:
            p = os.path.join(run_dir, cand)
            if os.path.isfile(p):
                vnet_path = p
                break
        # 若 run_dir 嵌入了 sft 信息，也可以加载 llm 先验；这里保守不做自动推断
        return JiuMCTSAgent(board_size=board_size, device=device, sims=sims, prior_mix=prior_mix,
                            llm_tok=None, llm_model=None, value_net_path=vnet_path)

    # 其他 -> 随机
    return JiuRandomAgent(board_size=board_size, device=device, sims=sims, prior_mix=prior_mix)
