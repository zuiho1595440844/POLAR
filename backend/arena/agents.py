# -*- coding: utf-8 -*-
"""
真实对弈 Agent 实现：
- RLAgent   : 使用 LightweightPVNet (policy+value) 的强化学习权重
- SFTAgent  : 使用监督微调的 LLM 作为先验 + 轻量 PV 评估（可替换为训练得到的更强评估）
- scan_models / build_agent : 模型扫描与实例化
"""
import os
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from ..rl.train import (
    LightweightPVNet,
    encode_state_planes,
    legal_moves_and_mask,
    index_to_move,
    move_to_token_go,
    move_to_token_gomoku,
    top_k_logits,
    get_game_api,
)

# =========================
# 公共结构
# =========================
@dataclass
class AgentOutput:
    move: Optional[object]
    value: float
    topk: List[Tuple[str, float, int]]
    pi: torch.Tensor
    info: Dict


class BaseAgent:
    def propose(self, state) -> AgentOutput:
        raise NotImplementedError

    def on_move_applied(self, move, state_after):
        pass


# =========================
# RL Agent（PVNet）
# =========================
class RLAgent(BaseAgent):
    def __init__(self, game: str, board_size: int, ckpt_path: str,
                 device: str = "cpu", sims: int = 128):
        self.game = game
        self.board_size = board_size
        self.device = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        self.sims = sims

        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"[RLAgent] 未找到权重文件: {ckpt_path}")

        self.api = get_game_api(game)

        self.net = LightweightPVNet(in_ch=3, board_size=board_size).to(self.device)
        self.net.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.net.eval()

    def propose(self, state) -> AgentOutput:
        planes = encode_state_planes(state, self.board_size, api=self.api).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, v = self.net(planes)
            probs = F.softmax(logits[0], dim=-1)
            value = float(v.item())

        _, mask, idxs = legal_moves_and_mask(state, self.board_size, api=self.api)
        pi = torch.where(mask.to(probs.device), probs, torch.zeros_like(probs))
        if float(pi.sum().item()) > 0:
            pi = pi / pi.sum()

        if not idxs:
            return AgentOutput(move=None, value=value, topk=[], pi=pi, info={"note": "no-legal"})

        best_idx = int(pi.argmax().item())
        move = index_to_move(best_idx, self.board_size, api=self.api, state=state)

        k = min(5, len(idxs))
        # 注意：先在 idxs 子集上取 topk
        import torch as _t
        sub = pi[_t.tensor(idxs, dtype=_t.long, device=pi.device)]
        vals, order = _t.topk(sub, k)
        topk: List[Tuple[str, float, int]] = []
        for vprob, oi in zip(vals.tolist(), order.tolist()):
            idx = idxs[oi]
            m = index_to_move(idx, self.board_size, api=self.api, state=state)
            topk.append((f"({m.point.row},{m.point.col})", float(vprob), idx))

        return AgentOutput(move=move, value=value, topk=topk, pi=pi, info={})


# =========================
# SFT Agent（LLM 先验 + 轻 PV 评估）
# =========================
class SFTAgent(BaseAgent):
    def __init__(self, game: str, board_size: int, sft_dir: str,
                 device: str = "cpu", prior_mix: float = 0.7):
        self.game = game
        self.board_size = board_size
        self.prior_mix = prior_mix
        self.device = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")

        if not os.path.isdir(sft_dir):
            raise FileNotFoundError(f"[SFTAgent] 模型目录不存在: {sft_dir}")

        self.api = get_game_api(game)

        self.tok = AutoTokenizer.from_pretrained(sft_dir, local_files_only=True)
        try:
            self.llm = AutoModelForCausalLM.from_pretrained(sft_dir, local_files_only=True)
        except EnvironmentError:
            tf_path = os.path.join(sft_dir, "tf_model.h5")
            if os.path.isfile(tf_path):
                self.llm = AutoModelForCausalLM.from_pretrained(sft_dir, from_tf=True, local_files_only=True)
            else:
                raise

        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token or self.tok.unk_token or "[PAD]"
        if getattr(self.llm.config, "pad_token_id", None) is None:
            self.llm.config.pad_token_id = self.tok.pad_token_id
        self.llm.resize_token_embeddings(len(self.tok))
        self.llm.to(self.device).eval()

        self.pv = LightweightPVNet(in_ch=3, board_size=board_size).to(self.device)
        self.pv.eval()

        self.history_ids: List[int] = []

    def propose(self, state) -> AgentOutput:
        planes = encode_state_planes(state, self.board_size, api=self.api).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits_pv, v = self.pv(planes)
            pv_policy = F.softmax(logits_pv[0], dim=-1)
            value = float(v.item())

        _, mask, idxs = legal_moves_and_mask(state, self.board_size, api=self.api)
        pv_prior = torch.where(mask.to(pv_policy.device), pv_policy, torch.zeros_like(pv_policy))
        if float(pv_prior.sum().item()) > 0:
            pv_prior = pv_prior / pv_prior.sum()

        N = self.board_size * self.board_size
        if self.history_ids:
            input_ids = torch.tensor([self.history_ids], dtype=torch.long, device=self.device)
        else:
            bos = self.tok.bos_token_id or self.tok.pad_token_id or 0
            input_ids = torch.tensor([[bos]], dtype=torch.long, device=self.device)

        with torch.no_grad():
            l_logits = self.llm(input_ids).logits[:, -1, :]
            l_logits = top_k_logits(l_logits, k=64)
            l_probs = F.softmax(l_logits, dim=-1)[0]

        if l_probs.numel() >= 2 * N:
            if self.api.is_black(state.next_player):
                llm_prior_full = l_probs[:N]
            else:
                llm_prior_full = l_probs[N:2 * N]
            llm_prior_full = torch.where(mask.to(llm_prior_full.device), llm_prior_full,
                                         torch.zeros_like(llm_prior_full))
            if float(llm_prior_full.sum().item()) > 0:
                llm_prior_full = llm_prior_full / llm_prior_full.sum()
            pi = self.prior_mix * llm_prior_full + (1 - self.prior_mix) * pv_prior
        else:
            pi = pv_prior

        if float(pi.sum().item()) > 0:
            pi = pi / pi.sum()

        if not idxs:
            return AgentOutput(move=None, value=value, topk=[], pi=pi, info={"note": "no-legal"})

        best_idx = int(pi.argmax().item())
        move = index_to_move(best_idx, self.board_size, api=self.api, state=state)

        import torch as _t
        sub = pi[_t.tensor(idxs, dtype=_t.long, device=pi.device)]
        vals, order = _t.topk(sub, min(5, len(idxs)))
        topk: List[Tuple[str, float, int]] = []
        for vprob, oi in zip(vals.tolist(), order.tolist()):
            idx = idxs[oi]
            m = index_to_move(idx, self.board_size, api=self.api, state=state)
            topk.append((f"({m.point.row},{m.point.col})", float(vprob), idx))

        return AgentOutput(move=move, value=value, topk=topk, pi=pi, info={})

    def on_move_applied(self, move, state_after):
        np = state_after.next_player
        mover = self.api.Player.black if self.api.is_white(np) else self.api.Player.white
        if self.game == "go":
            tok = move_to_token_go(move, mover, self.board_size, api=self.api)
        else:
            tok = move_to_token_gomoku(move, mover, self.board_size, api=self.api)
        self.history_ids.append(tok)


# =========================
# 模型扫描 / 构建
# =========================
def scan_models(model_root: str) -> Dict[str, List[str]]:
    out = {"sft": [], "rl": []}
    sft_root = os.path.join(model_root, "sft")
    rl_root = os.path.join(model_root, "rl")

    if os.path.isdir(sft_root):
        for d in sorted(os.listdir(sft_root)):
            dp = os.path.join(sft_root, d)
            if not os.path.isdir(dp):
                continue
            has_cfg = os.path.isfile(os.path.join(dp, "config.json"))
            has_w = any(os.path.isfile(os.path.join(dp, f))
                        for f in ["pytorch_model.bin", "model.safetensors", "tf_model.h5"])
            if has_cfg and has_w:
                out["sft"].append(d)

    if os.path.isdir(rl_root):
        for root, dirs, files in os.walk(rl_root):
            if "final" in root and "pvnet.pt" in files:
                ck = os.path.join(root, "pvnet.pt")
                out["rl"].append(os.path.relpath(ck, rl_root))

    return out


def build_agent(kind: str,
                game: str,
                board_size: int,
                model_root: str,
                device: str,
                sims: int = 128,
                prior_mix: float = 0.7) -> Optional[BaseAgent]:
    if kind == "none":
        return None
    if kind.startswith("rl:"):
        rel = kind.split("rl:", 1)[1]
        ckpt = os.path.join(model_root, "rl", rel)
        return RLAgent(game, board_size, ckpt, device=device, sims=sims)
    if kind.startswith("sft:"):
        name = kind.split("sft:", 1)[1]
        sft_dir = os.path.join(model_root, "sft", name)
        return SFTAgent(game, board_size, sft_dir, device=device, prior_mix=prior_mix)
    raise ValueError(f"unknown agent kind: {kind}")
