# backend/arena/match.py
# -*- coding: utf-8 -*-
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

import torch

# 统一棋种 API（Go / Gomoku）
from ..rl.train import get_game_api, legal_moves_and_mask, index_to_move

# Agent 抽象与工厂
from .agents import BaseAgent, AgentOutput, scan_models, build_agent


@dataclass
class MatchConfig:
    game: str                 # "go" | "gomoku" | （未来可扩： "jiu"）
    board_size: int
    black_kind: str           # "none" | "sft:xxx" | "rl:.../final/pvnet.pt" | （可扩："random"...）
    white_kind: str
    sims: int = 128
    prior_mix: float = 0.7
    device: str = "cpu"
    model_root: str = "models"


class MatchEngine:
    def __init__(self, cfg: MatchConfig):
        self.cfg = cfg
        self.api = get_game_api(cfg.game)  # 统一访问入口（GameState/Move/Point/Player/compute_game_result）

        # 初始化对局
        self.state = self.api.GameState.new_game(cfg.board_size)

        # 构建两侧 Agent（None 表示人类）
        self.black_agent: Optional[BaseAgent] = build_agent(
            cfg.black_kind, cfg.game, cfg.board_size, cfg.model_root, cfg.device, cfg.sims, cfg.prior_mix
        )
        self.white_agent: Optional[BaseAgent] = build_agent(
            cfg.white_kind, cfg.game, cfg.board_size, cfg.model_root, cfg.device, cfg.sims, cfg.prior_mix
        )

        # 历史（用于悔棋重放）
        self.move_history: List[Tuple[object, object]] = []  # [(player, move)]
        self.last_eval: Optional[AgentOutput] = None

    # ---------- 查询 ----------
    def who_to_play(self):
        return self.state.next_player

    def is_human_turn(self) -> bool:
        """当前轮到人类则 True。"""
        p = self.who_to_play()
        if p == self.api.Player.black:
            return self.black_agent is None
        else:
            return self.white_agent is None

    def is_over(self) -> bool:
        """采用各棋种自身的 is_over() 或通过 winner() 判定。"""
        if hasattr(self.state, "is_over"):
            try:
                return bool(self.state.is_over())
            except Exception:
                pass
        # 兜底：依据 winner()
        w = getattr(self.state, "winner", None)
        w = w() if callable(w) else None
        return w is not None

    def result_text(self) -> str:
        res = self.api.compute_game_result(self.state)
        if res.winner is None:
            return "未分胜负"
        side = "黑" if res.winner == self.api.Player.black else "白"
        return f"{side}胜"

    # ---------- 内部：统一落子并通知 agent ----------
    def _apply_move(self, mv):
        before = self.state
        self.state = self.state.apply_move(mv)
        self.move_history.append((before.next_player, mv))
        # 通知对应 agent（便于维护 LLM 历史等）
        agent = self.black_agent if before.next_player == self.api.Player.black else self.white_agent
        if agent is not None:
            agent.on_move_applied(mv, self.state)

    # ---------- 人类只落一步 ----------
    def human_play(self, row: int, col: int) -> Dict:
        """
        人类在 (row,col) 落子（1-based）。只应用这一步，不做自动回手。
        """
        if self.is_over():
            return {"ok": False, "msg": "对局已结束", "board": self.export_board()}

        if not self.is_human_turn():
            return {"ok": False, "msg": "当前轮到模型回合，请点击“AI 落子”", "board": self.export_board()}

        mv = self.api.Move.play(self.api.Point(row, col))
        if not self.state.is_valid_move(mv):
            return {"ok": False, "msg": f"非法落子：({row},{col})", "board": self.export_board()}

        p = self.who_to_play()
        self._apply_move(mv)

        # 若此手后终局，附上结果
        done_msg = ""
        if self.is_over():
            done_msg = f" | 终局：{self.result_text()}"

        color_cn = "黑" if p == self.api.Player.black else "白"
        return {
            "ok": True,
            "human": {"move": (row, col), "player": color_cn, "msg": f"人类（{color_cn}）在 ({row},{col}) 落子。"},
            "msg": f"人类（{color_cn}）在 ({row},{col}) 落子。{done_msg}",
            "board": self.export_board(),
        }

    # ---------- 当前若轮到模型，就只走“一步” ----------
    def side_is_model_to_play(self) -> bool:
        """True 表示当前行动方是模型；False 表示人类。"""
        if self.is_over():
            return False
        p = self.who_to_play()
        if p == self.api.Player.black:
            return self.black_agent is not None
        else:
            return self.white_agent is not None

    def _build_topk_from(self, pi: Optional[torch.Tensor], legal_idxs: List[int], k: int = 5) -> List[Tuple[str, float, int]]:
        """
        基于分布 pi（全局 N 向量）与合法集合，构造 Top-K；
        若 pi 为空或在合法集上全 0，则退化为“合法着均匀分布”。
        """
        if legal_idxs is None:
            legal_idxs = []
        if not legal_idxs:
            return []

        out: List[Tuple[str, float, int]] = []

        if isinstance(pi, torch.Tensor):
            p = pi.detach().float().cpu()
            sub = p[legal_idxs] if legal_idxs else torch.tensor([], dtype=torch.float32)
            if sub.numel() > 0 and float(sub.sum().item()) > 0.0:
                sub = sub / sub.sum()
                kk = min(k, len(legal_idxs))
                vals, order = torch.topk(sub, kk)
                for v, oi in zip(vals.tolist(), order.tolist()):
                    idx = legal_idxs[oi]
                    m = index_to_move(idx, self.cfg.board_size, api=self.api, state=self.state)
                    out.append((f"({m.point.row},{m.point.col})", float(v), idx))
                return out

        # 均匀兜底
        prob = 1.0 / len(legal_idxs)
        for idx in legal_idxs[:k]:
            m = index_to_move(idx, self.cfg.board_size, api=self.api, state=self.state)
            out.append((f"({m.point.row},{m.point.col})", prob, idx))
        return out

    def model_play_if_needed(self) -> Optional[Dict]:
        """
        若当前轮到模型，则计算并落子一步；若轮到人类/已结束，返回 None。
        统一构造 step['topk']；当 agent 未提供或 pi 全 0 时，用“合法着均匀分布”兜底。
        """
        if self.is_over():
            return {"msg": "对局已结束", "board": self.export_board()}

        if not self.side_is_model_to_play():
            return None  # 人类回合

        p = self.who_to_play()
        agent = self.black_agent if p == self.api.Player.black else self.white_agent
        assert agent is not None, "side_is_model_to_play 为真时，agent 不应为空"

        # 先拿“当前局面”的合法集合（用于 topk 更合理，也能做兜底）
        _, _, legal_idxs = legal_moves_and_mask(self.state, self.cfg.board_size, api=self.api)

        rec: AgentOutput = agent.propose(self.state)
        mv = rec.move

        # 兜底：无着法或非法就从合法集合挑一个
        if (mv is None) or (not self.state.is_valid_move(mv)):
            if not legal_idxs:
                return {"msg": "无合法着法", "board": self.export_board(), "value": float(rec.value) if rec.value is not None else None, "topk": []}
            mv = index_to_move(legal_idxs[0], self.cfg.board_size, api=self.api, state=self.state)

        # 计算 Top-K：
        # 1) 若 agent 提供了 topk，规范化其坐标展示；
        # 2) 否则利用 rec.pi 在合法集合上挑 topk；
        # 3) 仍不行则均匀分布兜底。
        topk_list: List[Tuple[str, float, int]] = []
        if isinstance(rec.topk, list) and len(rec.topk) > 0:
            for item in rec.topk:
                try:
                    coord_label, prob, idx = item
                except Exception:
                    # 兼容只有 (idx, prob) 的形态
                    idx, prob = item
                    coord_label = None
                if isinstance(idx, int):
                    m = index_to_move(idx, self.cfg.board_size, api=self.api, state=self.state)
                    topk_list.append((f"({m.point.row},{m.point.col})", float(prob), int(idx)))
                else:
                    # 没有 idx 就只能展示文本
                    topk_list.append((str(coord_label), float(prob), -1))
        if not topk_list:
            topk_list = self._build_topk_from(getattr(rec, "pi", None), legal_idxs, k=5)

        # 真正落子
        self._apply_move(mv)

        # 若此手后终局，附上结果
        done_msg = ""
        if self.is_over():
            done_msg = f" | 终局：{self.result_text()}"

        color_cn = "黑" if p == self.api.Player.black else "白"
        out = {
            "msg": f"模型（{color_cn}）落子：({mv.point.row},{mv.point.col}){done_msg}",
            "board": self.export_board(),
            "value": float(rec.value) if rec.value is not None else None,
            "topk": topk_list,
        }
        return out

    def model_play_once(self, sims: int = None, topk: int = None):
        """
        为了兼容老的 backend 调用：让当前若是模型回合就只走一步。
        - sims 参数在当前实现里不用（搜索次数由 Agent 内部控制），保留签名以兼容。
        - topk 若给定，就把返回的 topk 列表截断到该长度。
        """
        out = self.model_play_if_needed()
        if isinstance(out, dict) and out.get("topk") and topk is not None:
            out["topk"] = out["topk"][:int(topk)]
        return out

    # ---------- 导出给前端 ----------
    def export_board(self) -> Dict:
        """
        统一导出棋盘载荷：
          - size: 棋盘尺寸
          - to_play: "black"/"white"
          - stones: [{"row":r,"col":c,"color":"black"/"white"}, ...]
          - done: 是否终局
          - winner: "black"/"white"/None
          - game: 棋种（UI 若需特殊标记用）
        """
        stones = []
        bsz = self.cfg.board_size
        # 用统一访问方式读取棋盘
        for r in range(1, bsz + 1):
            for c in range(1, bsz + 1):
                p = self.api.Point(r, c)
                s = self.state.board.get(p)
                if s is None:
                    continue
                stones.append({
                    "row": r, "col": c,
                    "color": "black" if s == self.api.Player.black else "white"
                })

        res = self.api.compute_game_result(self.state)
        winner = None
        if res.winner is not None:
            winner = "black" if res.winner == self.api.Player.black else "white"

        return {
            "game": self.cfg.game,
            "size": bsz,
            "to_play": "black" if self.state.next_player == self.api.Player.black else "white",
            "stones": stones,
            "done": self.is_over(),
            "winner": winner,
        }

    # ---------- 操作 ----------
    def undo(self, k: int = 1) -> Dict:
        """悔棋 k 手：通过回放历史重建。"""
        if k <= 0 or len(self.move_history) == 0:
            return {"ok": False, "msg": "无可悔步", "board": self.export_board()}

        bsz = self.cfg.board_size
        hist = self.move_history[:-k] if k <= len(self.move_history) else []
        self.state = self.api.GameState.new_game(bsz)
        self.move_history = []
        for pl, mv in hist:
            self._apply_move(mv)
        return {"ok": True, "msg": f"已悔棋 {k} 手", "board": self.export_board()}

    def resign(self) -> Dict:
        if self.is_over():
            return {"ok": False, "msg": "对局已结束", "board": self.export_board()}
        # 当前行动方认输
        winner = self.state.next_player.other
        side_cn = "黑" if self.state.next_player == self.api.Player.black else "白"
        win_cn = "黑" if winner == self.api.Player.black else "白"
        return {
            "ok": True,
            "msg": f"认输：{side_cn}认输，{win_cn}胜",
            "board": self.export_board(),
        }


# ---------- 对外工具 ----------
def list_available_models(model_root: str) -> Dict[str, List[str]]:
    """
    返回 {"merged": [...], "sft": [...], "rl": [...]}
    供 UI “刷新模型列表” 按钮使用。
    """
    scanned = scan_models(model_root)
    merged = ["none"] + [f"sft:{n}" for n in scanned["sft"]] + [f"rl:{p}" for p in scanned["rl"]]
    return {"merged": merged, "sft": scanned["sft"], "rl": scanned["rl"]}
