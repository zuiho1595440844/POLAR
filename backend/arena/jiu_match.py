# backend/arena/jiu_match.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Any

from ..jiu.dlgo.goboard import GameState, Move
from ..jiu.dlgo.gotypes import Player, Point
from ..jiu.dlgo.scoring import compute_game_result

from .jiu_agents import BaseJiuAgent, build_jiu_agent, AgentOutput

# —— 公共的小工具：列合法复合动作 —— #
def _listify_legal(state: GameState, board_size: int) -> List[List[Move]]:
    legal_moves: List[List[Move]] = []
    if hasattr(state, "get_legal_moves"):
        legal = state.get_legal_moves()
        for mv in legal:
            legal_moves.append(mv if isinstance(mv, list) else [mv])
    return legal_moves


@dataclass
class JiuMatchConfig:
    board_size: int
    black_kind: str       # "none" | "sft:<subdir>" | "rl:<dir>"
    white_kind: str
    sims: int = 300
    prior_mix: float = 0.7
    device: str = "cpu"
    model_root: str = "models"


class JiuMatchEngine:
    """
    久棋的“对战引擎”，接口尽量与现有的 MatchEngine 相似，以便 UI/Backend 复用：
      - who_to_play() -> Player
      - is_human_turn() -> bool
      - is_over() -> bool
      - human_play(row, col) -> Dict
      - model_play_if_needed() -> Optional[Dict]
      - model_play_once(sims=None, topk=None) -> Dict
      - export_board() -> Dict
      - undo(k=1) -> Dict
    """
    def __init__(self, cfg: JiuMatchConfig):
        self.cfg = cfg
        self.state: GameState = GameState.new_game(cfg.board_size)

        # Agent：None 表示人类
        self.black_agent: Optional[BaseJiuAgent] = build_jiu_agent(
            cfg.black_kind, model_root=cfg.model_root, board_size=cfg.board_size,
            device=cfg.device, sims=cfg.sims, prior_mix=cfg.prior_mix
        )
        self.white_agent: Optional[BaseJiuAgent] = build_jiu_agent(
            cfg.white_kind, model_root=cfg.model_root, board_size=cfg.board_size,
            device=cfg.device, sims=cfg.sims, prior_mix=cfg.prior_mix
        )

        self._history: List[Tuple[Player, Any]] = []  # (player, move 或 复合动作)
        # 备注：久棋有复合动作（跳吃、成方吃），但 UI 现在只传单点坐标；这里先允许 play(Point)
        # 若你的 goboard.Move 支持“从点到点”的构造，也可以扩展 human_play 的两段点击模式

    # —— 基本状态 —— #
    def who_to_play(self) -> Player:
        return self.state.next_player

    def is_over(self) -> bool:
        # 久棋没有 is_over，使用 determine_winner()
        try:
            return bool(self.state.determine_winner())
        except Exception:
            # 安全兜底：没有 winner 也当作未结束
            return False

    def is_human_turn(self) -> bool:
        p = self.who_to_play()
        return (p == Player.black and self.black_agent is None) or (p == Player.white and self.white_agent is None)

    def _apply_composite(self, mv_list: List[Move]):
        """把复合动作逐步应用。"""
        before = self.state
        st = self.state
        for m in mv_list:
            st = st.apply_move(m)
        self.state = st
        self._history.append((before.next_player, mv_list))

        # 通知 Agent（用于维护历史）
        agent = self.black_agent if before.next_player == Player.black else self.white_agent
        if agent is not None and mv_list:
            try:
                agent.on_move_applied(mv_list[0], self.state)
            except Exception:
                pass

    def _apply_single(self, mv: Move):
        before = self.state
        self.state = self.state.apply_move(mv)
        self._history.append((before.next_player, [mv]))

        agent = self.black_agent if before.next_player == Player.black else self.white_agent
        if agent is not None:
            try:
                agent.on_move_applied(mv, self.state)
            except Exception:
                pass

    # —— 人类：一次只走一步（若后续支持跳吃/成方吃，可以在 UI 侧扩展为“多次点击”聚合成复合动作） —— #
    def human_play(self, row: int, col: int) -> Dict:
        if self.is_over():
            return {"ok": False, "msg": "对局已结束", "board": self.export_board()}

        if not self.is_human_turn():
            return {"ok": False, "msg": "当前是模型回合，请点击“AI 落子”", "board": self.export_board()}

        # 简化：把点击当作“落在 (row,col)”的单步；若久棋规则支持从点到点，可在 UI 侧做两次点击
        mv = Move.play(Point(row, col))
        try:
            # 如果 goboard 的 is_valid_move 对久棋禁止“直接落子”，这里可以扩展：
            # 1) 若点击在已有己方棋子上，记为“选子”；下一次点击为目标点（需要在 UI/Engine 存 pending）
            # 2) 先阶段直接落子 / 次阶段移动…… —— 这里先做最小可用实现
            if not self.state.is_valid_move(mv):
                return {"ok": False, "msg": f"非法落子：({row},{col})", "board": self.export_board()}
        except Exception:
            # 某些久棋实现没有 is_valid_move 的此分支，直接尝试 apply
            pass

        self._apply_single(mv)
        side = "黑" if self._history[-1][0] == Player.black else "白"
        return {
            "ok": True,
            "human": {"move": (row, col), "player": side},
            "msg": f"人类（{side}）落子：({row},{col})",
            "board": self.export_board(),
        }

    # —— 模型：若到模型回合则只走“一步”（一组复合动作视为一步） —— #
    def _which_agent(self) -> Optional[BaseJiuAgent]:
        p = self.who_to_play()
        return self.black_agent if p == Player.black else self.white_agent

    def side_is_model_to_play(self) -> bool:
        if self.is_over():
            return False
        return self._which_agent() is not None

    def model_play_if_needed(self) -> Optional[Dict]:
        if self.is_over():
            return {"msg": "对局已结束", "board": self.export_board()}

        if not self.side_is_model_to_play():
            return None

        agent = self._which_agent()
        assert agent is not None, "side_is_model_to_play 为真时 agent 不应为空"

        rec: AgentOutput = agent.propose(self.state)
        # rec.move 是该复合动作的首步；但是我们需要完整复合动作才能应用
        # 由于 propose 内部是基于 run_mcts_jiu 的，将选择的 legal_root[a_idx] 作为“复合动作”，这里只能按单步应用或再次对齐合法表
        # 为稳妥，这里重新按当前根合法复合动作匹配首步
        legal_root = _listify_legal(self.state, self.cfg.board_size)
        chosen: Optional[List[Move]] = None

        if rec.move is not None and hasattr(rec.move, "point"):
            r0, c0 = rec.move.point.row, rec.move.point.col
            for mv_list in legal_root:
                if not mv_list:
                    continue
                m0 = mv_list[0]
                if hasattr(m0, "point") and m0.point and m0.point.row == r0 and m0.point.col == c0:
                    chosen = mv_list
                    break

        # 若没匹配上，就退化为第一个合法复合动作（极端兜底）
        if chosen is None and legal_root:
            chosen = legal_root[0]

        if chosen:
            self._apply_composite(chosen)
            side = "黑" if self._history[-1][0] == Player.black else "白"
            out = {
                "msg": f"模型（{side}）落子：({chosen[0].point.row},{chosen[0].point.col})",
                "board": self.export_board(),
            }
            if rec.value is not None:
                out["value"] = float(rec.value)
            if rec.topk:
                out["topk"] = rec.topk
            return out

        return {"msg": "无合法着法", "board": self.export_board()}

    def model_play_once(self, sims: int = None, topk: int = None):
        out = self.model_play_if_needed()
        if isinstance(out, dict) and out.get("topk") and topk is not None:
            out["topk"] = out["topk"][: int(topk)]
        return out

    # —— 导出棋盘（给 UI 渲染） —— #
    def export_board(self) -> Dict:
        stones = []
        N = self.cfg.board_size
        for r in range(1, N + 1):
            for c in range(1, N + 1):
                p = Point(r, c)
                s = self.state.board.get(p)
                if s is None:
                    continue
                stones.append({
                    "row": r, "col": c,
                    "color": "black" if s == Player.black else "white"
                })
        try:
            res = compute_game_result(self.state)
            winner = res.winner
        except Exception:
            winner = None

        return {
            "size": N,
            "to_play": "black" if self.state.next_player == Player.black else "white",
            "stones": stones,
            "done": self.is_over(),
            "winner": (None if winner is None else ("black" if winner == Player.black else "white")),
        }

    # —— 操作：悔棋 —— #
    def undo(self, k: int = 1) -> Dict:
        if k <= 0 or len(self._history) == 0:
            return {"ok": False, "msg": "无可悔步", "board": self.export_board()}
        remain = self._history[:-k] if k <= len(self._history) else []
        # 重新从空局回放
        N = self.cfg.board_size
        self.state = GameState.new_game(N)
        self._history = []
        for player, mv_list in remain:
            for m in mv_list:
                self.state = self.state.apply_move(m)
            self._history.append((player, mv_list))
        return {"ok": True, "msg": f"已悔棋 {k} 手", "board": self.export_board()}
