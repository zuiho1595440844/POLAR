# -*- coding: utf-8 -*-
from __future__ import absolute_import
from collections import namedtuple

from .gotypes import Player
# 这里不直接引入 Point/Board，避免循环依赖；只依赖 GameState 的接口

class GameResult(namedtuple("GameResult", "b w")):
    @property
    def winner(self):
        if self.b > self.w:
            return Player.black
        elif self.w > self.b:
            return Player.white
        return None

    @property
    def winning_margin(self):
        return abs(self.b - self.w)

    def __str__(self):
        if self.b > self.w:
            return f"B+{self.b - self.w}"
        elif self.w > self.b:
            return f"W+{self.w - self.b}"
        return "Draw"


def compute_game_result(game_state) -> GameResult:
    """
    依据 GameState 的 winner() 判定：
    - 黑胜 -> GameResult(b=1, w=0)
    - 白胜 -> GameResult(b=0, w=1)
    - 未分胜负/和棋 -> GameResult(b=0, w=0)
    """
    w = getattr(game_state, "winner", None)
    w = w() if callable(w) else None
    if w == Player.black:
        return GameResult(b=1, w=0)
    if w == Player.white:
        return GameResult(b=0, w=1)
    return GameResult(b=0, w=0)
