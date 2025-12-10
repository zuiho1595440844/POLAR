# -*- coding: utf-8 -*-
import copy
from .gotypes import Player, Point

__all__ = ["Board", "GameState", "Move", "IllegalMoveError"]


class IllegalMoveError(Exception):
    pass


class Board:
    def __init__(self, num_rows=15, num_cols=15):
        self.num_rows = num_rows
        self.num_cols = num_cols
        # 稀疏表存子
        self.grid = {}

    def is_on_grid(self, point: Point) -> bool:
        return 1 <= point.row <= self.num_rows and 1 <= point.col <= self.num_cols

    def get(self, point: Point):
        """返回该点内容：Player.black / Player.white / None。越界一律视作 None。"""
        if self.is_on_grid(point):
            return self.grid.get(point)
        return None

    def place_stone(self, player: Player, point: Point):
        assert self.is_on_grid(point), "point out of board"
        assert self.grid.get(point) is None, "point not empty"
        self.grid[point] = player


class Move:
    def __init__(self, point: Point):
        self.point = point

    @classmethod
    def play(cls, point: Point) -> "Move":
        return Move(point)


class GameState:
    """五子棋对局状态"""
    def __init__(self, board: Board, next_player: Player, previous, move: Move,
                 cached_winner: Player = None):
        self.board = board
        self.next_player = next_player
        self.previous_state = previous
        self.last_move = move
        # 形成五连后设置
        self._cached_winner = cached_winner

    # ---------- 规则与合法性 ----------

    def apply_move(self, move: Move) -> "GameState":
        """
        落子。若当前已终局或落点非法，抛出 IllegalMoveError。
        落子后仅围绕最新落点做四向连子检测，若 ≥5 则缓存胜者。
        """
        if not self.is_valid_move(move):
            raise IllegalMoveError("Illegal move (game over / out of board / occupied)")

        next_board = copy.deepcopy(self.board)
        next_board.place_stone(self.next_player, move.point)

        # 生成新态
        new_state = GameState(next_board, self.next_player.other, self, move, cached_winner=None)

        # 仅围绕最新落点检测五连，提高可靠性与性能
        if new_state._five_in_a_row_from(move.point, self.next_player):
            new_state._cached_winner = self.next_player
        return new_state

    def is_valid_move(self, move: Move) -> bool:
        """合法当且仅当：对局未结束、落点在棋盘内、且为空。"""
        if self.is_over():
            return False
        if not self.board.is_on_grid(move.point):
            return False
        return self.board.get(move.point) is None

    def legal_moves(self):
        """若已终局，返回空列表；否则返回所有空点。"""
        if self.is_over():
            return []
        moves = []
        for row in range(1, self.board.num_rows + 1):
            for col in range(1, self.board.num_cols + 1):
                p = Point(row, col)
                if self.board.get(p) is None:
                    moves.append(Move.play(p))
        return moves

    # ---------- 终局与胜负 ----------

    def is_over(self) -> bool:
        """存在五连即终局。"""
        return self.winner() is not None

    def winner(self):
        """
        返回胜者：
        - 如已缓存（上一步刚成五连），直接返回；
        - 否则扫描棋盘（启动/恢复时仍然可靠）。
        """
        if self._cached_winner is not None:
            return self._cached_winner

        board = self.board
        for r in range(1, board.num_rows + 1):
            for c in range(1, board.num_cols + 1):
                p = Point(r, c)
                player = board.get(p)
                if player is None:
                    continue
                if self._five_in_a_row_from(p, player):
                    self._cached_winner = player
                    return player
        return None

    # 兼容旧后端调用（曾经用过 determine_winner()）
    def determine_winner(self):
        return self.winner()

    @staticmethod
    def _count_dir(board: Board, start: Point, player: Player, dr: int, dc: int) -> int:
        """从 start 沿 (dr, dc) 方向连续同色的个数（不含 start）。"""
        r, c = start.row + dr, start.col + dc
        cnt = 0
        while board.get(Point(r, c)) == player:
            cnt += 1
            r += dr
            c += dc
        return cnt

    def _five_in_a_row_from(self, start: Point, player: Player) -> bool:
        """以 start 为中心，四个主方向统计两侧连续同色的总数（含 start）。"""
        for dr, dc in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            cnt = 1  # 包含 start
            cnt += self._count_dir(self.board, start, player, dr, dc)
            cnt += self._count_dir(self.board, start, player, -dr, -dc)
            if cnt >= 5:  # 标准五子棋：五连或以上即胜
                return True
        return False

    # ---------- 工厂 ----------

    @classmethod
    def new_game(cls, board_size):
        if isinstance(board_size, int):
            board_size = (board_size, board_size)
        board = Board(*board_size)
        return GameState(board, Player.black, None, None)
