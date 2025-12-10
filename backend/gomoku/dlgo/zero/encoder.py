import numpy as np
from dlgo.goboard import Move
from dlgo.gotypes import Player, Point

class ZeroEncoder:
    def __init__(self, board_size):
        self.board_size = board_size
        # 4个特征平面：
        # 1. 当前玩家的棋子位置
        # 2. 对手玩家的棋子位置
        # 3. 对手棋子最后一次落子位置
        # 4. 当前玩家是先手或后手（先手平面全为1，否则全为0）
        self.num_planes = 4

    def encode(self, game_state):
        board_tensor = np.zeros(self.shape())
        next_player = game_state.next_player
        other_player = next_player.other

        # 编码当前玩家和对手玩家的棋子位置
        for row in range(1, self.board_size + 1):
            for col in range(1, self.board_size + 1):
                point = Point(row=row, col=col)
                stone = game_state.board.get(point)
                if stone == next_player:
                    board_tensor[0][row - 1][col - 1] = 1
                elif stone == other_player:
                    board_tensor[1][row - 1][col - 1] = 1

        # 对手棋子最后一次落子位置
        if game_state.last_move is not None:
            move_point = game_state.last_move.point
            board_tensor[2][move_point.row - 1][move_point.col - 1] = 1

        # 当前玩家是否为先手
        board_tensor[3][:][:] = 1 if next_player == Player.black else 0

        return board_tensor

    def shape(self):
        return self.num_planes, self.board_size, self.board_size

    def decode_move_index(self, index):
        row = index // self.board_size
        col = index % self.board_size
        return Move.play(Point(row=row + 1, col=col + 1))

    def num_moves(self):
        return self.board_size * self.board_size
