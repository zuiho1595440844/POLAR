import copy
import itertools
import random
from ..dlgo.gotypes import Player, Point
from ..dlgo.utils import coords_from_point, print_move

__all__ = [
    'Board',
    'GameState',
    'Move',
]


class IllegalMoveError(Exception):
    pass


class Board:
    def __init__(self, num_rows=14, num_cols=14):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.grid = {}

    def is_on_grid(self, point):
        return 1 <= point.row <= self.num_rows and 1 <= point.col <= self.num_cols

    def get(self, point):
        """Return the content of a point on the board: Player.black, Player.white, or None"""
        if self.is_on_grid(point):
            return self.grid.get(point)  # 返回该点的棋子，如果该点没有棋子，则返回None
        return None  # 如果点不在棋盘上，也返回None

    def place_stone(self, player, point):
        assert self.is_on_grid(point)
        assert self.grid.get(point) is None
        self.grid[point] = player

    def remove_stone(self, point):
        if self.is_on_grid(point) and self.grid.get(point):
            self.grid[point] = None
            # print("remove " + coords_from_point(point))

    def move_stone(self, from_point, to_point):
        assert self.is_on_grid(from_point) and self.is_on_grid(to_point)
        player = self.grid.get(from_point)
        assert player and self.grid.get(to_point) is None
        self.grid[to_point] = player
        self.grid[from_point] = None


class GameState:
    def __init__(self, board, next_player, previous, move, phase=1, round=1, black_in_phase_three=False,
                 white_in_phase_three=False, caputure_num=0):
        self.board = board
        self.next_player = next_player
        self.previous_state = previous
        self.last_move = move
        self.phase = phase  # 新增：游戏阶段
        self.round = round  # 新增：游戏手数
        self.black_in_phase_three = black_in_phase_three
        self.white_in_phase_three = white_in_phase_three
        self.caputure_num = caputure_num

    def determine_winner(self):
        board = self.board
        black_squares = self.find_squares(Player.black)
        white_squares = self.find_squares(Player.white)
        black_dalian = self.find_dalian(Player.black)
        white_dalian = self.find_dalian(Player.white)
        black_triangles = self.find_triangle(Player.black)
        white_triangles = self.find_triangle(Player.white)

        black_stones = sum(1 for point in board.grid.values() if point == Player.black)
        white_stones = sum(1 for point in board.grid.values() if point == Player.white)
        if self.phase == 2:
            # 规则1：一方有方阵，对方棋子只剩3颗或被吃光
            if black_squares and white_stones <= 3:
                return Player.black
            if white_squares and black_stones <= 3:
                return Player.white

            # 2：一方形成两个或两个以上褡裢形状时，对方没有有效形状或虽进入飞子但无三角形进行飞子成方
            if len(black_dalian) >= 2 and (not white_dalian or (self.white_in_phase_three and not white_triangles)):
                return Player.black
            if len(white_dalian) >= 2 and (not black_dalian or (self.black_in_phase_three and not black_triangles)):
                return Player.white

            # # 规则3：一方的棋子较多，但都不能进行有效移动成方
            # # 暂时简化逻辑：如果一方没有三角形，也没有方阵，且对方有方阵或褡裢形状，则判负
            # if not black_triangles and not black_squares and (white_squares or white_dalian):
            #     return Player.white
            # if not white_triangles and not white_squares and (black_squares or black_dalian):
            #     return Player.black
        return None  # 如果没有满足上述条件，游戏还未分出胜负

    # def get_legal_moves(self):
    #     legal_moves = []
    #     if self.phase == 1:
    #         for row in range(1, self.board.num_rows + 1):
    #             for col in range(1, self.board.num_cols + 1):
    #                 point = Point(row=row, col=col)
    #                 move = Move.play(point)
    #                 if self.is_valid_move(move):
    #                     legal_moves.append(move)
    #     else:
    #         if self.caputure_num > 0:
    #             print("方吃")
    #             # 收集棋盘上所有敌方棋子的位置
    #             enemy_stones = [point for point, occupant in self.board.grid.items() if occupant == self.next_player.other]
    #             if self.caputure_num >= 1:
    #                 for stone in enemy_stones:
    #                     legal_moves.append(Move.capture(stone))
    #                 self.caputure_num = self.caputure_num - 1
    #                 if self.caputure_num < 0:
    #                     self.caputure_num = 0
    #                 return legal_moves
    #
    #         # 根据当前玩家判断是否处于第三阶段
    #         if self.next_player == Player.black:
    #             in_phase_three = self.black_in_phase_three
    #         else:
    #             in_phase_three = self.white_in_phase_three
    #
    #         for row in range(1, self.board.num_rows + 1):
    #             for col in range(1, self.board.num_cols + 1):
    #                 point = Point(row=row, col=col)
    #                 if self.board.get(point) == self.next_player:
    #                     # 对于每个己方棋子，检查可以合法移动到的位置
    #                     # 根据是否处于第三阶段来调整移动生成
    #                     for move in self.generate_moves_for_piece(point, in_phase_three):
    #                         legal_moves.append(move)
    #
    #     return legal_moves
    def get_legal_moves(self):
        from ..rules_core import enumerate_jiu_legal_moves
        return enumerate_jiu_legal_moves(self)

    def apply_move_list(self, mv_list) -> "GameState":
        """
        复合动作执行：
          mv_list[0] : 位移（from_point, to_points..., to_point）
          mv_list[1:]: 提子（is_capture=True, point 不为 None）
        """
        st = self
        main = mv_list[0]
        # 1) 移子（包含单跳/连跳）
        cur = main.from_point
        # 清掉起点
        st = st._move_stone(cur, None)
        # 中继点
        for p in main.to_points:
            cur = p
        # 终点
        st = st._move_stone(main.to_point, st.next_player)  # 你已有 set 或 place 接口的话，调用对应方法

        # 2) 提掉被吃子
        for cap in mv_list[1:]:
            if getattr(cap, "is_capture", False) and getattr(cap, "point", None) is not None:
                st = st._move_stone(cap.point, None)

        # 换手等其他状态维护…
        st = st._flip_player()
        return st

    def explore_jump_moves(self, point, visited=None, is_initial_call=True):
        if visited is None:
            visited = set()
        visited.add(point)
        moves = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 上，右，下，左
        for d in directions:
            jump_point = Point(point.row + 2 * d[0], point.col + 2 * d[1])
            mid_point = Point(point.row + d[0], point.col + d[1])
            if self.board.is_on_grid(jump_point) and jump_point not in visited and \
                    self.board.get(jump_point) is None and self.board.get(mid_point) is not None and \
                    self.board.get(mid_point) != self.next_player:
                # 在初始调用中不记录单次跳跃，只探索连跳
                if not is_initial_call:
                    moves.append(Move.move(point, jump_point))
                # 递归探索连跳，允许记录连跳
                further_jumps = self.explore_jump_moves(jump_point, visited.copy(), False)
                for fj in further_jumps:
                    moves.append(
                        Move.move(point, fj.to_point, [jump_point] + fj.to_points if fj.to_points else [jump_point]))
        return moves

    def generate_moves_for_piece(self, point, in_phase_three=False):
        moves = []
        if in_phase_three:
            # 第三阶段：飞子 - 可以移动到任意空位
            for row in range(1, self.board.num_rows + 1):
                for col in range(1, self.board.num_cols + 1):
                    next_point = Point(row, col)
                    if self.board.get(next_point) is None:
                        moves.append(Move.move(point, next_point))
        else:
            # 第二阶段：相邻移动
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 上，右，下，左
            for d in directions:
                next_point = Point(point.row + d[0], point.col + d[1])
                if self.board.is_on_grid(next_point) and self.board.get(next_point) is None:
                    moves.append(Move.move(point, next_point))

            # 第二阶段：单次跳吃或者连跳
            jump_moves = self.explore_jump_moves(point, is_initial_call=True)
            moves.extend(jump_moves)

        return moves

    def apply_move(self, move):
        if move.is_play or move.is_play_stage2 or move.is_capture:
            next_board = copy.deepcopy(self.board)
            if move.is_capture:
                next_board.remove_stone(move.point)
            # 根据当前阶段和玩家是否进入第三阶段应用不同的逻辑
            if self.phase == 1:
                # 第一阶段：布子阶段
                next_board.place_stone(self.next_player.other, move.point)
            elif self.phase == 2 and move.is_capture is None:

                if (self.next_player == Player.black and not self.black_in_phase_three) or \
                        (self.next_player == Player.white and not self.white_in_phase_three):
                    # 第二阶段：走子阶段
                    next_board, caputure = self.apply_move_phase_two(self.next_player, move, next_board)
                else:
                    # 如果玩家已进入第三阶段，应用第三阶段的移动规则
                    next_board, caputure = self.apply_move_phase_three(self.next_player, move, next_board)
        else:
            next_board = self.board

        # 检查是否需要更换游戏阶段和更新进入第三阶段的状态
        new_phase, next_board = self.check_phase_change(next_board)

        if move.is_capture:
            new_round = self.round
        else:
            new_round = self.round + 1

        # 更新GameState实例，包括进入第三阶段的状态
        return GameState(next_board, self.next_player.other, self, move, new_phase, new_round,
                         self.black_in_phase_three, self.white_in_phase_three, self.caputure_num)

    def check_phase_change(self, next_board):

        # 如果棋盘被填满，游戏进入第二阶段
        if self.phase == 1 and self.round == self.board.num_rows * self.board.num_cols:
            # print("进入第二阶段：\n")
            # 移除中心对角线上的两个棋子以进入第二阶段
            center_point1 = Point(self.board.num_rows // 2, self.board.num_cols // 2)
            center_point2 = Point(self.board.num_rows // 2 + 1, self.board.num_cols // 2 + 1)
            next_board.remove_stone(center_point1)
            next_board.remove_stone(center_point2)

            return 2, next_board  # 返回新阶段

        # 计算每方的棋子数量
        black_stones = sum(1 for point in self.board.grid.values() if point == Player.black)
        white_stones = sum(1 for point in self.board.grid.values() if point == Player.white)

        if self.phase == 2:
            if black_stones <= self.board.num_rows and not self.black_in_phase_three:
                self.black_in_phase_three = True  # 黑方进入第三阶段
                print("黑方进入飞子")
            if white_stones <= self.board.num_cols and not self.white_in_phase_three:
                self.white_in_phase_three = True  # 白方进入第三阶段
                print("白方进入飞子")

        # 如果没有阶段变化，返回当前阶段
        return self.phase, next_board

    def apply_move_phase_two(self, player, move, next_board):
        from_point = move.from_point
        to_point = move.to_point

        # 如果 move.to_points 是 None 或空列表，只处理单次移动或跳吃
        if not move.to_points:
            # 执行单次移动或跳吃的逻辑
            assert next_board.get(from_point) == player, "Move from a point that does not belong to the player"
            assert next_board.get(to_point) is None, "Move to a non-empty point"
            if abs(from_point.row - to_point.row) > 1 or abs(from_point.col - to_point.col) > 1:
                # 处理跳吃逻辑
                mid_point = Point((from_point.row + to_point.row) // 2, (from_point.col + to_point.col) // 2)
                assert next_board.get(mid_point) == player.other, "Must jump over an opponent's stone"
                next_board.remove_stone(mid_point)  # 移除跳过的敌方棋子
            next_board.move_stone(from_point, to_point)
        else:
            # 处理连跳逻辑
            print("连跳")
            for to_point in move.to_points:
                assert next_board.get(from_point) == player, "Move from a point that does not belong to the player"
                assert next_board.get(to_point) is None, "Move to a non-empty point"
                # 中间每次跳跃的逻辑与单次跳吃相同
                next_board.move_stone(from_point, to_point)
                from_point = to_point  # 准备下一次跳吃

        # 检查并形成方阵的逻辑
        caputure = self.check_and_capture_square(player, to_point, next_board)
        return next_board, caputure

    def apply_move_phase_three(self, player, move, next_board):
        from_point = move.from_point
        to_point = move.to_point  # 单次移动
        to_points = move.to_points  # 连跳目标点序列，如果有的话

        if to_points:
            # 执行连跳
            for to_point in to_points:
                assert next_board.get(from_point) == player, "Move from a point that does not belong to the player"
                assert next_board.get(to_point) is None, "Move to a non-empty point"
                next_board.move_stone(from_point, to_point)
                from_point = to_point
        else:
            # 执行单次任意移动
            assert next_board.get(from_point) == player, "Move from a point that does not belong to the player"
            assert next_board.get(to_point) is None, "Move to a non-empty point"
            next_board.move_stone(from_point, to_point)

        # 检查并形成方阵的逻辑也适用于第三阶段
        caputure = self.check_and_capture_square(player, to_point, next_board)
        return next_board, caputure

    def check_and_capture_square(self, player, point, board):
        # 检查以当前点为中心的2x2区域是否形成方阵
        # 定义检查方阵的方向偏移量，这些偏移量代表了以当前点为方阵中的一个角时，其他三个点的位置
        directions = [(-1, -1), (-1, 0), (0, -1), (0, 0)]
        # 初始化一个计数器，用于跟踪形成的方阵数量
        squares_formed = 0

        # 遍历所有可能的方阵组合
        for drow, dcol in directions:
            # 计算方阵中四个点的位置
            square_points = [
                Point(point.row + drow, point.col + dcol),
                Point(point.row + drow, point.col + dcol + 1),
                Point(point.row + drow + 1, point.col + dcol),
                Point(point.row + drow + 1, point.col + dcol + 1)
            ]
            # 检查这四个点是否都属于当前玩家
            if all(board.get(p) == player for p in square_points):
                squares_formed += 1
        self.caputure_num = squares_formed
        if squares_formed > 0:
            pass
            #print("方吃")
        # #对于每个新形成的方阵，移除一个敌方棋子
        # for _ in range(squares_formed):
        #     self.capture_extra_enemy_stone(player, board)

    def find_dalian(self, player):
        found_positions = []
        for row in range(1, self.board.num_rows):
            for col in range(1, self.board.num_cols):
                points = [Point(row, col), Point(row + 1, col), Point(row, col + 1), Point(row + 1, col + 1)]
                player_stones = sum(self.board.get(p) == player for p in points)
                if player_stones == 4:
                    found_positions.append(points)
        return found_positions

    def find_triangle(self, player):
        found_positions = []
        for row in range(1, self.board.num_rows):
            for col in range(1, self.board.num_cols):
                points = [Point(row, col), Point(row + 1, col), Point(row, col + 1), Point(row + 1, col + 1)]
                player_stones = sum(self.board.get(p) == player for p in points)
                if player_stones == 3:
                    found_positions.append(points)
        return found_positions

    def find_squares(self, player):
        found_squares = []
        for row in range(1, self.board.num_rows):
            for col in range(1, self.board.num_cols):
                # 检查当前点、右侧、下方和对角线点是否都属于同一玩家
                if self.board.get(Point(row=row, col=col)) == player and \
                        self.board.get(Point(row=row, col=col + 1)) == player and \
                        self.board.get(Point(row=row + 1, col=col)) == player and \
                        self.board.get(Point(row=row + 1, col=col + 1)) == player:
                    # 如果这四个点都属于同一玩家，记录这个方阵
                    found_squares.append((Point(row=row, col=col),
                                          Point(row=row, col=col + 1),
                                          Point(row=row + 1, col=col),
                                          Point(row=row + 1, col=col + 1)))
        return found_squares

    @classmethod
    def new_game(cls, board_size):
        if isinstance(board_size, int):
            board_size = (board_size, board_size)
        board = Board(*board_size)
        return GameState(board, Player.black, None, None, 1, 1)  # 初始阶段设为1，即将落子第1手

    def is_valid_move(self, move):
        if self.phase == 1:
            if move.to_points or move.to_point or move.is_capture:
                # print("❌ 非法：布局阶段不能使用战斗操作")
                return False
            if not isinstance(move.point, Point) or self.board.get(move.point) is not None:
                # print(f"❌ 非法：布局点 {move.point} 非法或被占")
                return False
            return True

        if self.phase == 2:
            if not (move.to_points or move.to_point or move.is_capture):
                # print("❌ 非法：战斗阶段不能使用布局操作")
                return False

            if move.is_capture:
                stone = self.board.get(move.point)
                if stone is None:
                    #print(f"❌ 捕获失败：{move.point} 上无子")
                    return False
                if stone != self.next_player.other:
                    # print(f"❌ 捕获失败：{move.point} 上不是敌方子")
                    return False
                return True

            if move.to_points:
                for p in move.to_points:
                    if not self.board.is_on_grid(p) or self.board.get(p) is not None:
                        # print(f"❌ 非法跳跃路径：{p} 不在棋盘或被占")
                        return False
            if move.to_point:
                if not self.board.is_on_grid(move.to_point) or self.board.get(move.to_point) is not None:
                    # print(f"❌ 非法目标位置：{move.to_point} 不在棋盘或被占")
                    return False
            return True

        return False


class Move:
    def __init__(self, point=None, is_resign=False, from_point=None, to_point=None, is_play_stage2=None,
                 to_points=None, is_capture=None):
        self.point = point
        self.is_play = point is not None
        self.is_resign = is_resign
        self.from_point = from_point
        self.to_point = to_point
        self.to_points = to_points  # 用于存储连跳序列
        self.is_play_stage2 = is_play_stage2 or (bool(to_points) or bool(to_point))
        self.is_capture = is_capture

    @classmethod
    def play(cls, point):
        return Move(point=point)

    @classmethod
    def resign(cls):
        return Move(is_resign=True)

    # 新增：创建第二阶段和第三阶段的移动
    @classmethod
    def move(cls, from_point, to_point, to_points=None):
        return Move(from_point=from_point, to_point=to_point, to_points=to_points, is_play_stage2=True)

    # 吃子
    @classmethod
    def capture(cls, point):
        return Move(point=point, is_capture=True)
