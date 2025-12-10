import platform
import subprocess

import numpy as np

# tag::print_utils[]
from ..dlgo import gotypes

COLS = 'ABCDEFGHIJKLMNOPQRST'
STONE_TO_CHAR = {
    None: ' . ',
    gotypes.Player.black: ' x ',
    gotypes.Player.white: ' o ',
}


def print_move(player, move):

    if move.is_capture:
        move_str = 'remove %s%d' % (COLS[move.point.col - 1], move.point.row)
    elif move.is_play:
        move_str = '%s%d' % (COLS[move.point.col - 1], move.point.row)
    elif move.from_point is not None and move.to_point is not None:
        # 处理从一个点移动到另一个点的情况
        from_str = '%s%d' % (COLS[move.from_point.col - 1], move.from_point.row)
        to_str = '%s%d' % (COLS[move.to_point.col - 1], move.to_point.row)
        if move.to_points:  # 检查是否存在连跳路径
            # 处理连跳，将每个连跳的点转换为字符串并用箭头连接
            jump_points_str = ' -> '.join(
                ['%s%d' % (COLS[pt.col - 1], pt.row) for pt in [move.to_point] + move.to_points])
            move_str = 'jumps from %s to %s' % (from_str, jump_points_str)
        else:
            move_str = 'moves from %s to %s' % (from_str, to_str)

    else:
        # 如果不是以上情况，简单地打印"makes a move"
        move_str = 'makes a move'

    print('%s %s' % (player, move_str))


def print_board(board):
    for row in range(board.num_rows, 0, -1):
        bump = " " if row <= 9 else ""
        line = []
        for col in range(1, board.num_cols + 1):
            stone = board.get(gotypes.Point(row=row, col=col))
            line.append(STONE_TO_CHAR[stone])
        print('%s%d %s' % (bump, row, ''.join(line)))
    print('    ' + '  '.join(COLS[:board.num_cols]))


# end::print_utils[]


# tag::human_coordinates[]
def point_from_coords(coords):
    col = COLS.index(coords[0]) + 1
    row = int(coords[1:])
    return gotypes.Point(row=row, col=col)


# end::human_coordinates[]


def coords_from_point(point):
    return '%s%d' % (
        COLS[point.col - 1],
        point.row
    )


def clear_screen():
    # see https://stackoverflow.com/a/23075152/323316
    if platform.system() == "Windows":
        subprocess.Popen("cls", shell=True).communicate()
    else:  # Linux and Mac
        # the link uses print("\033c", end=""), but this is the original sequence given in the book.
        print(chr(27) + "[2J")


# NOTE: MoveAge is only used in chapter 13, and doesn't make it to the main text.
# This feature will only be implemented in goboard_fast.py so as not to confuse
# readers in early chapters.
class MoveAge():
    def __init__(self, board):
        self.move_ages = - np.ones((board.num_rows, board.num_cols))

    def get(self, row, col):
        return self.move_ages[row, col]

    def reset_age(self, point):
        self.move_ages[point.row - 1, point.col - 1] = -1

    def add(self, point):
        self.move_ages[point.row - 1, point.col - 1] = 0

    def increment_all(self):
        self.move_ages[self.move_ages > -1] += 1
