# backend/jiu/rules_core.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Tuple, Optional, Iterable, Set
from dataclasses import dataclass

# ———————— 只使用“久棋”的类型，避免和 go/gomoku 混用 ————————
from ..jiu.dlgo.gotypes import Player, Point
from ..jiu.dlgo.goboard import GameState, Move

__all__ = [
    "enumerate_jiu_legal_moves",
    "count_new_squares_after_move",
]

# ============================================================================
# 棋盘适配：兼容不同 Board 实现（size 或 num_rows/num_cols）
# ============================================================================
@dataclass(frozen=True)
class BoardView:
    rows: int
    cols: int

    @classmethod
    def from_state(cls, state: GameState) -> "BoardView":
        bd = state.board
        if hasattr(bd, "num_rows") and hasattr(bd, "num_cols"):
            return cls(int(bd.num_rows), int(bd.num_cols))
        if hasattr(bd, "size"):
            s = int(bd.size)
            return cls(s, s)
        # 最保守兜底：尝试属性探测失败时，假设 14 路
        return cls(14, 14)

    def is_on_board(self, p: Point) -> bool:
        return 1 <= p.row <= self.rows and 1 <= p.col <= self.cols

    def get(self, state: GameState, p: Point) -> Optional[Player]:
        # 直接转发给真正的 Board.get
        return state.board.get(p)

    def all_points(self) -> Iterable[Point]:
        for r in range(1, self.rows + 1):
            for c in range(1, self.cols + 1):
                yield Point(r, c)

# ============================================================================
# 开局限制：前两手只能在对角中段两端（14 路为 (7,7)/(8,8) 即 G7/H8）
# ============================================================================
def _count_stones(state: GameState, view: BoardView) -> int:
    n = 0
    for p in view.all_points():
        if view.get(state, p) is not None:
            n += 1
    return n

def _opening_points(view: BoardView) -> Optional[Tuple[Point, Point]]:
    # 偶数路才有“正中心两点”（mid,mid) 与 (mid+1,mid+1)
    if view.rows != view.cols:
        return None
    if view.rows % 2 != 0:
        return None
    mid = view.rows // 2
    return Point(mid, mid), Point(mid + 1, mid + 1)

def _apply_opening_constraint_to_results(
    state: GameState,
    view: BoardView,
    results: List[List[Move]]
) -> List[List[Move]]:
    stones = _count_stones(state, view)
    if stones >= 2:
        return results

    pair = _opening_points(view)
    if not pair:
        return results
    p1, p2 = pair

    # 若第2手，则只允许“另外那个空点”
    allowed: Set[Tuple[int, int]] = set()
    if stones == 0:
        allowed = {(p1.row, p1.col), (p2.row, p2.col)}
    elif stones == 1:
        if view.get(state, p1) is None:
            allowed.add((p1.row, p1.col))
        if view.get(state, p2) is None:
            allowed.add((p2.row, p2.col))

    if not allowed:
        return results  # 理论上不会发生

    def _end_of_main(m: Move) -> Optional[Point]:
        """主移动的终点：优先 to_point，没有则取 to_points 最后一个。"""
        if getattr(m, "to_point", None) is not None:
            return m.to_point
        tps = getattr(m, "to_points", None)
        if tps:
            return tps[-1]
        # 纯“放子”/一步走子：有些实现也可能使用 m.point
        if getattr(m, "point", None) is not None and getattr(m, "is_capture", False) is False:
            return m.point
        return None

    filtered: List[List[Move]] = []
    for seq in results:
        if not seq:
            continue
        end = _end_of_main(seq[0])
        if end is None:
            continue
        if (end.row, end.col) in allowed:
            filtered.append(seq)
    return filtered

# ============================================================================
# 跳吃（单跳/连跳）生成
# ============================================================================
ORTHO_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

def _adjacent(p: Point, dr: int, dc: int) -> Point:
    return Point(p.row + dr, p.col + dc)

def _beyond(p: Point, dr: int, dc: int) -> Point:
    return Point(p.row + 2 * dr, p.col + 2 * dc)

def _gen_single_jumps(state: GameState, view: BoardView, player: Player, at: Point) -> List[Tuple[Point, Point]]:
    """从 at 出发所有“单跳”候选：(被吃子 mid, 落点 land)"""
    res: List[Tuple[Point, Point]] = []
    opp = Player.black if player == Player.white else Player.white
    for dr, dc in ORTHO_DIRS:
        mid = _adjacent(at, dr, dc)
        land = _beyond(at, dr, dc)
        if view.is_on_board(mid) and view.is_on_board(land):
            if view.get(state, mid) == opp and view.get(state, land) is None:
                res.append((mid, land))
    return res

class _FakeAfterJump:
    """
    仅供 DFS 连跳检测使用的“视图”：
      - start 视为空
      - mid   视为空（被吃）
      - land  视为当前 player
    """
    def __init__(self, base_state: GameState, start: Point, land: Point, mid: Point, player: Player):
        self._s = base_state
        self._start = start
        self._land = land
        self._mid = mid
        self._pl = player

    @property
    def board(self):
        return self

    def get(self, p: Point):
        if p == self._start: return None
        if p == self._mid: return None
        if p == self._land: return self._pl
        return self._s.board.get(p)

def _dfs_jump_sequences(
    state: GameState,
    view: BoardView,
    player: Player,
    start: Point,
    used_caps: Set[Point]
) -> List[Tuple[List[Point], List[Point]]]:
    """
    回溯生成“连跳”路径：
      返回若干 (path_points, captured_points)
      - path_points: 包含中继落点（不含 start），最后一个是终点
      - captured_points: 与 path_points 一一对应的被吃子列表（顺序一致）
    """
    seqs: List[Tuple[List[Point], List[Point]]] = []

    moves = _gen_single_jumps(state, view, player, start)
    if not moves:
        return []

    for mid, land in moves:
        if mid in used_caps:
            continue
        used_caps.add(mid)

        fake = _FakeAfterJump(state, start, land, mid, player)
        tails = _dfs_jump_sequences(fake, view, player, land, used_caps)
        if not tails:
            seqs.append(([land], [mid]))
        else:
            for path_pts, caps in tails:
                seqs.append(([land] + path_pts, [mid] + caps))

        used_caps.remove(mid)

    return seqs

# ============================================================================
# 成方检测（以终点为中心检查 4 个 1×1 单位格）
# ============================================================================
def _square_anchors_around(p: Point) -> Iterable[Point]:
    r, c = p.row, p.col
    yield Point(r - 1, c - 1)
    yield Point(r - 1, c)
    yield Point(r, c - 1)
    yield Point(r, c)

def _is_square_owned(view_like, owner: Player, anchor: Point) -> bool:
    r, c = anchor.row, anchor.col
    pts = [Point(r, c), Point(r + 1, c), Point(r, c + 1), Point(r + 1, c + 1)]
    # view_like 需要同时提供 is_on_board(p) 与 get(p)
    if not all(view_like.is_on_board(q) for q in pts):
        return False
    return all(view_like.get(q) == owner for q in pts)

class _FakeAfterPath:
    """
    仅用于“终点周围成方检测”的轻量局面：
      - start 置空
      - path_points 每个点置为 owner
    """
    def __init__(self, base_state: GameState, owner: Player, view: BoardView,
                 start: Point, path_points: List[Point]):
        self._s = base_state
        self._owner = owner
        self._view = view
        self._start = start
        self._path = set(path_points)

    def is_on_board(self, p: Point) -> bool:
        return self._view.is_on_board(p)

    def get(self, p: Point):
        if p == self._start:
            return None
        if p in self._path:
            return self._owner
        return self._s.board.get(p)

def count_new_squares_after_move(
    state: GameState,
    view: BoardView,
    owner: Player,
    start: Point,
    path_points: List[Point],
) -> int:
    """
    只看以“终点”为中心是否新形成“方”。（效率与规则匹配性兼顾）
    """
    if not path_points:
        return 0
    end = path_points[-1]
    fake = _FakeAfterPath(state, owner, view, start, path_points)
    k = 0
    for a in _square_anchors_around(end):
        if _is_square_owned(fake, owner, a):
            k += 1
    return k

# ============================================================================
# 合法着法生成（走一步、单跳、连跳 + 成方吃）
# 返回“复合着法”列表：每个元素是 [主移动, capture..., capture...]
# ============================================================================
def _auto_pick_k_opponent(
    state: GameState,
    view: BoardView,
    me: Player,
    k: int,
    prefer_center: Optional[Point] = None,
    exclude: Set[Point] = frozenset()
) -> List[Point]:
    """
    AI/RL 启发式：从对方全盘挑选 k 枚用于“成方吃”（避免组合爆炸）。
    策略：按距离 prefer_center 从近到远排序，跳过 exclude 集合。
    """
    opp = Player.black if me == Player.white else Player.white
    opp_pts = [p for p in view.all_points() if view.get(state, p) == opp and p not in exclude]
    if not opp_pts or k <= 0:
        return []
    if prefer_center is None:
        return opp_pts[:k] if len(opp_pts) >= k else opp_pts
    opp_pts.sort(key=lambda p: abs(p.row - prefer_center.row) + abs(p.col - prefer_center.col))
    return opp_pts[:k]

def enumerate_jiu_legal_moves(state: GameState) -> List[List[Move]]:
    """
    列出“久棋”在当前局面的所有**复合着法**：
      每个复合着法为一个列表：
        [ Move(from_point=..., to_points=[...], to_point=...),     # 主移动（走一步或连跳）
          Move(is_capture=True, point=cap1),
          Move(is_capture=True, point=cap2),
          ...
        ]
    说明：
      - 普通走子：上下左右一格
      - 跳吃：越过对方一子落到其后空点，可连跳，顺序在 to_points 中记录中继点、to_point 为终点
      - 成方吃：以终点为中心检查新形成的“方”，若形成 k 个，则还需“提对方 k 枚”，
               人机局可使用占位（point=None）让前端二次选择；AI/RL 用启发式自动挑选。
      - **开局限制**：前两手只能落在主对角中段两端（偶数路棋盘）。
    """
    view = BoardView.from_state(state)
    me = state.next_player

    results: List[List[Move]] = []

    # 允许人机局由前端选择“成方吃”的具体点位：给 state 挂个标记
    HUMAN_SELECT = getattr(state, "_human_select_captures", False)

    # 我方所有棋子位置
    my_points = [p for p in view.all_points() if view.get(state, p) == me]

    # 1) 普通一步
    for s in my_points:
        for dr, dc in ORTHO_DIRS:
            t = Point(s.row + dr, s.col + dc)
            if view.is_on_board(t) and view.get(state, t) is None:
                main = Move(from_point=s, to_points=[], to_point=t)
                k = count_new_squares_after_move(state, view, me, s, [t])
                if k > 0:
                    if HUMAN_SELECT:
                        # 交给前端选择具体被提子：用占位 capture（point=None）
                        mv = [main] + [Move(is_capture=True, point=None) for _ in range(k)]
                        results.append(mv)
                    else:
                        caps = _auto_pick_k_opponent(state, view, me, k, prefer_center=t)
                        mv = [main] + [Move(is_capture=True, point=q) for q in caps]
                        results.append(mv)
                else:
                    results.append([main])

    # 2) 单跳/连跳
    for s in my_points:
        seqs = _dfs_jump_sequences(state, view, me, s, used_caps=set())
        for path_pts, cap_pts in seqs:
            # path_pts 至少有一个点；to_points 记录中继（不含最后一个），to_point 为最后一个
            if len(path_pts) == 1:
                main = Move(from_point=s, to_points=[], to_point=path_pts[0])
            else:
                main = Move(from_point=s, to_points=path_pts[:-1], to_point=path_pts[-1])

            post: List[Move] = [Move(is_capture=True, point=q) for q in cap_pts]  # 跳吃被吃子
            # 成方吃
            k = count_new_squares_after_move(state, view, me, s, path_pts)
            if k > 0:
                if HUMAN_SELECT:
                    post += [Move(is_capture=True, point=None) for _ in range(k)]
                else:
                    extra = _auto_pick_k_opponent(
                        state, view, me, k, prefer_center=path_pts[-1], exclude=set(cap_pts)
                    )
                    post += [Move(is_capture=True, point=q) for q in extra]

            results.append([main] + post)

    # 3) 开局限制（前两手只许在中线两端）
    results = _apply_opening_constraint_to_results(state, view, results)

    return results
