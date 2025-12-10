# backend/backend.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, glob
from typing import Optional, List, Dict, Any
# 确保能导入 RL 训练入口；若缺失则给出兜底占位

from .rl.train import run_rl as _run_rl  # Go / Gomoku
# backend/backend.py  —— 放在文件最顶端、任何其他导入之前
import sys as _sys
import importlib as _importlib

# 把 backend.jiu.dlgo 注册成顶层包 "dlgo"，并把常用子模块也注册为 "dlgo.xxx"
try:
    # 先导入包本体（确保 backend/jiu/dlgo/__init__.py 存在）
    _dlgo_pkg = _importlib.import_module("backend.jiu.dlgo")
    _sys.modules.setdefault("dlgo", _dlgo_pkg)

    # 再把常用子模块挂到顶层别名下（按需可继续加）
    for _sub in ("goboard", "gotypes", "scoring", "utils"):
        try:
            _m = _importlib.import_module(f"backend.jiu.dlgo.{_sub}")
            _sys.modules[f"dlgo.{_sub}"] = _m
        except Exception:
            pass
except Exception:
    # 没有久棋包也不影响 Go/Gomoku
    pass

try:
    from .rl.train import run_rl as _run_rl
except Exception as e:
    _run_rl = None
    print(f"[ImportWarn] 无法导入 backend.rl.train.run_rl：{e}")

try:
    from .rl.train_jiu import run_rl_jiu as _run_rl_jiu
except Exception as e:
    _run_rl_jiu = None
    print(f"[ImportWarn] 无法导入 backend.rl.train_jiu.run_rl_jiu：{e}")
from .arena.match import MatchConfig, MatchEngine, list_available_models


# 确保 backend 是包（需要 backend/__init__.py & backend/llm/__init__.py）
class Backend:
    def __init__(
            self,
            model_root: str = "./models",
            llm_subdir: str = "deepseek-finetuned-go",
            valuenet_path: Optional[str] = None,
            device: str = "cuda",
            prior_mix: float = 0.7,
            temperature: float = 1.0,
    ):
        self.model_root = model_root
        self.llm_subdir = llm_subdir
        self.device = device
        self.prior_mix = prior_mix
        self.temperature = temperature
        self._models_index: Dict[str, List[str]] = {}
        self._scan_models()
        self._match: MatchEngine | None = None

    # ========== 供 UI 调用的接口（最小实现） ==========
    def list_pretrained_models(self) -> List[str]:
        root = os.path.join(self.model_root, "pretrained")
        if not os.path.isdir(root):
            return []
        cands = [p for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p)]
        return sorted([os.path.basename(p) for p in cands])

    def list_sft_models(self):
        """返回 models/sft 下的子目录名列表（仅目录名），更健壮地处理相对/绝对路径。"""
        out = []
        seen = set()

        # 收集可能的根路径：self.model_root 与 ./models
        roots = []
        if getattr(self, "model_root", None):
            roots.append(os.path.abspath(self.model_root))
        roots.append(os.path.abspath("./models"))

        for root in roots:
            sft_dir = os.path.join(root, "sft")
            if not os.path.isdir(sft_dir):
                continue
            try:
                for name in os.listdir(sft_dir):
                    p = os.path.join(sft_dir, name)
                    if os.path.isdir(p) and name not in seen:
                        out.append(name)
                        seen.add(name)
            except Exception:
                # 读目录失败时忽略该 root
                continue

        return sorted(out)

    def list_models(self, game: str) -> List[str]:
        return self._models_index.get(game, [])

    def load_model(self, name: str, game: str) -> None:
        # 这里暂时不加载真实模型，先占位
        return

    def current_model_name(self, side: str) -> str:
        return "（未加载）"

    def reset(self, game: str, board_size: int):
        return

    # 实战部分
    # 1) 刷新列表模型
    def refresh_model_list(self):
        return list_available_models(self.model_root)

    # 2) 开始对局
    def start_match(self, game: str, board_size: int,
                    black_choice: str, white_choice: str,
                    sims: int = 128, prior_mix: float = 0.7):
        """
        black_choice/white_choice ∈ refresh_model_list()["merged"]
        - "none" => 人类
        - "sft:<name>"
        - "rl:<relpath under models/rl>"
        """
        # 两边都 none -> 仅人对人：给信息提示，不阻塞
        if black_choice == "none" and white_choice == "none":
            self._match = MatchEngine(MatchConfig(
                game=game, board_size=board_size,
                black_kind="none", white_kind="none",
                sims=sims, prior_mix=prior_mix, device=self.device, model_root=self.model_root
            ))
            return {
                "ok": True,
                "msg": "已创建“人类 vs 人类”对局。请在棋盘交叉点点击落子。",
                "board": self._match.export_board()
            }

        # 至少一方为模型
        self._match = MatchEngine(MatchConfig(
            game=game, board_size=board_size,
            black_kind=black_choice, white_kind=white_choice,
            sims=sims, prior_mix=prior_mix, device=self.device, model_root=self.model_root
        ))
        info = {
            "ok": True,
            "msg": f"对局已开始：黑={black_choice}，白={white_choice}",
            "board": self._match.export_board()
        }

        return info

    # 3) 人类点击（行列 1-based）
    def human_click(self, row: int, col: int) -> dict:
        """
        人类点击棋盘：仅在人类当棋时生效，不自动让模型回手。
        """
        if self._match is None:
            return {"msg": "请先点击“新对局 / 重置”。", "board": self._empty_board_payload()}
        step = self._match.human_play(row, col)  # 只下一步
        # step 建议返回 {"msg": "...", "board": export_board()}
        return step

    def model_play_if_needed(self, sims: int = 300, topk: int = 5) -> dict:
        """
        仅当当前行动方是模型时，走一步；否则提示“当前轮到人类”。
        """
        if self._match is None:
            return {"msg": "请先点击“新对局 / 重置”。", "board": self._empty_board_payload()}

        if not self._match.side_is_model_to_play():
            return {"msg": "当前轮到人类，无法代走。", "board": self._match.export_board()}

        step = self._match.model_play_once(sims=sims, topk=topk)
        if step is None:
            return {"msg": "无可行 AI 步或对局已结束。", "board": self._match.export_board()}
        return step

    # 4) 悔棋/认输
    def undo(self, k: int = 1):
        if not self._match:
            return {"ok": False, "msg": "未开始对局"}
        out = self._match.undo(k)
        return {**out, "board": self._match.export_board()}

    def resign(self):
        if not self._match:
            return {"ok": False, "msg": "未开始对局"}
        out = self._match.resign()
        return {**out, "board": self._match.export_board()}

    # ========= 训练（流式输出） =========
    def start_sft_training(
            self,
            game: str,
            board_size: int,
            epochs: int,
            lr: float,
            batch_size: int,
            base_model: Optional[str] = None,
    ):
        # 入参检查
        if not base_model or "刷新" in base_model or "未发现" in base_model:
            yield "[SFT] 失败：请在“预训练模型”中选择一个目录"
            return
        base_dir = os.path.join(self.model_root, "pretrained", base_model)
        if not os.path.isdir(base_dir):
            yield f"[SFT] 失败：未找到预训练模型目录 {base_dir}"
            return

        # 相对导入（同包）
        try:
            from .llm.train import run_sft
        except Exception:
            try:
                from backend.llm.train import run_sft  # 兜底
            except Exception as e:
                yield f"[SFT] 未触发：缺少 backend.llm.train.run_sft —— {e}"
                return

        # 调用训练：逐条把 run_sft 的生成器日志透传给 UI
        try:
            for line in run_sft(
                    base_model_dir=base_dir,
                    game=game,
                    board_size=int(board_size),
                    epochs=int(epochs),
                    lr=float(lr),
                    batch_size=int(batch_size),
                    save_root=os.path.join(self.model_root, "sft"),
            ):
                yield line
        finally:
            self._scan_models()  # 刷新一次产出索引

    def start_rl_training(
            self,
            game: str,
            board_size: int,
            episodes: int,
            sims: int,
            gamma: float,
            lam: float,
            note: str,
            sft_model_name: str = "",
    ):
        try:
            # 以 UI 选择优先；为空则退回默认的 self.llm_subdir
            llm_subdir = (sft_model_name or self.llm_subdir or "").strip()

            # 先给前端一个立即可见的占位行，避免用户以为没启动
            buf = []

            def _push(line: str):
                if not line:
                    return
                buf.append(line.rstrip())
                yield "\n".join(buf[-500:])

            # 选择对应的 RL 训练入口（直接用顶部导入的符号）
            if game == "jiu":
                if _run_rl_jiu is None:
                    yield from _push(
                        "[RL/Jiu] 未实现：缺少 backend.rl.train_jiu.run_rl_jiu（请确认 train_jiu.py 存在且可导入）")
                    return
                runner = _run_rl_jiu
            else:
                if _run_rl is None:
                    yield from _push("[RL] 未实现：缺少 backend.rl.train.run_rl（请确认 rl/train.py 可导入）")
                    return
                runner = _run_rl

            # 启动 RL 生成器
            gen = runner(
                game=game,
                board_size=board_size,
                episodes=episodes,
                sims=sims,
                gamma=gamma,
                lam=lam,
                note=note,
                model_root=self.model_root,
                llm_subdir=llm_subdir,
                prior_mix=self.prior_mix,
                device=self.device,
                save_root=os.path.join(self.model_root, "rl"),
            )

            # 明确告诉前端当前用的是哪个入口，方便你确认“是否真的使用了”
            yield from _push("[RL] 已启动，自对弈与训练进行中...")
            yield from _push(f"[RL] 训练入口：{runner.__module__}.{runner.__name__}")
            yield from _push(
                f"[RL] 参数：game={game}, board_size={board_size}, episodes={episodes}, sims={sims}, prior_mix={self.prior_mix}, llm_subdir='{llm_subdir}'")

            # 逐条累积输出到前端
            for msg in gen:
                yield from _push(str(msg))

        except Exception as e:
            import traceback
            yield f"[RL] 异常：{e}"
            yield traceback.format_exc()

    # ========= 内部工具 =========
    def _scan_models(self):
        """扫描 models/sft/* 作为可选模型列表（按棋种粗分类）。"""
        root = os.path.join(self.model_root, "sft")
        idx: Dict[str, List[str]] = {"go": [], "gomoku": [], "jiu": []}
        if os.path.isdir(root):
            for p in glob.glob(os.path.join(root, "*")):
                name = os.path.basename(p)
                if name.startswith("go-"):
                    idx["go"].append(name)
                elif name.startswith("gomoku-"):
                    idx["gomoku"].append(name)
                elif name.startswith("jiu-"):
                    idx["jiu"].append(name)
        self._models_index = {k: sorted(v) for k, v in idx.items()}

    # ========= 对弈（暂用占位） =========
    def recommend_move(self, state: dict, player_side: str, topk: int = 5, sims: int = 300) -> dict:
        # 先给 UI 跑通：随机合法
        board = state["board"]
        sz = state["board_size"]
        empty = [(r, c) for r in range(sz) for c in range(sz) if board[r][c] == 0]
        if not empty:
            return {"move": None, "value": 0.0, "policy": []}
        import random
        mv = random.choice(empty)
        random.shuffle(empty)
        policy = [(m, 1.0 / max(1, min(topk, len(empty)))) for m in empty[:min(topk, len(empty))]]
        return {"move": mv, "value": 0.0, "policy": policy}
