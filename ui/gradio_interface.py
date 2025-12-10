# ui/gradio_interface.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
import gradio as gr

_GAME_LABEL2KEY = {
    "围棋(Go)": "go",
    "五子棋(Gomoku)": "gomoku",
    "久棋(Jiu)": "jiu",
}

# =========================
# 常量 / 预设
# =========================
GAME_PRESETS = {
    "围棋(Go)": {"key": "go", "sizes": [9, 13, 19]},
    "五子棋(Gomoku)": {"key": "gomoku", "sizes": [15]},
    "久棋(Jiu)": {"key": "jiu", "sizes": [14]},
}
STONE_NONE, STONE_BLACK, STONE_WHITE = 0, 1, 2
PLAYER_NAME = {STONE_BLACK: "黑", STONE_WHITE: "白"}


# =========================
# 简易本地状态/工具（训练页示例用）
# =========================
def new_board(sz: int) -> np.ndarray:
    return np.zeros((sz, sz), dtype=np.int8)

def _go_star_points(sz: int) -> List[Tuple[int, int]]:
    if sz == 9:
        pts = [2, 4, 6]
    elif sz == 13:
        pts = [3, 6, 9]
    elif sz == 19:
        pts = [3, 9, 15]
    else:
        return []
    return [(r, c) for r in pts for c in pts]


# =========================
# 后端适配与回退
# =========================
def _safe(backend, fn_name: str, default):
    if backend is None:
        return default
    fn = getattr(backend, fn_name, None)
    if callable(fn):
        return fn
    return default

class DummyBackend:
    def __init__(self):
        self._models = {"go": ["llm_go_v1"], "gomoku": ["llm_gmk_v1"], "jiu": ["llm_jiu_v1"]}
        self._current = {"A": None, "B": None}

    def list_pretrained_models(self) -> List[str]:
        root = "models/pretrained"
        if not os.path.isdir(root):
            return []
        return sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])

    def list_models(self, game: str) -> List[str]:
        return self._models.get(game, [])

    def load_model(self, name: str, game: str) -> None:
        self._current["A"] = name

    def current_model_name(self, side: str) -> str:
        return self._current.get(side) or "（未加载）"

    def start_sft_training(self, *args, **kwargs):
        yield "预处理完成，开始训练..."
        import time
        for i in range(3):
            time.sleep(0.4)
            yield f"Epoch {i+1}/3  loss=7.{i}23  acc=0.00{i}"
        yield "[SFT] 训练完成"

    def start_rl_training(self, *a, **k):
        yield "（未实现）"


# =========================
# 训练页：一些工具
# =========================
def _on_game_change_sizes(game_label):
    return gr.update(choices=GAME_PRESETS[game_label]["sizes"],
                     value=GAME_PRESETS[game_label]["sizes"][-1])

def _scan_sft_models(root: str = "./models/sft") -> List[str]:
    if not os.path.isdir(root):
        return []
    return sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])

def _refresh_sft_models(backend, game_label):
    fn = _safe(backend, "list_sft_models", lambda *a, **k: [])
    items = fn() or []
    if not items:
        items = ["（未发现）"]
    return gr.update(choices=items, value=items[0])

def _refresh_pretrained(backend):
    lst = _safe(backend, "list_pretrained_models", lambda: [])()
    if not lst:
        lst = ["（未发现 models/pretrained 下的模型目录）"]
    return gr.update(choices=lst, value=lst[0])

def _start_sft(backend, game_label, size, epochs, lr, batch, pretrained):
    game = GAME_PRESETS[game_label]["key"]
    fn = _safe(backend, "start_sft_training", None)
    if fn is None:
        yield "后端未实现 start_sft_training()"
        return
    buf = []
    for line in fn(game, int(size), int(epochs), float(lr), int(batch), base_model=str(pretrained)):
        if line is None:
            continue
        buf.append(str(line))
        yield "\n".join(buf)

def _start_rl(backend, game_label, size, episodes, sims, gamma, lam, note, rl_llm_subdir):
    game = GAME_PRESETS[game_label]["key"]
    fn = _safe(backend, "start_rl_training", lambda *a, **k: (x for x in ["后端未实现 start_rl_training()"]))
    use_subdir = None if (not rl_llm_subdir or "未发现" in rl_llm_subdir) else rl_llm_subdir
    for line in fn(game, size, int(episodes), int(sims), float(gamma), float(lam), note or "", use_subdir):
        yield line


# =========================
# 主界面
# =========================
def launch_ui(backend=None, server_name="0.0.0.0", server_port=7860, share=False):
    backend = backend or DummyBackend()

    # =========================
    # 绘盘/坐标换算
    # =========================
    def _draw_board(board_payload: dict, canvas_px: int = 640) -> np.ndarray:
        """
        board_payload 需包含：
          {
            "size": N,
            "stones": [{"row":r,"col":c,"color":"black"/"white"}, ...],
            "done": bool, "winner": "black"/"white"/None,
            "game": "go"/"gomoku"/"jiu"   # ✅ 我们会强制写入这个字段
          }
        """
        N = int(board_payload["size"])
        game_name = board_payload.get("game", "go")  # ✅ 缺省按 go

        img = Image.new("RGB", (canvas_px, canvas_px), (238, 204, 120))  # 木色背景
        draw = ImageDraw.Draw(img)

        # 布局
        PAD = 36
        if N == 9:
            PAD = 44
        CELL = (canvas_px - 2 * PAD) / (N - 1)

        # 网格
        for i in range(N):
            # 横线
            y = PAD + i * CELL
            draw.line((PAD, y, canvas_px - PAD, y), fill=(60, 60, 60), width=2)
            # 竖线
            x = PAD + i * CELL
            draw.line((x, PAD, x, canvas_px - PAD), fill=(60, 60, 60), width=2)

        # 星位（围棋常见）
        star_locs = []
        if N in (19, 13, 9):
            step = {19: 6, 13: 4, 9: 3}[N]
            offs = {19: 3, 13: 3, 9: 2}[N]
            for i in range(offs, N, step):
                for j in range(offs, N, step):
                    star_locs.append((i, j))
        r_star = max(2, int(CELL * 0.08))
        for rr, cc in star_locs:
            x = PAD + (cc) * CELL
            y = PAD + (rr) * CELL
            draw.ellipse((x - r_star, y - r_star, x + r_star, y + r_star), fill=(40, 40, 40))

        # ✅ 久棋：中心格左上->右下对角线
        if game_name == "jiu" and (N % 2 == 0):
            mid = N // 2 - 1
            r0, c0 = mid, mid
            r1, c1 = mid + 1, mid + 1
            x0 = PAD + c0 * CELL; y0 = PAD + r0 * CELL
            x1 = PAD + c1 * CELL; y1 = PAD + r1 * CELL
            draw.line((x0, y0, x1, y1), fill=(40, 40, 40), width=3)

        # 棋子
        R = CELL * 0.46
        for s in board_payload["stones"]:
            r, c = s["row"] - 1, s["col"] - 1
            x = PAD + c * CELL
            y = PAD + r * CELL
            color = s["color"]
            if color == "black":
                draw.ellipse((x - R, y - R, x + R, y + R), fill=(15, 15, 15))
            else:
                draw.ellipse((x - R, y - R, x + R, y + R), fill=(245, 245, 245), outline=(0, 0, 0), width=2)

        # 终局提示
        if board_payload.get("done"):
            w = board_payload.get("winner")
            msg = "黑胜" if w == "black" else ("白胜" if w == "white" else "和棋")
            draw.rectangle((0, 0, 160, 36), fill=(0, 0, 0))
            draw.text((8, 8), msg, fill=(255, 255, 255))

        return np.array(img)

    def _nearest_intersection(xy, board_payload: dict, canvas_px: int = 640):
        N = int(board_payload["size"])
        PAD = 36 if N != 9 else 44
        CELL = (canvas_px - 2 * PAD) / (N - 1)
        x, y = xy
        gx = (x - PAD) / CELL
        gy = (y - PAD) / CELL
        c = int(round(gx)); r = int(round(gy))
        if 0 <= r < N and 0 <= c < N:
            if abs(gx - c) <= 0.5 and abs(gy - r) <= 0.5:
                return (r + 1, c + 1)  # 1-based
        return None

    # =========================
    # 与后端交互的回调
    # =========================
    def _reset_match(backend_state, game_label, board_size, black_choice_label, white_choice_label):
        game = _GAME_LABEL2KEY.get(game_label, "go")
        N = int(board_size)

        def _to_key(lbl: str) -> str:
            return "none" if (lbl is None or lbl == "无" or "无可用模型" in str(lbl)) else str(lbl)

        def _to_human(lbl: str) -> str:
            return "人类棋手" if (lbl is None or lbl == "无" or "无可用模型" in str(lbl)) else str(lbl)

        black_key = _to_key(black_choice_label)
        white_key = _to_key(white_choice_label)

        # 用位置参数调用，避免关键字不匹配
        info = backend_state.start_match(game, N, black_key, white_key, 300, 0.7)

        # ✅ 强制把棋种写入 board_payload，供绘图判断是否久棋
        info["board"]["game"] = game

        board_np = _draw_board(info["board"])
        state = {"game": game, "size": N, "board": info["board"]}
        msg = f"新对局：黑={_to_human(black_choice_label)}，白={_to_human(white_choice_label)}。"
        return board_np, state, msg, "—", "—"

    def _refresh_model_list(backend_state, game_label):
        models = backend_state.refresh_model_list()  # {"merged":[...], "sft":[...], "rl":[...]}
        merged = models.get("merged", [])
        if "none" not in merged:
            merged = ["none"] + merged
        labels = [("无" if m == "none" else m) for m in merged]
        return (
            gr.update(choices=labels, value="无"),
            gr.update(choices=labels, value="无"),
        )

    def _load_models(backend_state, game_label, black_choice_label, white_choice_label):
        def _to_human(lbl: str) -> str:
            return "人类棋手" if (lbl is None or lbl == "无" or "无可用模型" in str(lbl)) else str(lbl)
        return f"已选择：黑={_to_human(black_choice_label)}，白={_to_human(white_choice_label)}。点击“新对局 / 重置”开始。"

    def _board_click(backend_state, state, evt: gr.SelectData):
        if not state:
            return gr.update(), state, "请先点击“新对局 / 重置”。", "—", "—"
        if getattr(backend_state, "_match", None) is None:
            return _draw_board(state["board"]), state, "请先点击“新对局 / 重置”。", "—", "—"

        pos = _nearest_intersection(evt.index, state["board"])
        if pos is None:
            return _draw_board(state["board"]), state, "请点击靠近交叉点的位置", "—", "—"
        row, col = pos

        # ✅ 按棋种分流（若没有 jiu 专用接口，则回落到通用 human_click）
        if state.get("game") == "jiu" and hasattr(backend_state, "human_click_jiu"):
            out = backend_state.human_click_jiu(row, col)
        else:
            out = backend_state.human_click(row, col)

        board_payload = out.get("board") or backend_state._match.export_board()
        board_payload["game"] = state.get("game", board_payload.get("game", "go"))  # ✅ 补 game
        state["board"] = board_payload

        img = _draw_board(board_payload)
        msg = out.get("msg", "已落子")  # ✅ 统一从顶层 msg 取
        return img, state, msg, "—", "—"

    def _ai_move(backend_state, state, sims, topk_k, _side_unused):
        if not state:
            return gr.update(), state, "请先开始对局", "—", "—"

        step = backend_state.model_play_if_needed()
        if not step:
            return gr.update(), state, "当前轮到人类，无法代走", "—", "—"

        board_payload = step.get("board") or backend_state._match.export_board()
        board_payload["game"] = state.get("game", board_payload.get("game", "go"))  # ✅ 补 game
        img = _draw_board(board_payload)
        state["board"] = board_payload

        msg = step.get("msg", "AI 已落子")
        v_txt = f"{step['value']:+.3f}" if isinstance(step.get("value"), (float, int)) else "—"
        if step.get("topk"):
            tk = step["topk"][:int(topk_k)]
            topk_txt = "\n".join([f"{coord}: {prob * 100:.1f}%" for coord, prob, _ in tk])
        else:
            topk_txt = "—"

        return img, state, msg, v_txt, topk_txt

    def _ai_vs_ai_once(backend_state, state, sims, topk_k):
        if not state:
            return gr.update(), state, "请先开始对局", "—", "—"

        msgs = []
        last_step = None
        for _ in range(2):
            step = backend_state.model_play_if_needed()
            if not step:
                break
            msgs.append(step.get("msg", "AI 落子"))
            last_step = step
            if backend_state._match.is_over():
                break

        board_payload = (last_step or {}).get("board") or backend_state._match.export_board()
        board_payload["game"] = state.get("game", board_payload.get("game", "go"))  # ✅ 补 game
        img = _draw_board(board_payload)
        state["board"] = board_payload

        v_txt = f"{last_step['value']:+.3f}" if (last_step and isinstance(last_step.get("value"), (float, int))) else "—"
        if last_step and last_step.get("topk"):
            tk = last_step["topk"][:int(topk_k)]
            topk_txt = "\n".join([f"{coord}: {prob * 100:.1f}%" for coord, prob, _ in tk])
        else:
            topk_txt = "—"

        return img, state, ("；".join(msgs) if msgs else "无可行 AI 步或当前是人类回合"), v_txt, topk_txt

    def _undo_click(backend_state, state):
        if not state or getattr(backend_state, "_match", None) is None:
            return gr.update(), state, "请先点击“新对局 / 重置”。", "—", "—"
        out = backend_state.undo(1)
        board_payload = out["board"]
        board_payload["game"] = state.get("game", board_payload.get("game", "go"))  # ✅ 补 game
        state["board"] = board_payload
        return _draw_board(board_payload), state, out.get("msg", "已悔棋"), "—", "—"

    # =========================
    # UI
    # =========================
    with gr.Blocks(css=_CSS()) as demo:
        backend_state = gr.State(backend)
        gr.Markdown("## ♟️ 双人棋类：训练对弈可视化平台")

        with gr.Tabs():
            with gr.TabItem("训练（SFT / 强化学习）"):
                gr.Markdown("### 🧪 训练配置")
                with gr.Row():
                    with gr.Column(scale=1):
                        game_dd = gr.Dropdown(choices=list(GAME_PRESETS.keys()), value="围棋(Go)", label="棋类")
                        size_dd = gr.Dropdown(choices=[9, 13, 19], value=19, label="棋盘尺寸")
                        game_dd.change(_on_game_change_sizes, inputs=[game_dd], outputs=[size_dd])

                        gr.Markdown("#### 阶段一：监督微调（SFT）")
                        pretrained_dd = gr.Dropdown(choices=["（点击右侧刷新）"], value="（点击右侧刷新）",
                                                    label="预训练模型（models/pretrained/*）")
                        btn_refresh_pretrained = gr.Button("刷新预训练模型")

                        sft_epochs = gr.Slider(1, 50, value=5, step=1, label="训练轮数(epochs)")
                        sft_lr = gr.Number(value=2e-5, precision=6, label="学习率")
                        sft_batch = gr.Slider(1, 256, value=64, step=1, label="批大小(batch)")
                        btn_sft = gr.Button("启动 SFT 训练", variant="primary")

                        gr.Markdown("#### 阶段二：强化学习")
                        _init_sft_list = _scan_sft_models() or ["（未发现SFT模型）"]
                        btn_refresh_sft = gr.Button("刷新 SFT 模型列表")

                        sft_dd = gr.Dropdown(choices=["（未发现）"], value="（未发现）", label="RL 使用的 SFT 模型")
                        rl_episodes = gr.Slider(10, 5000, value=200, step=10, label="训练局数(episodes)")
                        rl_sims = gr.Slider(50, 2000, value=300, step=50, label="MCTS 模拟次数")
                        rl_gamma = gr.Slider(0.80, 0.999, value=0.99, step=0.001, label="折扣因子 γ")
                        rl_lambda = gr.Slider(0.80, 0.999, value=0.95, step=0.001, label="GAE λ")
                        rl_note = gr.Textbox(label="备注/实验标识", value="")
                        btn_rl = gr.Button("启动 强化学习 训练", variant="secondary")

                    with gr.Column(scale=1):
                        gr.Markdown("### 📜 训练日志输出")
                        train_log = gr.Textbox(lines=22, label="训练日志 / 结果", value="等待启动...", autoscroll=True)

                btn_refresh_pretrained.click(_refresh_pretrained, inputs=[gr.State(backend)], outputs=[pretrained_dd])
                btn_sft.click(_start_sft,
                              inputs=[gr.State(backend), game_dd, size_dd, sft_epochs, sft_lr, sft_batch, pretrained_dd],
                              outputs=[train_log])
                btn_refresh_sft.click(_refresh_sft_models, inputs=[gr.State(backend), game_dd], outputs=[sft_dd])
                btn_rl.click(_start_rl,
                             inputs=[gr.State(backend), game_dd, size_dd, rl_episodes, rl_sims, rl_gamma, rl_lambda,
                                     rl_note, sft_dd],
                             outputs=[train_log])

            with gr.TabItem("实战（人机 / 模型对战）"):
                with gr.Row():
                    with gr.Column(scale=1):
                        game2_dd = gr.Dropdown(choices=list(GAME_PRESETS.keys()), value="围棋(Go)", label="棋类")
                        size2_dd = gr.Dropdown(choices=[9, 13, 19], value=19, label="棋盘尺寸")
                        game2_dd.change(_on_game_change_sizes, inputs=[game2_dd], outputs=[size2_dd])

                        btn_new = gr.Button("新对局 / 重置", variant="primary")

                        gr.Markdown("### 模型管理（黑方 / 白方）")
                        btn_refresh_models = gr.Button("刷新模型列表")
                        model_a_dd = gr.Dropdown(choices=["（无可用模型）"], value="（无可用模型）", label="黑方模型")
                        model_b_dd = gr.Dropdown(choices=["（无可用模型）"], value="（无可用模型）", label="白方模型")
                        btn_load_models = gr.Button("加载/切换模型")

                        gr.Markdown("### 搜索设置")
                        sims2 = gr.Slider(50, 2000, value=300, step=50, label="MCTS 模拟次数")
                        topk2 = gr.Slider(1, 10, value=5, step=1, label="展示 Top-K 策略")

                        gr.Markdown("### 对弈控制")
                        btn_ai_move = gr.Button("AI 落子（当前行动方）")
                        btn_ai_vs_ai = gr.Button("AI vs AI （各走一步）")
                        btn_undo = gr.Button("悔棋")

                    with gr.Column(scale=1.5):
                        board_img = gr.Image(type="numpy", height=640, interactive=True, show_label=False, sources=[],
                                             elem_id="board_img")
                        state_box = gr.State()
                        info_box = gr.Textbox(label="对局信息", value="点击“新对局 / 重置”开始", lines=3)
                        v_box = gr.Textbox(label="局面估值 V(s)", value="—")
                        topk_box = gr.Textbox(label="Top-K 策略（坐标: 概率）", value="—", lines=8)

                btn_new.click(_reset_match,
                              inputs=[backend_state, game2_dd, size2_dd, model_a_dd, model_b_dd],
                              outputs=[board_img, state_box, info_box, v_box, topk_box])

                btn_refresh_models.click(_refresh_model_list, inputs=[backend_state, game2_dd],
                                         outputs=[model_a_dd, model_b_dd])

                btn_load_models.click(_load_models,
                                      inputs=[backend_state, game2_dd, model_a_dd, model_b_dd],
                                      outputs=[info_box])

                board_img.select(_board_click,
                                 inputs=[backend_state, state_box],
                                 outputs=[board_img, state_box, info_box, v_box, topk_box])

                btn_undo.click(_undo_click,
                               inputs=[backend_state, state_box],
                               outputs=[board_img, state_box, info_box, v_box, topk_box])

                btn_ai_move.click(_ai_move,
                                  inputs=[backend_state, state_box, sims2, topk2, gr.State("A")],
                                  outputs=[board_img, state_box, info_box, v_box, topk_box])

                btn_ai_vs_ai.click(_ai_vs_ai_once,
                                   inputs=[backend_state, state_box, sims2, topk2],
                                   outputs=[board_img, state_box, info_box, v_box, topk_box])

    demo.launch(server_name=server_name, server_port=server_port, share=share)


def _CSS() -> str:
    return """
    .gr-button { font-weight: 600; }
    .gradio-container { max-width: 1200px !important; margin: auto; }
    #board_img { margin-top: 8px; }
    #board_img [data-testid="image-toolbar"],
    #board_img [data-testid="image-controls"],
    #board_img .image-toolbar,
    #board_img .image-controls,
    #board_img footer,
    #board_img .tools,
    #board_img .controls,
    #board_img .edit-buttons { display: none !important; }
    """


# 独立运行调试
if __name__ == "__main__":
    from backend.backend import Backend
    launch_ui(backend=Backend(model_root="./models"), share=False)
