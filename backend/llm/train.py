# backend/llm/train.py
# -*- coding: utf-8 -*-
"""
统一 SFT 训练入口（Go / Gomoku / Jiu）

- 预训练底模：HF 本地目录（deepseek / llama / gpt2 / 其它）
  * 兼容仅有 TensorFlow 权重的目录（from_tf=True）
- 数据目录（固定）：
    go      -> data/sgf/sgf_go/**/*.sgf
    gomoku  -> data/sgf/sgf_gomoku/**/*.sgf
    jiu     -> data/sgf/sgf_jiu.csv
- 实时输出训练日志（yield）
- 保存至：models/sft/{game}-{base}-{epoch/lr/batch}-{YYYYmmdd-HHMMSS}
"""
from __future__ import annotations
import os, re, csv, glob, json, time
from typing import List, Tuple, Iterable, Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
from tqdm import tqdm


# ------------------ 基础工具 ------------------
def _ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _save_dir_name(game: str, base_model_name: str, epochs: int, lr: float, batch_size: int) -> str:
    return f"{game}-{base_model_name}-epoch{epochs}-lr{lr}-batch{batch_size}-{_ts()}"

def _default_data_path(game: str) -> str:
    if game == "go":
        return "data/sgf/sgf_go"
    if game == "gomoku":
        return "data/sgf/sgf_gomoku"
    if game == "jiu":
        return "data/sgf/sgf_jiu.csv"
    raise ValueError(f"未知棋种: {game}")


# ------------------ Go / Gomoku：SGF -> 整数 token ------------------
def sgf_to_tokens_general(sgf_text: str, board_size: int) -> List[int]:
    """
    将 SGF 文本解析为整数 token 序列：
      - 提取 B[ab] / W[cd]（大小写兼容）
      - 超界坐标会被过滤
    编码规则：
      黑手: idx = r*B + c
      白手: idx = (r*B + c) + B*B
    """
    text = sgf_text.lower()
    moves = re.findall(r'(b|w)\[([a-z]{1,2})\]', text)
    toks: List[int] = []
    off = board_size * board_size

    for player, pos in moves:
        if len(pos) != 2:
            continue
        r = ord(pos[0]) - ord('a')
        c = ord(pos[1]) - ord('a')
        if not (0 <= r < board_size and 0 <= c < board_size):
            continue
        base = r * board_size + c
        toks.append(base if player == 'b' else base + off)
    return toks


def load_go_gomoku_seqs(root: str, board_size: int) -> List[List[int]]:
    """
    递归加载 .sgf 文件并解析为整数 token 序列
    """
    patt1 = os.path.join(root, "**", "*.sgf")
    patt2 = os.path.join(root, "**", "*.SGF")
    patt3 = os.path.join(root, "**", "*.[sS][gG][fF]")
    files = set(glob.glob(patt1, recursive=True) + glob.glob(patt2, recursive=True) + glob.glob(patt3, recursive=True))
    seqs: List[List[int]] = []
    for fp in sorted(files):
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                sgf = f.read()
            toks = sgf_to_tokens_general(sgf, board_size)
            if toks:
                seqs.append(toks)
        except Exception:
            continue
    return seqs


class IntTokenDataset(Dataset):
    """
    用于 Go/Gomoku 的整数 token 数据集：
      输入:  s[:-1]
      标签:  s[ 1:]
    """
    def __init__(self, seqs: List[List[int]], max_len: int = 512):
        self.samples = []
        for s in seqs:
            if len(s) < 2:  # 至少一对
                continue
            if len(s) > max_len:
                s = s[:max_len]
            self.samples.append((s[:-1], s[1:]))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        inp, lab = self.samples[idx]
        return torch.LongTensor(inp), torch.LongTensor(lab)


def _collate_int_factory(pad_id: int):
    """
    注意：pad_id 不能落在真实 token 空间里（否则会把真实标签当作 ignore_index）
    """
    def _collate_int(batch):
        inputs, labels = zip(*batch)
        inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=pad_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=pad_id)
        attn = (inputs != pad_id).long()
        return {"input_ids": inputs, "labels": labels, "attention_mask": attn}
    return _collate_int


# ------------------ 久棋：CSV -> 词级 token 行 ------------------
def load_jiu_lines(csv_path: str) -> List[str]:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"未找到久棋CSV: {csv_path}")
    lines: List[str] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            joined = " ".join([x for x in row if x and x.strip()])
            joined = re.sub(r"[，,]+", " ", joined)
            joined = re.sub(r"\s+", " ", joined).strip()
            if joined:
                lines.append(joined)
    return lines


class JiuDataset(Dataset):
    """
    直接使用 tokenizer 的词表：
      输入:  token_ids[:-1]
      标签:  token_ids[ 1:]
    """
    def __init__(self, token_lines: List[str], tokenizer, max_len: int = 512):
        self.samples = []
        for ln in token_lines:
            toks = ln.split()
            if not toks:
                continue
            ids = tokenizer.convert_tokens_to_ids(toks)
            # 如果有未知词，通常说明 tokenizer 未添加该词，跳过该样本
            if tokenizer.unk_token_id is not None and tokenizer.unk_token_id in ids:
                continue
            if len(ids) < 2:
                continue
            inp = ids[:-1][:max_len]
            lab = ids[1:][:max_len]
            self.samples.append((inp, lab))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        inp, lab = self.samples[idx]
        return torch.LongTensor(inp), torch.LongTensor(lab)


def collate_jiu(batch, pad_id: int):
    inputs, labels = zip(*batch)
    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=pad_id)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=pad_id)
    attn = (inputs != pad_id).long()
    return {"input_ids": inputs, "labels": labels, "attention_mask": attn}


# ------------------ 底模加载（兼容 TF 权重） ------------------
def _load_base_model_and_tokenizer(base_model_dir: str, device: torch.device):
    """
    优先以 PyTorch 权重加载；若仅有 TF 权重，则 fallback 到 from_tf=True
    """
    base_model_dir = os.path.abspath(base_model_dir)
    has_pt = any(os.path.isfile(os.path.join(base_model_dir, fn)) for fn in ["pytorch_model.bin", "model.safetensors"])
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True, local_files_only=True)
    try:
        if has_pt:
            model = AutoModelForCausalLM.from_pretrained(base_model_dir, trust_remote_code=True, local_files_only=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(base_model_dir, trust_remote_code=True,
                                                        local_files_only=True, from_tf=True)
    except Exception:
        # 环境允许联网时可去掉 local_files_only=True；此处保守处理
        raise
    # 确保有 pad token（尤其是 GPT2 家族）
    need_resize = False
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            need_resize = True
    if need_resize:
        model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    return tokenizer, model


# ------------------ 训练主入口（生成器：实时日志） ------------------
def run_sft(
    base_model_dir: str,
    game: str,
    board_size: int,
    epochs: int,
    lr: float,
    batch_size: int,
    save_root: str = "models/sft",
):
    assert game in {"go", "gomoku", "jiu"}, f"未知棋种: {game}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 底模与分词器（含 TF 兼容）
    base_model_dir = os.path.abspath(base_model_dir)
    base_model_name = os.path.basename(os.path.normpath(base_model_dir))
    try:
        tokenizer, model = _load_base_model_and_tokenizer(base_model_dir, device)
    except Exception as e:
        yield f"[SFT] 底模加载失败：{e}"
        return

    # ------------------ 数据 ------------------
    data_path = _default_data_path(game)

    if game in {"go", "gomoku"}:
        seqs = load_go_gomoku_seqs(data_path, board_size)
        if not seqs:
            yield f"[SFT] 失败：未在 {data_path} 找到任何 .sgf 样本"
            return

        N = board_size * board_size
        pad_id = 2 * N  # 预留一个“超界”id作为 padding（不与真实 token 冲突）
        need_vocab = max(model.config.vocab_size, pad_id + 1)
        model.resize_token_embeddings(need_vocab)

        dataset = IntTokenDataset(seqs, max_len=512)
        loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True,
                            collate_fn=_collate_int_factory(pad_id))
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_id)

        yield f"[SFT] 已加载 {len(seqs)} 盘 {game}（board={board_size}），词表扩展至 ≥ {need_vocab}"
    else:
        lines = load_jiu_lines(data_path)
        if not lines:
            yield f"[SFT] 失败：未从 {data_path} 解析到样本"
            return
        # 根据 CSV 中出现的 token 扩表（非 special）
        vocab_tokens = set()
        for ln in lines:
            vocab_tokens.update(ln.split())
        added = tokenizer.add_tokens(sorted(list(vocab_tokens)), special_tokens=False)
        if added > 0:
            model.resize_token_embeddings(len(tokenizer))
            with torch.no_grad():
                emb = model.get_input_embeddings().weight
                # 简单初始化新加词的 embedding
                emb[-added:] = torch.nn.init.normal_(torch.empty_like(emb[-added:]))
        dataset = JiuDataset(lines, tokenizer, max_len=512)
        loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True,
                            collate_fn=lambda b: collate_jiu(b, tokenizer.pad_token_id))
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

        yield f"[SFT] 已加载 {len(lines)} 条久棋样本；新增词 {added} 个，当前词表大小 {len(tokenizer)}"

    # 预处理完成 → 立即提示
    yield "预处理完成，开始训练..."

    # ------------------ 保存目录与配置 ------------------
    out_dir = os.path.join(save_root, _save_dir_name(game, base_model_name, int(epochs), float(lr), int(batch_size)))
    _ensure_dir(out_dir)
    with open(os.path.join(out_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "base_model_dir": base_model_dir,
            "game": game,
            "board_size": board_size,
            "epochs": int(epochs),
            "lr": float(lr),
            "batch_size": int(batch_size),
            "data_path": data_path,
        }, f, ensure_ascii=False, indent=2)

    # ------------------ 优化器 / AMP ------------------
    optimizer = AdamW(model.parameters(), lr=float(lr))
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # 日志文件
    log_path = os.path.join(out_dir, "training_log.txt")

    # ------------------ 训练循环 ------------------
    num_epochs = int(epochs)
    for ep in range(num_epochs):
        model.train()
        tot_loss = 0.0
        tot_acc  = 0.0
        steps    = 0
        t0 = time.time()

        pbar = tqdm(loader, desc=f"Epoch {ep+1}/{num_epochs}", leave=False)
        for batch in pbar:
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attn   = batch["attention_mask"].to(device)

            # 我们的 Dataset 已经对齐 (inputs, labels)；此处不再“再位移”
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out = model(input_ids=inputs, attention_mask=attn)
                logits = out.logits  # (B, L, V)
                L = labels.size(1)
                if logits.size(1) != L:
                    logits = logits[:, :L, :]
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            # 计算 token 级准确率（忽略 pad）
            with torch.no_grad():
                preds = torch.argmax(logits, dim=-1)
                mask_val = (labels != (tokenizer.pad_token_id if game == "jiu" else (2 * board_size * board_size)))
                denom = mask_val.sum()
                if denom.item() > 0:
                    acc = (preds.eq(labels) & mask_val).sum().float() / denom.float()
                    acc_val = acc.item()
                else:
                    acc_val = 0.0

            tot_loss += float(loss.detach().cpu().item())
            tot_acc  += acc_val
            steps    += 1
            pbar.set_postfix({"loss": f"{tot_loss/steps:.4f}", "acc": f"{tot_acc/steps:.4f}"})

        dt = time.time() - t0
        avg_loss = tot_loss / max(1, steps)
        avg_acc  = tot_acc  / max(1, steps)
        line = f"Epoch {ep+1}/{num_epochs}: Loss={avg_loss:.4f}, Accuracy={avg_acc:.4f}, Time={dt:.2f}s"

        # 写日志文件
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

        # 保存（覆盖该目录最新权重/分词器）
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)

        # 实时输出
        yield line

    done = f"[SFT] 训练完成，模型已保存至：{out_dir}"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(done + "\n")
    yield done
