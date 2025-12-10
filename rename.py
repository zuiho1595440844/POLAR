#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
功能函数文件，五子棋网络下载的的sgf文件名包含特殊字符，需要重新命名
将指定文件夹下的文件按数字命名 (1,2,3,...)。
默认：只处理文件(不含子文件夹)，按“自然顺序的文件名”排序，保留原扩展名，从1开始，不做零填充。

示例：
    python rename_numeric.py /path/to/dir
    python rename_numeric.py /path/to/dir --sort mtime --start 1 --pad 0
    python rename_numeric.py /path/to/dir --no-ext       # 不保留扩展名
    python rename_numeric.py /path/to/dir --ext .jpg,.png  # 只处理这些扩展名
    python rename_numeric.py /path/to/dir --dry-run      # 仅查看将要改名的结果
"""

import argparse
import os
import re
import sys
import uuid
from typing import List, Tuple

def natural_key(s: str):
    """用于“自然排序”的 key：按数字块与文本块混合排序，避免 '10' < '2' 的问题。"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def list_targets(
    folder: str,
    include_dirs: bool = False,
    exts: List[str] = None,
    include_hidden: bool = False
) -> List[str]:
    items = []
    for name in os.listdir(folder):
        if not include_hidden and name.startswith('.'):
            continue
        p = os.path.join(folder, name)
        if os.path.isdir(p):
            if include_dirs:
                items.append(name)
        elif os.path.isfile(p):
            if exts:
                # 扩展名过滤（大小写不敏感）
                _, e = os.path.splitext(name)
                if e.lower() not in [x.lower() for x in exts]:
                    continue
            items.append(name)
        # 其他类型（符号链接等）默认跳过
    return items

def two_phase_rename(folder: str, plan: List[Tuple[str, str]], dry_run: bool = False):
    """
    两阶段改名：
    1) 先把源文件全部改为临时名，避免与目标名冲突；
    2) 再从临时名改为最终名。
    plan: [(src_name, dst_name), ...] 仅包含 '文件名'（不含路径）
    """
    if dry_run:
        width = max(len(src) for src, _ in plan) if plan else 10
        print("[Dry-Run] 将执行以下改名：")
        for src, dst in plan:
            print(f"  {src:<{width}}  ->  {dst}")
        return

    # 生成唯一临时名映射
    tmp_suffix = f".__tmp_ren_{uuid.uuid4().hex[:8]}"
    tmp_map = []
    for src, _ in plan:
        tmp_name = src + tmp_suffix
        # 理论上不会重名；若存在（极小概率），继续追加随机段
        while os.path.exists(os.path.join(folder, tmp_name)):
            tmp_name = src + f".__tmp_{uuid.uuid4().hex[:8]}"
        tmp_map.append((src, tmp_name))

    # 阶段1：改为临时名
    for src, tmp in tmp_map:
        os.rename(os.path.join(folder, src), os.path.join(folder, tmp))

    # 阶段2：临时名 -> 目标名
    # 建立从临时名到最终名的映射
    tmp_to_dst = {}
    for (src, dst), (_, tmp) in zip(plan, tmp_map):
        # 若目标已存在（例如目录同名），抛错回滚更复杂；这里直接报错并提示人工处理
        target_path = os.path.join(folder, dst)
        if os.path.exists(target_path):
            print(f"[错误] 目标已存在，无法覆盖：{dst}", file=sys.stderr)
            print("请手动处理冲突后重试。", file=sys.stderr)
            # 尝试回滚已改名的部分
            for t, s in [(t2, s2) for (_, t2), (s2, _) in zip(tmp_map, plan)]:
                tp = os.path.join(folder, t)
                sp = os.path.join(folder, s)
                if os.path.exists(tp) and not os.path.exists(sp):
                    try:
                        os.rename(tp, sp)
                    except Exception as e:
                        print(f"[回滚失败] {t} -> {s}: {e}", file=sys.stderr)
            sys.exit(1)
        tmp_to_dst[tmp] = dst

    for tmp, dst in tmp_to_dst.items():
        os.rename(os.path.join(folder, tmp), os.path.join(folder, dst))

def main():
    parser = argparse.ArgumentParser(description="将文件按数字顺序重命名。")
    parser.add_argument("folder", help="目标文件夹路径")
    parser.add_argument("--start", type=int, default=1, help="起始编号（默认：1）")
    parser.add_argument("--pad", type=int, default=0, help="零填充位数（默认：0，不填充）")
    parser.add_argument("--sort", choices=["name", "mtime", "ctime"], default="name",
                        help="排序方式：name(自然顺序文件名) / mtime(修改时间) / ctime(创建/变更时间)")
    parser.add_argument("--reverse", action="store_true", help="是否倒序")
    parser.add_argument("--include-dirs", action="store_true", help="也重命名子文件夹（默认仅文件）")
    parser.add_argument("--include-hidden", action="store_true", help="包含隐藏项（以.开头）")
    parser.add_argument("--no-ext", action="store_true", help="不保留原扩展名")
    parser.add_argument("--ext", type=str, default="",
                        help="仅处理这些扩展名，逗号分隔，如：.jpg,.png（大小写不敏感）")
    parser.add_argument("--dry-run", action="store_true", help="试运行（仅打印不会真正改名）")
    args = parser.parse_args()

    folder = os.path.abspath(args.folder)
    if not os.path.isdir(folder):
        print(f"[错误] 非法文件夹：{folder}", file=sys.stderr)
        sys.exit(1)
    # 简单安全防护：避免误操作根目录等极端情况
    if folder in ("/", "C:\\", "C:/"):
        print("[错误] 为避免风险，不支持在系统根目录直接执行。", file=sys.stderr)
        sys.exit(1)

    exts = None
    if args.ext.strip():
        exts = [e.strip() if e.strip().startswith('.') else '.' + e.strip()
                for e in args.ext.split(',') if e.strip()]

    items = list_targets(
        folder,
        include_dirs=args.include_dirs,
        exts=exts,
        include_hidden=args.include_hidden
    )

    if not items:
        print("[提示] 未找到需要处理的对象。")
        return

    # 排序
    if args.sort == "name":
        items.sort(key=natural_key, reverse=args.reverse)
    elif args.sort == "mtime":
        items.sort(key=lambda n: os.path.getmtime(os.path.join(folder, n)), reverse=args.reverse)
    else:  # ctime
        items.sort(key=lambda n: os.path.getctime(os.path.join(folder, n)), reverse=args.reverse)

    # 生成改名计划
    plan = []
    idx = args.start
    width = args.pad if args.pad and args.pad > 0 else 0
    for name in items:
        root, ext = os.path.splitext(name)
        if args.no_ext:
            new_name = f"{idx:0{width}d}" if width > 0 else str(idx)
        else:
            new_name = (f"{idx:0{width}d}{ext}" if width > 0 else f"{idx}{ext}")
        # 避免无谓改名（同名同扩展的情况，不在两阶段内也可跳过）
        if new_name == name:
            idx += 1
            continue
        plan.append((name, new_name))
        idx += 1

    if not plan:
        print("[提示] 所有项目在当前规则下已经是目标命名，无需更改。")
        return

    two_phase_rename(folder, plan, dry_run=args.dry_run)
    if not args.dry_run:
        print(f"[完成] 共处理 {len(plan)} 个项目。")

if __name__ == "__main__":
    main()
