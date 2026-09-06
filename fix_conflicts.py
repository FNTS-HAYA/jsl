#!/usr/bin/env python3
# ===================================================
# 残ってしまった衝突マーカーを片付ける
#
#   python fix_conflicts.py
#       どのファイルのどこが衝突しているか一覧で見る
#
#   python fix_conflicts.py --show index.html
#       そのファイルの両方の中身を index.html.A / index.html.B に書き出す
#       （エディタで見比べるため。元のファイルは触らない）
#
#   python fix_conflicts.py --pick A
#       ぜんぶ A側（<<<<<<< HEAD のすぐ下）を採用して直す
#
#   python fix_conflicts.py --pick B
#       ぜんぶ B側（======= のすぐ下）を採用して直す
#
#   python fix_conflicts.py --pick A --only style.css
#       そのファイルだけ直す
#
# 直す前に .bak を作るので、間違えても戻せる。
# ===================================================

import argparse, os, re, shutil, sys

SKIP_DIRS = {'.git', 'node_modules', '__pycache__', 'legacy'}
START = re.compile(r'^<{7}[ \t]')
MID   = re.compile(r'^={7}\s*$')
END   = re.compile(r'^>{7}[ \t]')


def find_files(root='.'):
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if fn.endswith(('.bak', '.A', '.B')):
                continue
            p = os.path.join(dirpath, fn)
            try:
                with open(p, encoding='utf-8') as f:
                    text = f.read()
            except (UnicodeDecodeError, PermissionError, OSError):
                continue
            # 行頭にマーカーがある場合だけ。説明文に書かれた文字列は拾わない
            lines = text.split('\n')
            if (any(START.match(l) for l in lines)
                    and any(END.match(l) for l in lines)
                    and os.path.abspath(p) != os.path.abspath(__file__)):
                out.append(p)
    return sorted(out)


def split_sides(lines):
    """衝突ブロックを見つけて (A側だけ, B側だけ, ブロック情報) を返す"""
    a, b, blocks = [], [], []
    i, state = 0, 0          # 0=通常 1=A側 2=B側
    cur = None
    while i < len(lines):
        line = lines[i]
        if state == 0 and START.match(line):
            state = 1
            cur = {'line': i + 1, 'a': [], 'b': []}
        elif state == 1 and MID.match(line):
            state = 2
        elif state == 2 and END.match(line):
            state = 0
            blocks.append(cur)
            a.extend(cur['a']); b.extend(cur['b'])
            cur = None
        elif state == 1:
            cur['a'].append(line)
        elif state == 2:
            cur['b'].append(line)
        else:
            a.append(line); b.append(line)
        i += 1
    if state != 0:
        raise ValueError('マーカーが閉じていません')
    return a, b, blocks


def preview(lines, n=2):
    out = []
    for l in lines[:n]:
        s = l.strip()
        if s:
            out.append(s[:70])
    return ' / '.join(out) if out else '(空)'


def cmd_list(files):
    if not files:
        print('衝突マーカーは見つかりませんでした。きれいです。')
        return
    print(f'{len(files)} ファイルにマーカーが残っています\n')
    for p in files:
        lines = open(p, encoding='utf-8').read().split('\n')
        try:
            _, _, blocks = split_sides(lines)
        except ValueError as e:
            print(f'  {p}  ← {e}'); continue
        print(f'  {p}  ({len(blocks)}箇所)')
        for bl in blocks[:3]:
            print(f'      {bl["line"]}行目')
            print(f'        A: {preview(bl["a"])}')
            print(f'        B: {preview(bl["b"])}')
        if len(blocks) > 3:
            print(f'      … ほか{len(blocks)-3}箇所')
        print()
    print('次にやること:')
    print('  中身を見比べる     python fix_conflicts.py --show ファイル名')
    print('  A側を採用して直す  python fix_conflicts.py --pick A')
    print('  B側を採用して直す  python fix_conflicts.py --pick B')


def cmd_show(path):
    lines = open(path, encoding='utf-8').read().split('\n')
    a, b, blocks = split_sides(lines)
    open(path + '.A', 'w', encoding='utf-8').write('\n'.join(a))
    open(path + '.B', 'w', encoding='utf-8').write('\n'.join(b))
    print(f'{path}  衝突 {len(blocks)}箇所')
    print(f'  A側 → {path}.A  ({len(a)}行)')
    print(f'  B側 → {path}.B  ({len(b)}行)')
    print('\nVS Code で両方を開き、片方を右クリック →')
    print('「選択項目を比較」で違いが色分けされて見えます。')


def cmd_pick(files, side):
    idx = 0 if side == 'A' else 1
    for p in files:
        lines = open(p, encoding='utf-8').read().split('\n')
        try:
            sides = split_sides(lines)
        except ValueError as e:
            print(f'  飛ばす {p}: {e}'); continue
        shutil.copy2(p, p + '.bak')
        open(p, 'w', encoding='utf-8').write('\n'.join(sides[idx]))
        print(f'  直した {p}  ({len(sides[2])}箇所, {side}側を採用, 元は {p}.bak)')
    print(f'\n{len(files)} ファイルを処理しました。')
    print('ブラウザで確認して、問題なければ .bak を消してください。')
    print('戻したいときは .bak を元の名前に戻します。')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--show', metavar='FILE')
    ap.add_argument('--pick', choices=['A', 'B'])
    ap.add_argument('--only', metavar='NAME', help='ファイル名を含むものだけ対象にする')
    args = ap.parse_args()

    if args.show:
        cmd_show(args.show); sys.exit()

    files = find_files('.')
    if args.only:
        files = [f for f in files if args.only in f]

    if args.pick:
        cmd_pick(files, args.pick)
    else:
        cmd_list(files)
