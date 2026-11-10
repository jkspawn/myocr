"""Rebuild paddleocr_table.ipynb from input/*.png.  Usage: python tools/build_notebook.py <project_dir>"""
import base64, json, sys, textwrap
from pathlib import Path

proj = Path(sys.argv[1])
images = {p.name: base64.b64encode(p.read_bytes()).decode() for p in sorted((proj / "input").glob("*.png"))}

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(s).strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": textwrap.dedent(s).strip("\n")})

md("""
# PaddleOCR로 표 이미지 글자 인식하기

1. PaddlePaddle·PaddleOCR 설치
2. 영어 인식 모델 내려받기
3. 표 이미지에서 글자와 위치 인식
4. 인식한 글자를 행 단위로 다시 묶어 CSV로 저장
5. 선택한 표 Table 1의 행·열을 연결하고, 원본 대조 결과(확인 필요·수정 내역)를 문서로 남기기
6. 손으로 검증한 정답과 비교
7. 출처·입력·모델·패키지·실행 환경·설정을 `outputs/manifest.json`에 기록

> 런타임은 CPU로도 충분합니다. 위에서부터 차례로 실행하세요(**런타임 → 모두 실행**).
""")

md("## 1. 설치\n설치가 끝나면 버전이 출력됩니다. 경고가 나와도 마지막 줄에 버전이 찍히면 정상입니다.")
code("""
from datetime import datetime, timezone
RUN_STARTED = datetime.now(timezone.utc).isoformat(timespec="seconds")   # manifest.json에 남는 시작 시각
PIP_PACKAGES = "paddlepaddle paddleocr"

!pip install -q {PIP_PACKAGES}

import paddle, paddleocr
print("paddlepaddle", paddle.__version__)
print("paddleocr   ", paddleocr.__version__)
""")

md("## 2. 표 이미지 준비\n이미지가 노트북 안에 들어 있으므로 따로 업로드할 필요가 없습니다.")
code("IMAGES_B64 = " + json.dumps(images, indent=1) + """

import base64
from pathlib import Path
from IPython.display import Image, display

IN, OUT = Path("input"), Path("outputs/ocr")
DIRS = {k: OUT / k for k in ("input", "raw", "text", "vis", "tables")}
IN.mkdir(exist_ok=True)
for d in DIRS.values(): d.mkdir(parents=True, exist_ok=True)
for name, b64 in IMAGES_B64.items():
    (IN / name).write_bytes(base64.b64decode(b64))
    print(name); display(Image(filename=str(IN / name), width=600))
""")

md("""
### 2-1. 출처와 원문 페이지
이미지를 어디서 가져왔는지 적어 두세요. 여기 적은 내용이 그대로 `outputs/manifest.json`의 `sources`에 남습니다.
**모르는 칸은 `None`으로 두세요.** 비워 둔 칸은 manifest의 `missing`에 이름이 남아, 나중에 채워야 할 곳을 알 수 있습니다.
""")
code(r"""
SOURCES = {
    "census-table.png": {
        "title": None,       # 자료 이름 (보고서·통계집 제목)
        "publisher": None,   # 발행 기관
        "year": None,        # 발행 연도
        "url": None,         # 자료를 내려받은 주소
        "page": None,        # 원문 인쇄 쪽 번호
        "pdf_page": None,    # PDF 안의 쪽 번호 (인쇄 쪽 번호와 다를 때)
        "table": "Table 1",  # 원문 안의 표 번호
        "accessed": None,    # 내려받은 날짜 (예: "2026-11-10")
        "notes": None,       # 그 밖의 메모
    },
    "fraser-table.png": {
        "title": None, "publisher": None, "year": None, "url": None,
        "page": None, "pdf_page": None, "table": None, "accessed": None, "notes": None,
    },
}
""")

md("""
## 3. 영어 인식 모델 내려받기
`lang="en"`으로 만들면 글자 위치를 찾는 **검출 모델**과 영어 **인식 모델**을 처음 한 번 내려받습니다.
이 표들은 반듯하게 스캔된 문서라 방향 보정·왜곡 보정 모델은 끕니다.
""")
code("""
from paddleocr import PaddleOCR

OCR_KWARGS = dict(            # manifest.json의 settings.ocr에 그대로 남음
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    enable_mkldnn=False,   # CPU에서 predict 중 나는 oneDNN 오류 회피
)
ocr = PaddleOCR(**OCR_KWARGS)

# 내려받은 모델 폴더 확인
for d in sorted((Path.home() / ".paddlex" / "official_models").glob("*")):
    print("✔", d.name)
""")

md("""
## 4. 글자 인식과 저장
이미지마다 아래 파일을 `outputs/ocr/`에 저장합니다.

| 폴더 | 파일 | 내용 |
|---|---|---|
| `input/` | `<이름>.png` | 입력 표 이미지 사본 |
| `raw/` | `<이름>_ocr_raw.json` | PaddleOCR이 내놓은 원시 결과 그대로 |
| `text/` | `<이름>_text_boxes.csv` | 글자 덩어리마다 번호·텍스트·신뢰도·상자 좌표·꼭짓점 좌표 |
| `vis/` | `<이름>_regions.png` | 인식 영역에 번호를 붙여 그린 이미지 (초록 ≥ 0.9, 주황 < 0.9) |
""")
code("""
import csv, json, shutil
import numpy as np
from PIL import Image as PILImage, ImageDraw

def save_text_boxes(res, path):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "text", "score", "x1", "y1", "x2", "y2", "poly"])
        for i, (t, s, b, p) in enumerate(zip(res["rec_texts"], res["rec_scores"], res["rec_boxes"], res["rec_polys"])):
            w.writerow([i, t, f"{s:.4f}", *map(int, b), json.dumps(np.asarray(p).astype(int).tolist())])

def save_regions(img_path, res, path):
    im = PILImage.open(img_path).convert("RGB")
    d = ImageDraw.Draw(im)
    for i, (s, p) in enumerate(zip(res["rec_scores"], res["rec_polys"])):
        pts = [tuple(map(int, q)) for q in np.asarray(p)]
        color = (0, 170, 0) if s >= 0.9 else (255, 140, 0)
        d.polygon(pts, outline=color, width=2)
        d.text((pts[0][0], max(pts[0][1] - 11, 0)), str(i), fill=(220, 0, 0))
    im.save(path)

results = {}
for img in sorted(IN.glob("*.png")):
    res = ocr.predict(str(img))[0]
    results[img.stem] = res
    shutil.copy(img, DIRS["input"] / img.name)
    res.save_to_json(str(DIRS["raw"] / f"{img.stem}_ocr_raw.json"))
    save_text_boxes(res, DIRS["text"] / f"{img.stem}_text_boxes.csv")
    save_regions(img, res, DIRS["vis"] / f"{img.stem}_regions.png")
    print(f"{img.name}: 글자 덩어리 {len(res['rec_texts'])}개")

for f in sorted(DIRS["vis"].glob("*.png")):
    print(f.name); display(Image(filename=str(f), width=700))

print()
for f in sorted(OUT.rglob("*")):
    if f.is_file(): print(f"{f}  ({f.stat().st_size:,} bytes)")
""")

md("""
## 5. 행 단위로 다시 묶기
OCR은 글자 덩어리마다 상자 좌표를 줍니다. 세로 위치가 비슷한 상자끼리 한 행으로 묶고,
행 안에서는 왼쪽부터 정렬합니다. 목차처럼 이어진 점(`.....`)은 지웁니다.
""")
code(r"""
import re
ROW_GAP = 0.6   # 상자 세로 중심이 중앙 상자 높이 × ROW_GAP 보다 벌어지면 새 행

def clean(t):
    return re.sub(r"\s*\.{2,}[.\s]*", " ", t).strip(" .")

def to_rows(res):
    # 항목 = (상자 [x1,y1,x2,y2], 정리한 텍스트, 신뢰도, OCR id)
    items = [(np.asarray(b), clean(t), s, i)
             for i, (b, t, s) in enumerate(zip(res["rec_boxes"], res["rec_texts"], res["rec_scores"])) if clean(t)]
    items.sort(key=lambda it: (it[0][1] + it[0][3]) / 2)
    h = np.median([it[0][3] - it[0][1] for it in items])
    rows, cur, cur_y = [], [], None
    for it in items:
        y = (it[0][1] + it[0][3]) / 2
        if cur and y - cur_y > h * ROW_GAP:
            rows.append(cur); cur = []
        cur.append(it); cur_y = y
    rows.append(cur)
    return [sorted(r, key=lambda it: it[0][0]) for r in rows]

tables = {}
for name, res in results.items():
    rows = to_rows(res)
    tables[name] = [[it[1] for it in r] for r in rows]
    with open(DIRS["tables"] / f"{name}.csv", "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(tables[name])
    print(f"\n=== {name} ({len(rows)}행) ===")
    for r in rows:
        print(" | ".join(f"{it[1]} ({it[2]:.2f})" if it[2] < 0.9 else it[1] for it in r))
print("\n(괄호 안 숫자 = 신뢰도 0.9 미만인 글자)")
""")

md("""
## 6. 선택한 표 Table 1 구조화 → `outputs/table1/`
행 단위로 묶은 OCR 결과를 **행과 열에 연결**하고, 원본과 대조할 항목을 골라 문서에 남깁니다.

- **열:** 이미지의 세로 괘선 위치로 열 경계를 찾고, 상자 가운데가 어느 칸에 드는지로 열을 정합니다. 괘선을 못 찾으면 기본 비율을 씁니다.
- **머리글:** 첫 숫자 행 위의 글자를 열마다 위에서부터 이어 붙여 열 이름으로 씁니다. `Table 1`은 표 제목으로 뺍니다.
- **행 계층:** 첫 행은 총계(0단계)입니다. 나머지는 들여쓰기로 1단계·2단계를 나누고, 2단계 행은 바로 위 1단계 행의 하위 행으로 연결합니다(`parent_id`).
- **각주:** 위첨자가 숫자에 붙어 `17,635,426`처럼 읽힌 칸은 하위 행 합계가 상위 행과 맞도록 앞자리 각주 번호를 떼어 냅니다. 이 자동 수정도 수정 내역에 남깁니다.

### 확인 필요 항목은 이렇게 골라집니다
| 이유 | 기준 |
|---|---|
| OCR 신뢰도 낮음 | 칸을 만든 OCR 상자 중 하나라도 신뢰도 < `CONF_MIN` (기본 0.9) |
| 상자 여러 개 합침 | 한 칸에 OCR 상자가 2개 이상 들어감 |
| 빈 칸 | 숫자 열이 비어 있음 (원본에도 값이 없는지 확인) |
| 자동 수정 | 각주 번호를 자동으로 떼어 낸 칸 |
| 합계 불일치 | 하위 행 합계가 상위 행 값과 다름 (검토 반영 후 다시 계산) |
| 검토자 표시 | 아래 `REVIEW`에서 직접 `"확인 필요"`로 표시한 칸 |

### 만들어지는 파일
| 파일 | 내용 |
|---|---|
| `table.csv` | 연결된 표 (검토 수정 반영). `check` 열에 아직 확인이 필요한 열 이름 |
| `review_items.csv` | 확인 필요 항목과 상태 (`확인 필요` / `확인함` / `수정함`) |
| `changes.csv` | 수정 내역: 언제, 어느 칸을, 무엇에서 무엇으로, 왜, 누가 (자동·검토자 모두) |
| `document.md` | 표, 합계 검증, 확인 필요, 수정 내역, OCR 출처를 한 문서로 |
""")

md("""
### 6-1. 원본 대조 기록
**처음에는 비워 둔 채로 실행하세요.** 6-2 셀이 확인 필요 항목을 출력하면서, 아래 `REVIEW`에 붙여 넣을 줄도 같이 보여 줍니다.
원본 이미지(`outputs/ocr/vis/census-table_regions.png`)와 대조한 뒤 줄을 골라 붙여 넣고, 조치·값·메모를 채워 **이 셀과 6-2 셀을 다시 실행**하면 됩니다.

항목 형식: `(row_id, 열, 조치, 값, 메모)` — 날짜를 따로 남기려면 6번째에 `"2026-11-10"`처럼 넣습니다.

| 칸 | 넣는 값 |
|---|---|
| `row_id` | `table.csv`의 `row_id` |
| 열 | `0` = 행 이름, `1`·`2` = 숫자 열. 열 이름 앞부분(`"Population"`)도 됩니다 |
| 조치 | `"확인"` 원본과 같음 → 확인 필요 해제 · `"수정"` 값을 고침 → 수정 내역에 남음 · `"확인 필요"` 직접 표시 |
| 값 | `"수정"`일 때 새 값 (`7635426`, `"7,635,426"` 모두 됨, 빈 칸은 `None`). 그 밖에는 `None` |
| 메모 | 근거. 문서의 확인 필요·수정 내역에 그대로 남습니다 |
""")
code(r"""
REVIEWER = ""        # 검토자 이름 (문서에 남음)
REVIEW_DATE = ""     # 비우면 오늘 날짜
CONF_MIN = 0.9       # 이 값보다 신뢰도가 낮은 칸은 확인 필요

REVIEW = [
    # (row_id, 열, 조치, 값, 메모[, 날짜])
    # (7, 2, "확인", None, "원본 ¹7,635,426 — 각주 1 맞음"),
    # (3, 1, "수정", 590884, "원본 대조: 590,884"),
    # (9, 2, "확인 필요", None, "추정치라 출처 재확인"),
]
""")

md("### 6-2. 표 만들기와 문서 기록")

code(r"""
from itertools import combinations

T1_NAME = "census-table"
T1_OUT = Path("outputs/table1"); T1_OUT.mkdir(parents=True, exist_ok=True)
SUP = {"¹": "1", "²": "2", "³": "3"}
N_COLS = 3

def column_bounds(img_path):
    dark = np.asarray(PILImage.open(img_path).convert("L")) < 128
    H, W = dark.shape
    xs = [x for x in np.where(dark.mean(axis=0) > 0.5)[0] if W * 0.05 < x < W * 0.95]
    groups = []
    for x in xs:
        if groups and x - groups[-1][-1] <= 8: groups[-1].append(x)
        else: groups.append([x])
    if len(groups) >= N_COLS - 1:
        return [float(np.mean(g)) for g in groups[:N_COLS - 1]], "세로 괘선"
    return [W * 0.64, W * 0.815], "기본 비율"

def col_of(box, bounds):
    return int(sum((box[0] + box[2]) / 2 > b for b in bounds))

def nice_header(parts):
    t = " ".join(parts).strip(" .")
    return t.title() if t.isupper() else t

def fmt(v): return "" if v is None else f"{v:,}"

res = results[T1_NAME]
bounds, bounds_how = column_bounds(IN / f"{T1_NAME}.png")

title, header_parts, body, footer = "Table 1", [[] for _ in range(N_COLS)], [], []
for row in to_rows(res):
    cells = [[] for _ in range(N_COLS)]
    for it in row:
        cells[col_of(it[0], bounds)].append(it)
    num_txt = [" ".join(it[1] for it in c) for c in cells[1:]]
    has_num = any(re.search(r"\d", t) for t in num_txt)
    has_words = any(re.search(r"[A-Za-z]{3,}", t) for t in num_txt)
    # 본문은 "3, 743, 306"처럼 천 단위로 끊긴 숫자가 처음 나오는 행부터 (머리글의 "1910"은 제외)
    if not body and not any(re.search(r"\d{1,3}(?:,\s?\d{3})+", t) for t in num_txt):
        for k, c in enumerate(cells):
            for it in c:
                if k == 0 and re.match(r"(?i)^table\s*\d+", it[1]): title = it[1]
                else: header_parts[k].append(it)
    elif body and (has_words or not has_num):
        footer += [it for c in cells for it in c]
    else:
        body.append(cells)

# 머리글 줄들은 한 행으로 묶이기 쉬우므로 열마다 위→아래 순서로 다시 이어 붙인다
headers = [nice_header([it[1] for it in sorted(p, key=lambda it: it[0][1])]) or f"col{k}"
           for k, p in enumerate(header_parts)]

# 각주: "¹ Population in 1903." → {"1": "Population in 1903."}
notes = {}
for n, it in enumerate(sorted(footer, key=lambda it: it[0][0]), start=1):
    t = it[1].translate(str.maketrans(SUP))
    m = re.match(r"^\s*(\d)\s*(?=[A-Za-z])(.*)$", t)
    notes[m.group(1) if m else str(n)] = (m.group(2) if m else t).strip()

recs = []
for r_id, cells in enumerate(body):
    lab = cells[0]
    rec = {"row_id": r_id, "label": " ".join(it[1] for it in lab), "x1": min(it[0][0] for it in lab) if lab else 0,
           "ids": {}, "score": {}, "value": {}, "marker": {}}
    for k in range(N_COLS):
        rec["ids"][k] = [it[3] for it in cells[k]]
        rec["score"][k] = min((it[2] for it in cells[k]), default=None)
        if k == 0: continue
        t = " ".join(it[1] for it in cells[k])
        sup = next((SUP[c] for c in t if c in SUP), None)
        d = re.sub(r"\D", "", t.translate(str.maketrans({c: "" for c in SUP})))
        rec["value"][k] = int(d) if d else None
        rec["marker"][k] = sup
    recs.append(rec)

base_x = min(r["x1"] for r in recs[1:])
last1 = None
for r in recs:
    if r["row_id"] == 0: r["level"], r["parent_id"] = 0, None
    elif r["x1"] < base_x + 20: r["level"], r["parent_id"] = 1, 0; last1 = r["row_id"]
    else: r["level"], r["parent_id"] = 2, last1


# ── 칸 단위 검토 기록 ─────────────────────────────────────────
from datetime import date
TODAY = REVIEW_DATE or date.today().isoformat()
flags = {}          # (row_id, k) -> [이유, ...]
forced_open = {}    # 검토자가 "확인 필요"로 직접 표시한 칸 -> 메모
resolved = {}       # (row_id, k) -> {"status", "memo", "date"}
changes = []        # 수정 내역

def cid(row_id, k): return f"R{row_id}-C{k}"
def flag(r, k, why): flags.setdefault((r["row_id"], k), []).append(why)
def get_val(r, k): return r["label"] if k == 0 else r["value"][k]
def show(r, k):
    v = get_val(r, k)
    return (v or "") if k == 0 else fmt(v)
def log(r, k, before, after, kind, reason, by, when):
    changes.append({"date": when, "item_id": cid(r["row_id"], k), "row_id": r["row_id"], "row": r["label"],
                    "column": headers[k], "before": before, "after": after, "kind": kind, "reason": reason, "by": by})

# 1) OCR 자체에서 보이는 의심 칸
for r in recs:
    for k in range(N_COLS):
        s = r["score"][k]
        if s is not None and s < CONF_MIN: flag(r, k, f"OCR 신뢰도 {s:.2f}")
        if len(r["ids"][k]) > 1: flag(r, k, f"OCR 상자 {len(r['ids'][k])}개를 합침")
        if k > 0 and r["value"][k] is None: flag(r, k, "빈 칸 — 원본에도 값이 없는지 확인")

# 2) 합계 검증 (+ 각주 번호가 숫자에 붙은 칸 자동 수정)
def sum_checks(autofix):
    out = []
    for lvl in (2, 1):
        for p in [r for r in recs if any(c["parent_id"] == r["row_id"] and c["level"] == lvl for c in recs)]:
            kids = [c for c in recs if c["parent_id"] == p["row_id"]]
            for k in range(1, N_COLS):
                want, got = p["value"][k], sum(c["value"][k] or 0 for c in kids)
                fixed = []
                if autofix and want is not None and got != want:
                    cands = [c for c in kids if c["value"][k] and not c["marker"][k]
                             and len(str(c["value"][k])) > 1 and str(c["value"][k])[0] in notes]
                    for n in range(1, len(cands) + 1):
                        combo = next((cb for cb in combinations(cands, n)
                                      if got - sum(c["value"][k] - int(str(c["value"][k])[1:]) for c in cb) == want), None)
                        if combo:
                            for c in combo:
                                before = c["value"][k]
                                c["marker"][k], c["value"][k] = str(before)[0], int(str(before)[1:])
                                why = f"각주 번호 {c['marker'][k]} 자동 분리 ({fmt(before)} → {fmt(c['value'][k])})"
                                flag(c, k, why)
                                log(c, k, fmt(before), fmt(c["value"][k]), "자동", why + " — 하위 합계를 상위 값에 맞춤",
                                    "notebook", date.today().isoformat())
                            fixed, got = [c["label"] for c in combo], want
                            break
                out.append({"parent": p, "k": k, "col": headers[k], "want": want, "got": got,
                            "ok": want == got, "fixed": fixed})
    return out

auto_checks = sum_checks(autofix=True)

# 3) 원본 대조 기록(REVIEW) 반영
def col_index(c):
    if isinstance(c, int) and 0 <= c < N_COLS: return c
    hit = [k for k, h in enumerate(headers) if isinstance(c, str) and h.lower().startswith(c.lower())]
    if len(hit) == 1: return hit[0]
    raise ValueError(f"열 {c!r}을 찾을 수 없습니다. 0~{N_COLS - 1} 또는 {headers} 중 하나의 앞부분을 쓰세요.")

def parse_value(k, v):
    if k == 0 or v is None: return v
    d = re.sub(r"\D", "", str(v))
    return int(d) if d else None

review_errors = []
for n, entry in enumerate(REVIEW, start=1):
    try:
        row_id, col, action, value, memo, *rest = list(entry) + [None] * (5 - len(entry))
        when = rest[0] if rest and rest[0] else TODAY
        if not (isinstance(row_id, int) and 0 <= row_id < len(recs)):
            raise ValueError(f"row_id {row_id!r}가 없습니다 (0~{len(recs) - 1}).")
        r, k = recs[row_id], col_index(col)
        key, memo = (row_id, k), memo or ""
        if action == "수정":
            new = parse_value(k, value)
            before = show(r, k)
            if new == get_val(r, k):      # 값이 같으면 수정이 아니라 확인으로 남긴다
                resolved[key] = {"status": "확인함", "memo": memo or "수정 값이 OCR 값과 같음", "date": when}
            else:
                if k == 0: r["label"] = new
                else: r["value"][k] = new
                log(r, k, before, show(r, k), "검토", memo, REVIEWER or "검토자", when)
                resolved[key] = {"status": "수정함", "memo": memo, "date": when}
            forced_open.pop(key, None)
        elif action == "확인":
            resolved[key] = {"status": "확인함", "memo": memo, "date": when}
            forced_open.pop(key, None)
        elif action == "확인 필요":
            forced_open[key] = memo or "검토자 표시"
            resolved.pop(key, None)
        else:
            raise ValueError(f"조치 {action!r}는 '확인' · '수정' · '확인 필요' 중 하나여야 합니다.")
    except Exception as e:
        review_errors.append(f"REVIEW {n}번째 {entry!r}: {e}")

# 4) 검토 반영 후 합계 다시 검증 → 어긋나면 상위 칸을 확인 필요로
checks = sum_checks(autofix=False)
for c, a in zip(checks, auto_checks): c["fixed"] = a["fixed"]
for c in checks:
    if not c["ok"]:
        flag(c["parent"], c["k"], f"합계 불일치: 상위 {fmt(c['want'])}, 하위 합 {fmt(c['got'])}")

# 5) 항목 목록 (확인 필요 → 확인함/수정함 순)
items = []
for key in sorted(set(flags) | set(forced_open) | set(resolved)):
    r, k = recs[key[0]], key[1]
    reasons = list(flags.get(key, []))
    if key in forced_open: reasons.append(f"검토자 표시: {forced_open[key]}")
    if not reasons: reasons = ["검토자 " + resolved[key]["status"]]
    rv = resolved.get(key)
    status = "확인 필요" if key in forced_open or not rv else rv["status"]
    items.append({"item_id": cid(*key), "row_id": key[0], "row": r["label"], "column": headers[k],
                  "value": show(r, k), "reasons": "; ".join(reasons), "status": status,
                  "memo": rv["memo"] if rv and status != "확인 필요" else "",
                  "date": rv["date"] if rv and status != "확인 필요" else ""})
ORDER = {"확인 필요": 0, "수정함": 1, "확인함": 2}
items.sort(key=lambda it: (ORDER[it["status"]], it["row_id"], it["item_id"]))
open_cells = {(it["row_id"], headers.index(it["column"])) for it in items if it["status"] == "확인 필요"}
edited_cells = {(ch["row_id"], headers.index(ch["column"])) for ch in changes if ch["kind"] == "검토"}
count = {s: sum(it["status"] == s for it in items) for s in ORDER}

def note_of(r):
    return "; ".join(f"{headers[k]}: {notes.get(m, '각주 ' + m)}" for k, m in r["marker"].items() if m)

def write_csv(path, cols, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

# ── 파일 쓰기 ─────────────────────────────────────────────────
with open(T1_OUT / "table.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["row_id", "level", "parent_id", *headers, "note", "check"])
    for r in recs:
        chk = ", ".join(headers[k] for k in range(N_COLS) if (r["row_id"], k) in open_cells)
        w.writerow([r["row_id"], r["level"], "" if r["parent_id"] is None else r["parent_id"], r["label"],
                    *("" if r["value"][k] is None else r["value"][k] for k in range(1, N_COLS)), note_of(r),
                    f"확인 필요: {chk}" if chk else ""])
write_csv(T1_OUT / "review_items.csv", ["item_id", "row_id", "row", "column", "value", "reasons", "status", "memo", "date"], items)
write_csv(T1_OUT / "changes.csv", ["date", "item_id", "row_id", "row", "column", "before", "after", "kind", "reason", "by"], changes)

def esc(t): return str(t).replace("|", "\\|")
SUPCHR = {v: k for k, v in SUP.items()}
L = [f"# {title}", "",
     f"- 원본 이미지: `outputs/ocr/input/{T1_NAME}.png`",
     f"- OCR: PaddleOCR (`lang=\"en\"`), 원시 결과 `outputs/ocr/raw/{T1_NAME}_ocr_raw.json`",
     f"- 열 경계: {bounds_how} (x = {', '.join(f'{b:.0f}' for b in bounds)})",
     "- 출처·모델·패키지 버전·실행 환경: `outputs/manifest.json`",
     f"- 검토: {REVIEWER or '(검토자 미기재)'} · 기준일 {TODAY} · "
     f"**확인 필요 {count['확인 필요']}건** · 수정함 {count['수정함']}건 · 확인함 {count['확인함']}건", "",
     "## 표", "", "⚠️ = 아직 원본 대조가 필요한 칸 (아래 '확인 필요') · ✏️ = 검토자가 원본을 보고 고친 칸 (아래 '수정 내역')", "",
     "| " + " | ".join(headers) + " |", "|---|" + "---:|" * (N_COLS - 1)]
for r in recs:
    ind = "&nbsp;" * 4 * max(r["level"] - 1, 0)
    cells_ = []
    for k in range(N_COLS):
        t = show(r, k)
        if k > 0 and r["marker"][k]: t += f" {SUPCHR.get(r['marker'][k], r['marker'][k])}"
        if r["level"] == 0 and t: t = f"**{t}**"
        if (r["row_id"], k) in edited_cells: t += " ✏️"
        if (r["row_id"], k) in open_cells: t += " ⚠️"
        cells_.append((ind if k == 0 else "") + esc(t))
    L.append("| " + " | ".join(cells_) + " |")
if notes:
    L += [""] + [f"{SUPCHR.get(k, k)} {v}  " for k, v in notes.items()]

L += ["", "## 합계 검증", "", "검토 수정을 반영한 뒤 다시 계산한 결과입니다.", "",
      "| 상위 행 | 열 | 상위 값 | 하위 합 | 결과 |", "|---|---|---:|---:|---|"]
for c in checks:
    t = "✅ 일치" if c["ok"] else "❌ 불일치"
    if c["fixed"]: t += f" (각주 번호 자동 분리: {', '.join(c['fixed'])})"
    L.append(f"| {esc(c['parent']['label'])} | {esc(c['col'])} | {fmt(c['want'])} | {fmt(c['got'])} | {t} |")

L += ["", "## 확인 필요", "",
      "원본과 대조해야 할 칸입니다. 대조한 결과는 노트북 6-1 셀의 `REVIEW`에 적고 다시 실행하면 상태와 수정 내역이 바뀝니다.", ""]
if items:
    L += ["| 상태 | 항목 | 행 | 열 | 현재 값 | 이유 | 검토 메모 | 날짜 |", "|---|---|---|---|---:|---|---|---|"]
    ICON = {"확인 필요": "⚠️ 확인 필요", "수정함": "✏️ 수정함", "확인함": "✅ 확인함"}
    for it in items:
        L.append(f"| {ICON[it['status']]} | {it['item_id']} | {esc(it['row'])} | {esc(it['column'])} | {esc(it['value']) or '(빈 칸)'} | "
                 f"{esc(it['reasons'])} | {esc(it['memo'])} | {it['date']} |")
else:
    L.append("확인이 필요한 항목이 없습니다.")

L += ["", "## 수정 내역", "",
      "OCR이 읽은 값에서 바뀐 칸을 모두 적습니다. `자동`은 노트북이 합계 검증으로 고친 것, `검토`는 검토자가 원본을 보고 고친 것입니다.", ""]
if changes:
    L += ["| 날짜 | 항목 | 행 | 열 | 이전 값 | 새 값 | 구분 | 근거 | 수정자 |", "|---|---|---|---|---:|---:|---|---|---|"]
    for ch in changes:
        L.append(f"| {ch['date']} | {ch['item_id']} | {esc(ch['row'])} | {esc(ch['column'])} | {esc(ch['before']) or '(빈 칸)'} | "
                 f"{esc(ch['after']) or '(빈 칸)'} | {ch['kind']} | {esc(ch['reason'])} | {esc(ch['by'])} |")
else:
    L.append("수정한 칸이 없습니다.")

L += ["", "## 행·열 연결 (OCR 출처)", "",
      "칸마다 값을 만든 OCR 상자 번호입니다. 번호는 `outputs/ocr/text/` CSV의 `id`, `outputs/ocr/vis/` 이미지의 빨간 번호와 같습니다.", "",
      "| row_id | 단계 | 상위 | " + " | ".join(map(esc, headers)) + " |", "|---:|---:|---:|" + "---|" * N_COLS]
for r in recs:
    def src(k):
        if not r["ids"][k]: return "—"
        s = r["score"][k]
        return ", ".join(map(str, r["ids"][k])) + f" ({s:.2f}{' ⚠️' if s < CONF_MIN else ''})"
    L.append(f"| {r['row_id']} | {r['level']} | {'' if r['parent_id'] is None else r['parent_id']} | "
             + " | ".join(src(k) for k in range(N_COLS)) + " |")
L += ["", f"⚠️ = 신뢰도 {CONF_MIN} 미만."]
(T1_OUT / "document.md").write_text("\n".join(L) + "\n", encoding="utf-8")

# ── 화면 출력 ─────────────────────────────────────────────────
print(f"열 경계: {bounds_how} {[round(b) for b in bounds]} · 머리글: {headers}")
for c in checks:
    print(("✅" if c["ok"] else "❌"), "합계", c["parent"]["label"], "/", c["col"],
          fmt(c["want"]), "vs", fmt(c["got"]), f"← 각주 자동 분리 {c['fixed']}" if c["fixed"] else "")
if review_errors:
    print("\n⛔ REVIEW에서 반영하지 못한 항목:"); print("\n".join("   " + e for e in review_errors))
print(f"\n확인 필요 {count['확인 필요']}건 · 수정함 {count['수정함']}건 · 확인함 {count['확인함']}건 · 수정 내역 {len(changes)}건")
todo = [it for it in items if it["status"] == "확인 필요"]
if todo:
    print("\n원본과 대조한 뒤 6-1 셀 REVIEW에 붙여 넣을 줄 (조치·값·메모를 채우세요):")
    for it in todo:
        k = headers.index(it["column"])
        print(f'    ({it["row_id"]}, {k}, "확인", None, ""),   # {it["row"]} | {it["column"]} = {it["value"] or "(빈 칸)"} — {it["reasons"]}')
print(f"\n저장: {', '.join(str(T1_OUT / n) for n in ('table.csv', 'review_items.csv', 'changes.csv', 'document.md'))}")
""")

md("""
## 7. 정답과 비교 (Table 1)
사람이 옮기고 합계까지 검산한 값과, 6단계에서 구조화한 표를 행마다 비교합니다.
""")
code(r"""
TRUTH = [
    ("United States, with outlying possessions",        3743306, 101115487),
    ("United States, exclusive of outlying possessions", 3026789, 91972266),
    ("Outlying possessions",                             716517,  9143221),
    ("Alaska",                                           590884,  64356),
    ("Hawaii",                                           6449,    191909),
    ("Porto Rico",                                       3435,    1118012),
    ("Military and naval service abroad",                None,    55608),
    ("Philippine Islands",                               115026,  7635426),
    ("Panama Canal Zone",                                436,     62810),
    ("Guam (population estimated)",                      210,     9000),
    ("Samoa (population estimated)",                     77,      6100),
]
from difflib import SequenceMatcher

ok = 0
for label, area, pop in TRUTH:
    r = max(recs, key=lambda r: SequenceMatcher(None, r["label"].lower(), label.lower()).ratio())
    got = (r["value"][1], r["value"][2])
    hit = got == (area, pop)
    ok += hit
    print(("✅" if hit else "❌"), f"{label:<50} 정답 {fmt(area):>10} {fmt(pop):>12}   OCR {fmt(got[0]):>10} {fmt(got[1]):>12}")
T1_TRUTH_MATCH = {"matched": int(ok), "total": len(TRUTH), "rows_found": len(recs)}
print(f"\n{ok}/{len(TRUTH)} 행 일치 · OCR 행 수 {len(recs)} (정답 {len(TRUTH)})")
""")

md("""
## 8. 실행 기록 → `outputs/manifest.json`
이번 실행을 나중에 똑같이 되풀이하거나 검증할 수 있도록 한 파일에 모아 적습니다.

| 항목 | 내용 |
|---|---|
| `sources` | 2-1 셀에 적은 출처·원문 페이지. 비워 둔 칸은 `missing`에 이름이 남습니다 |
| `inputs` | 입력 이미지마다 경로·크기·SHA-256·가로세로 픽셀, 사본(`outputs/ocr/input/`)과 해시가 같은지 |
| `models` | 실제로 쓴 검출·인식 모델 이름, 내려받은 모델 폴더의 파일 크기·SHA-256 |
| `packages` | Python과 PaddlePaddle·PaddleOCR·PaddleX 등 주요 패키지 버전, 설치 명령 |
| `execution` | 실행 위치(Colab/로컬), 시작·종료 시각(UTC), OS, CPU·GPU, Paddle 장치 |
| `settings` | OCR 설정, 행 묶기 기준, Table 1 열 경계·머리글, 확인 필요 기준, 검토자·검토 건수 |
| `results` | 이미지별 OCR 상자 수·평균 신뢰도, Table 1 합계 검증·검토 현황·정답 일치 수 |
| `outputs` | `outputs/` 안에 만든 파일마다 크기와 SHA-256 |
""")
code(r"""
import hashlib, os, platform, subprocess, sys
from datetime import datetime, timezone
from importlib import metadata

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def file_info(p):
    p = Path(p)
    info = {"path": p.as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)}
    if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        with PILImage.open(p) as im:
            info.update(width=im.width, height=im.height, format=im.format, mode=im.mode)
    return info

def dist_version(name):
    try: return metadata.version(name)
    except metadata.PackageNotFoundError: return None

def model_names(obj, depth=0, found=None):
    # PaddleX 파이프라인 설정에서 {"모듈": {"model_name": ...}} 를 찾아 모은다
    found = {} if found is None else found
    if depth <= 8 and isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict) and isinstance(v.get("model_name"), str): found[str(k)] = v["model_name"]
            model_names(v, depth + 1, found)
    return found

now = datetime.now(timezone.utc).isoformat(timespec="seconds")

# ── 출처 ──
sources = {}
for name, src in SOURCES.items():
    sources[name] = {**src, "missing": [k for k, v in src.items() if v in (None, "") and k != "notes"]}

# ── 입력 파일 ──
inputs = []
for img in sorted(IN.glob("*.png")):
    copy = DIRS["input"] / img.name
    info = file_info(img)
    info.update(name=img.name, origin="노트북 안에 base64로 포함 (2단계에서 파일로 풀어 씀)",
                source=img.name if img.name in SOURCES else None,
                copy=copy.as_posix() if copy.exists() else None,
                copy_matches=sha256(copy) == info["sha256"] if copy.exists() else None,
                used_for=["OCR"] + (["Table 1 구조화"] if img.stem == T1_NAME else []))
    inputs.append(info)

# ── 모델 ──
used = {}
for attr in ("_merged_paddlex_config", "_paddlex_config", "paddlex_config"):
    cfg = getattr(ocr, attr, None)
    try: cfg = cfg() if callable(cfg) else cfg
    except Exception: cfg = None
    used = model_names(cfg) if isinstance(cfg, dict) else {}
    if used: break
models_dir = Path.home() / ".paddlex" / "official_models"
downloaded = []
for d in sorted(p for p in models_dir.glob("*") if p.is_dir()):
    downloaded.append({"name": d.name, "path": d.as_posix(),
                       "used": (d.name in used.values()) if used else None,
                       "files": [{**file_info(f), "path": f.relative_to(d).as_posix()} for f in sorted(d.rglob("*")) if f.is_file()]})
models = {"lang": OCR_KWARGS.get("lang"),
          "used_by_pipeline": used or None,
          "note": None if used else "파이프라인 설정에서 모델 이름을 읽지 못함 — downloaded 목록 참고",
          "cache_dir": models_dir.as_posix(), "downloaded": downloaded}

# ── 패키지 ──
packages = {"python": sys.version.split()[0], "install_command": f"pip install -q {PIP_PACKAGES}"}
for n in ("paddlepaddle", "paddlepaddle-gpu", "paddleocr", "paddlex", "numpy", "pillow",
          "opencv-python", "opencv-python-headless", "opencv-contrib-python"):
    v = dist_version(n)
    if v: packages[n] = v

# ── 실행 위치 ──
def gpu_info():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return out.splitlines() or None
    except Exception:
        return None
try:
    import paddle
    paddle_device, cuda_built = paddle.device.get_device(), paddle.device.is_compiled_with_cuda()
except Exception:
    paddle_device, cuda_built = None, None
in_colab = "google.colab" in sys.modules
execution = {"location": "Google Colab" if in_colab else "local",
             "colab_release": os.environ.get("COLAB_RELEASE_TAG") if in_colab else None,
             "started_at_utc": globals().get("RUN_STARTED"), "finished_at_utc": now,
             "os": platform.platform(), "machine": platform.machine(), "cpu_count": os.cpu_count(),
             "gpu": gpu_info(), "paddle_device": paddle_device, "paddle_cuda_built": cuda_built,
             "working_dir": os.getcwd()}

# ── 설정 ──
settings = {"ocr": OCR_KWARGS,
            "row_grouping": {"rule": "상자 세로 중심 간격 > 중앙 상자 높이 × gap_ratio 이면 새 행", "gap_ratio": ROW_GAP},
            "table1": {"image": f"{T1_NAME}.png", "title": title, "n_cols": N_COLS, "headers": headers,
                       "column_bounds_px": [round(float(b), 1) for b in bounds], "column_bounds_method": bounds_how,
                       "body_start_rule": "천 단위 쉼표 숫자가 처음 나오는 행", "footnote_autofix": "하위 합계가 상위 값과 맞을 때만"},
            "review": {"conf_min": CONF_MIN, "reviewer": REVIEWER or None, "review_date": TODAY,
                       "entries": len(REVIEW), "errors": review_errors}}

# ── 결과 요약 ──
results_summary = {"ocr": {name: {"text_boxes": len(r["rec_texts"]),
                                  "mean_score": round(float(np.mean(r["rec_scores"])), 4) if len(r["rec_scores"]) else None,
                                  "min_score": round(float(np.min(r["rec_scores"])), 4) if len(r["rec_scores"]) else None}
                           for name, r in results.items()},
                   "table1": {"rows": len(recs), "sum_checks_ok": sum(c["ok"] for c in checks), "sum_checks": len(checks),
                              "review": count, "changes": len(changes),
                              "truth_match": globals().get("T1_TRUTH_MATCH")}}

MANIFEST = Path("outputs/manifest.json")
outputs = [file_info(f) for f in sorted(Path("outputs").rglob("*")) if f.is_file() and f != MANIFEST]

manifest = {"manifest_version": 1, "created_at_utc": now, "notebook": "paddleocr_table.ipynb",
            "sources": sources, "inputs": inputs, "models": models, "packages": packages,
            "execution": execution, "settings": settings, "results": results_summary, "outputs": outputs}
MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

print(f"저장: {MANIFEST} ({MANIFEST.stat().st_size:,} bytes)\n")
print(f"실행 위치  {execution['location']} · {execution['paddle_device']} · {execution['started_at_utc']} → {now}")
print(f"패키지     " + ", ".join(f"{k} {v}" for k, v in packages.items() if k != "install_command"))
print(f"모델       {used or [d['name'] for d in downloaded]}")
print(f"입력       " + ", ".join(f"{i['name']} {i['width']}×{i['height']} sha256 {i['sha256'][:12]}…" for i in inputs))
print(f"출력 파일  {len(outputs)}개")
for name, s in sources.items():
    if s["missing"]: print(f"⚠️ {name} 출처 미기재: {', '.join(s['missing'])} — 2-1 셀을 채우고 다시 실행하세요")
""")

md("## 9. 결과 내려받기\n`outputs/` 폴더 전체(`ocr/`, `table1/`, `manifest.json`)를 zip으로 내려받습니다.")
code("""
import shutil
from google.colab import files
shutil.make_archive("ocr_outputs", "zip", root_dir=".", base_dir="outputs")
files.download("ocr_outputs.zip")
""")

nb = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}
out = proj / "paddleocr_table.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(out, out.stat().st_size, "bytes")
