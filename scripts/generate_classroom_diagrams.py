"""Hand-drawn schematic figures for the classroom report.

Each figure is saved as both SVG (slide reuse) and PNG (so the existing
docx embeds keep working without modification).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

plt.rcParams["font.family"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11

C_PCAP   = "#3F6FBF"
C_PKT    = "#A7C7E7"
C_SESS_A = "#7AB87F"
C_SESS_B = "#E0A45A"
C_SESS_C = "#C36B7A"
C_IMG    = "#5B7DB0"
C_STAT   = "#E1A14E"
C_MODEL  = "#7A4FA0"
C_GOOD   = "#3E8E47"
C_BAD    = "#C0392B"
C_BG     = "#F8F9FB"
C_TRAIN  = "#A8D8B9"
C_TEST   = "#F5C99A"
C_LEAK   = "#D9534F"
C_LINE   = "#2C2C2C"

FIG_DIR = Path(__file__).resolve().parent.parent / "docs" / "figures"
SVG_DIR = FIG_DIR / "svg"
FIG_DIR.mkdir(parents=True, exist_ok=True)
SVG_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig, name: str) -> None:
    fig.savefig(SVG_DIR / f"{name}.svg", format="svg", bbox_inches="tight",
                facecolor="white")
    fig.savefig(FIG_DIR / f"{name}.png", format="png", dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {name}.svg + {name}.png")


def _box(ax, xy, w, h, text, *, face=C_BG, edge=C_LINE,
         text_size=10.5, text_color="black", lw=1.2,
         radius=0.08, bold=False, va="center", ha="center"):
    x, y = xy
    bbox = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.005,rounding_size={radius}",
        linewidth=lw, edgecolor=edge, facecolor=face, zorder=2,
    )
    ax.add_patch(bbox)
    weight = "bold" if bold else "normal"
    if ha == "left":
        tx, txa = x + 0.08, "left"
    elif ha == "right":
        tx, txa = x + w - 0.08, "right"
    else:
        tx, txa = x + w / 2, "center"
    if va == "top":
        ty, tva = y + h - 0.08, "top"
    elif va == "bottom":
        ty, tva = y + 0.08, "bottom"
    else:
        ty, tva = y + h / 2, "center"
    ax.text(tx, ty, text, ha=txa, va=tva, fontsize=text_size,
            color=text_color, weight=weight, zorder=3)
    return (x, y, x + w, y + h)


def _arrow(ax, p1, p2, *, color=C_LINE, lw=1.6, style="-|>", mut=14,
           connectionstyle="arc3", zorder=2.5):
    a = FancyArrowPatch(
        p1, p2, arrowstyle=style, mutation_scale=mut,
        linewidth=lw, color=color,
        connectionstyle=connectionstyle, zorder=zorder,
    )
    ax.add_patch(a)


def _title(ax, txt, *, size=14):
    ax.set_title(txt, fontsize=size, weight="bold", color="#1F3A68", pad=12)


def _setup(width, height):
    """Axes whose data coordinates equal inches (xlim=(0,W), ylim=(0,H))."""
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


# ---------------------------------------------------------------------------
# Figure 1: pcap / packet / session 三层关系
# ---------------------------------------------------------------------------

def fig_sample_definition():
    W, H = 14.0, 8.5
    fig, ax = _setup(W, H)
    _title(ax, "图 · pcap、packet、session 三层单位的真实例子", size=15)

    # ---- Layer 1: pcap file ----
    _box(ax, (0.5, H - 1.4), W - 1.0, 0.95,
         "p2p_multipleSpeed2-1.pcap   (~2.4 GB,  ~1600 万 packet,  label = p2p-nontor)",
         face=C_PCAP, edge=C_PCAP, text_color="white", bold=True,
         text_size=12.5)
    _arrow(ax, (W / 2, H - 1.45), (W / 2, H - 2.1), lw=1.8)

    # ---- Layer 2: packets ----
    pkt_y = H - 2.9
    pkt_w = 0.70
    pkt_h = 0.85
    start_x = 0.6
    gap = 0.10
    pkts_to_draw = 14
    for i in range(pkts_to_draw):
        x = start_x + i * (pkt_w + gap)
        _box(ax, (x, pkt_y), pkt_w, pkt_h, f"pkt\n#{i + 1}",
             face=C_PKT, edge="#5B7DB0", text_size=8, radius=0.04)
    ellipsis_x = start_x + pkts_to_draw * (pkt_w + gap)
    ax.text(ellipsis_x + 0.1, pkt_y + pkt_h / 2,
            "…   (共 ~1600 万 packet)", fontsize=10.5, va="center",
            color="#444")

    ax.text(0.5, pkt_y - 0.35,
            "每个 packet 含：时间戳 / 五元组 (src_ip,sport,dst_ip,dport,proto) "
            "/ TCP flags / payload / 长度",
            fontsize=10, style="italic", color="#555")

    # arrow + label
    _arrow(ax, (W / 2, pkt_y - 0.65), (W / 2, pkt_y - 1.55), lw=2.0)
    ax.text(W / 2 + 0.2, pkt_y - 1.1,
            "按 5-元组聚合（双向无序）",
            fontsize=10.5, weight="bold", color=C_LINE, va="center")

    # ---- Layer 3: sessions ----
    sess_y = pkt_y - 2.85
    sess_h = 1.4
    sessions = [
        ("S1", "10.152.152.11:55357\n↔ 103.52.253.34:27965\nproto = TCP (6)",
         "247 packets", C_SESS_A),
        ("S2", "10.152.152.11:37070\n↔ 190.239.206.162:8999\nproto = TCP (6)",
         "1832 packets", C_SESS_B),
        ("S3", "10.152.152.11:9100\n↔ 182.185.113.123:57558\nproto = UDP (17)",
         "55 packets", C_SESS_C),
    ]
    box_w = 3.9
    gap_s = 0.5
    total = len(sessions) * box_w + (len(sessions) - 1) * gap_s
    start = (W - total) / 2
    for i, (sid, five, count, color) in enumerate(sessions):
        x = start + i * (box_w + gap_s)
        _box(ax, (x, sess_y), box_w, sess_h, "",
             face=color, edge=color, lw=1.5, radius=0.08)
        ax.text(x + 0.20, sess_y + sess_h - 0.25, sid,
                fontsize=13, weight="bold", color="white")
        ax.text(x + box_w / 2, sess_y + sess_h / 2,
                five, ha="center", va="center", fontsize=10,
                color="white")
        ax.text(x + box_w - 0.20, sess_y + 0.18, count,
                ha="right", va="bottom", fontsize=9.5,
                color="white", style="italic")

    ax.text(W / 2, sess_y - 0.45,
            "一个 pcap → 3,265 条 session（≥ 3 包）→ 3,265 个训练样本",
            ha="center", fontsize=12, weight="bold", color="#1F3A68")

    # legend at bottom
    ax.text(0.5, 0.35,
            "★ 关键认识：模型的一个样本 = 一条 session，不是一个 pcap。"
            "整个项目 154 pcap → 54,460 session（Algo 1）→ 24,995 session（Algo 2 阈值 ≥10 包）",
            fontsize=10.5, color="#444", style="italic")

    _save(fig, "classroom_sample_definition")


# ---------------------------------------------------------------------------
# Figure 2: Algorithm 1 + Algorithm 2 + ANDE
# ---------------------------------------------------------------------------

def fig_algorithm_ande_overview():
    W, H = 16.0, 10.0
    fig, ax = _setup(W, H)
    _title(ax, "图 · 一条 session 如何被 Algorithm 1 + Algorithm 2 + ANDE 处理",
           size=15)

    # ---- input session ----
    inp_x, inp_y, inp_w, inp_h = 0.4, H - 3.0, 1.9, 1.7
    _box(ax, (inp_x, inp_y), inp_w, inp_h,
         "一条 session\n例：S1\n247 packets",
         face=C_SESS_A, edge=C_SESS_A, text_color="white",
         bold=True, text_size=11)

    # arrow to upper branch
    upper_y = H - 1.4
    lower_y = H - 4.0
    _arrow(ax, (inp_x + inp_w, inp_y + inp_h - 0.3),
           (inp_x + inp_w + 0.6, upper_y - 0.05), lw=1.8)
    _arrow(ax, (inp_x + inp_w, inp_y + 0.3),
           (inp_x + inp_w + 0.6, lower_y + 1.05), lw=1.8)

    # ---- Algorithm 1 branch (top row) ----
    ax.text(inp_x + inp_w + 0.7, upper_y + 1.0,
            "Algorithm 1   raw-byte image",
            fontsize=12, weight="bold", color="#1F3A68")
    steps_a1 = [
        "① 按时间序\n取每包字节",
        "② 拼接成\n长字节流",
        "③ 截断 / 零补\n到 8100 byte",
        "④ reshape\n90 × 90 灰度图",
    ]
    sx_a1 = inp_x + inp_w + 0.7
    step_w, step_h = 1.55, 1.05
    for i, s in enumerate(steps_a1):
        x = sx_a1 + i * (step_w + 0.15)
        _box(ax, (x, upper_y - 0.2), step_w, step_h, s,
             face="#E6F0FB", edge=C_IMG, text_size=9,
             text_color="#1F3A68")
        if i > 0:
            _arrow(ax, (x - 0.18, upper_y + step_h / 2 - 0.2),
                   (x, upper_y + step_h / 2 - 0.2), lw=1.2, mut=10)

    # final image icon
    img_x = sx_a1 + 4 * (step_w + 0.15)
    img_w, img_h = 1.15, 1.05
    _box(ax, (img_x, upper_y - 0.2), img_w, img_h, "",
         face="#2C3E50", edge="#2C3E50", radius=0.03)
    for r in range(4):
        for c in range(4):
            ax.add_patch(patches.Rectangle(
                (img_x + 0.10 + c * 0.235,
                 upper_y - 0.10 + r * 0.21),
                0.21, 0.20,
                facecolor=plt.cm.gray((r * 4 + c) / 16),
                edgecolor="none", zorder=3))
    ax.text(img_x + img_w / 2, upper_y - 0.40,
            "(1, 90, 90)",
            ha="center", fontsize=9, style="italic", color="#444")

    # ---- Algorithm 2 branch (middle row) ----
    ax.text(inp_x + inp_w + 0.7, lower_y + 1.4,
            "Algorithm 2   per-session 26 维统计特征",
            fontsize=12, weight="bold", color="#1F3A68")
    feat_groups = [
        ("TCP flag\n占比", "6 维"),
        ("协议\n占比", "4 维"),
        ("时间\n统计", "5 维"),
        ("包长\n统计", "4 维"),
        ("payload\n统计", "5 维"),
        ("DNS/TCP\n+ 总包数", "2 维"),
    ]
    feat_w = 1.05
    for i, (name, dim) in enumerate(feat_groups):
        x = inp_x + inp_w + 0.7 + i * (feat_w + 0.12)
        _box(ax, (x, lower_y - 0.05), feat_w, 1.15,
             f"{name}\n{dim}",
             face="#FCEFD3", edge=C_STAT, text_size=8.5, radius=0.05,
             text_color="#7A4F00")

    feat_end = inp_x + inp_w + 0.7 + 6 * (feat_w + 0.12)
    _box(ax, (feat_end, lower_y - 0.05), 1.7, 1.15,
         "26 维向量\n经 z-score\n标准化",
         face=C_STAT, edge=C_STAT, text_color="white", bold=True,
         text_size=10)
    _arrow(ax, (feat_end - 0.1, lower_y + 0.5),
           (feat_end, lower_y + 0.5), lw=1.3, mut=10)

    # ---- ANDE Model (bottom row, full width) ----
    model_y = H - 6.7
    model_h = 1.7

    # SE-ResNet
    se_x = 2.0
    _box(ax, (se_x, model_y), 3.5, model_h,
         "SE-ResNet-18  (image backbone)\n\n"
         "Conv7×7 → 32ch  → 32→64→128→256\n"
         "每个 BasicBlock 内插 SEBlock\n\n"
         "输出 256 维 image feature",
         face=C_MODEL, edge=C_MODEL, text_color="white", bold=False,
         text_size=9.5)

    # StatMLP
    sm_x = 6.3
    _box(ax, (sm_x, model_y), 3.0, model_h,
         "StatMLP  (stat backbone)\n\n"
         "Linear 26 → 18 → 9\n"
         "ReLU 激活\n\n"
         "输出 9 维 stat feature",
         face="#8B5FBF", edge="#8B5FBF", text_color="white",
         text_size=9.5)

    # Concat
    cc_x = 10.0
    _box(ax, (cc_x, model_y + 0.3), 1.6, model_h - 0.6,
         "concat\n\n256 + 9\n= 265 维",
         face="#345D8C", edge="#345D8C", text_color="white", bold=True,
         text_size=10)

    # FusionHead
    fh_x = 12.3
    _box(ax, (fh_x, model_y), 3.0, model_h,
         "FusionHead\n\n"
         "Linear 265 → 100\n→ 30\n→ 14 类\n\n"
         "softmax 输出",
         face="#1F3A68", edge="#1F3A68", text_color="white", bold=True,
         text_size=9.5)

    # arrows: image → SE-ResNet, stats → StatMLP, both → Concat → FusionHead
    _arrow(ax, (img_x + img_w / 2, upper_y - 0.25),
           (se_x + 1.75, model_y + model_h),
           lw=1.8, mut=14, connectionstyle="arc3,rad=-0.1")
    _arrow(ax, (feat_end + 0.85, lower_y - 0.1),
           (sm_x + 1.5, model_y + model_h),
           lw=1.8, mut=14, connectionstyle="arc3,rad=0.1")
    _arrow(ax, (se_x + 3.5, model_y + model_h / 2),
           (cc_x, model_y + model_h / 2 + 0.2), lw=1.6)
    _arrow(ax, (sm_x + 3.0, model_y + model_h / 2),
           (cc_x, model_y + model_h / 2 - 0.2), lw=1.6)
    _arrow(ax, (cc_x + 1.6, model_y + model_h / 2),
           (fh_x, model_y + model_h / 2), lw=1.8)

    # final output label
    _arrow(ax, (fh_x + 1.5, model_y - 0.1),
           (fh_x + 1.5, model_y - 0.7),
           lw=2.0, mut=16)
    ax.text(fh_x + 1.5, model_y - 1.05,
            "→  p2p-nontor",
            ha="center", fontsize=13, weight="bold", color=C_GOOD)

    # footer
    ax.text(0.5, 0.7,
            "ANDE 双分支总参数约 285 万。灰度图捕捉「字节级模式」，"
            "26 维特征补充「流量级语义」。",
            fontsize=10.5, color="#444", style="italic")
    ax.text(0.5, 0.25,
            "★ 关键约束：image 和 stats 都 **按 session 算**，"
            "这一点正是 Round 1/2/3 的核心争议。",
            fontsize=10.5, color=C_BAD, weight="bold")

    _save(fig, "classroom_algorithm_ande_overview")


# ---------------------------------------------------------------------------
# Figure 3: Round 1 leakage
# ---------------------------------------------------------------------------

def fig_round1_leakage():
    W, H = 15.0, 10.0
    fig, ax = _setup(W, H)
    _title(ax,
           "图 · Round 1 数据泄漏：同一 pcap 的所有 session 共用 stats，被随机切到 train / test",
           size=14)

    # ---- top row: pcap → 1 stats vector ----
    _box(ax, (0.5, H - 1.7), 4.6, 1.0,
         "p2p_multipleSpeed2-1.pcap\n(整个 pcap 文件)",
         face=C_PCAP, edge=C_PCAP, text_color="white", bold=True,
         text_size=11)
    _arrow(ax, (5.2, H - 1.2), (6.5, H - 1.2), lw=2.0, mut=14)
    ax.text(5.85, H - 0.85,
            "Algorithm 2\n（per-pcap，错误）", ha="center",
            fontsize=10, weight="bold", color="#444")
    _box(ax, (6.6, H - 1.7), 5.5, 1.0,
         "stats_p2p   （唯一一份 26 维向量）",
         face=C_STAT, edge=C_BAD, text_color="white", bold=True,
         text_size=11.5, lw=2.5)
    ax.text(12.2, H - 1.2,
            "★ 错误：整个 pcap\n  只算一次",
            fontsize=10.5, color=C_BAD, weight="bold", va="center")

    # ---- middle: 5 sessions all linked to the same stats ----
    sess_y = H - 4.5
    sess_w, sess_h = 1.95, 1.35
    gap_s = 0.30
    base_x = 0.6
    session_examples = [
        ("S1", "55357↔27965", "image_S1"),
        ("S2", "37070↔8999",  "image_S2"),
        ("S3", "37069↔38553", "image_S3"),
        ("S4", "9100↔57558",  "image_S4"),
        ("S5", "...",         "image_S5"),
    ]
    centers = []
    for i, (sid, five, img) in enumerate(session_examples):
        x = base_x + i * (sess_w + gap_s)
        _box(ax, (x, sess_y), sess_w, sess_h,
             f"{sid}  ·  {five}\n──────────\n{img}\n+ stats_p2p",
             face=C_SESS_A, edge=C_SESS_A, text_color="white",
             text_size=9.5)
        centers.append(x + sess_w / 2)

    ax.text(base_x + 5 * (sess_w + gap_s) + 0.15,
            sess_y + sess_h / 2,
            "…  共 3,265 条\n     session",
            fontsize=10, va="center", style="italic", color="#555")

    # red lines from each session top to single stats box
    stats_anchor = (9.35, H - 1.7)
    for cx in centers:
        _arrow(ax, (cx, sess_y + sess_h + 0.02),
               stats_anchor,
               color=C_LEAK, lw=1.3, mut=10, style="->",
               connectionstyle="arc3,rad=-0.18")
    ax.text(stats_anchor[0] - 1.5, stats_anchor[1] - 0.4,
            "↑ 所有 session 都指向同一份 stats_p2p",
            fontsize=10.5, color=C_LEAK, weight="bold")

    # ---- bottom: 8:2 session split ----
    sp_y = sess_y - 2.3
    _box(ax, (0.5, sp_y + 1.1), W - 1.0, 0.85,
         "↓  随机 session 级 8:2 split  →  2,595 train  /  670 test",
         face="#FFF8E1", edge="#999", text_size=11)

    _box(ax, (0.5, sp_y - 0.6), 7.0, 1.5,
         "TRAIN 样本（举例）\n"
         "  S1 + image_S1 + stats_p2p + label = p2p-nontor\n"
         "  S2 + image_S2 + stats_p2p + label = p2p-nontor",
         face=C_TRAIN, edge=C_GOOD, text_size=10,
         ha="left")
    _box(ax, (W - 7.5, sp_y - 0.6), 7.0, 1.5,
         "TEST 样本（举例）\n"
         "  S3 + image_S3 + stats_p2p + label = p2p-nontor\n"
         "  S4 + image_S4 + stats_p2p + label = p2p-nontor",
         face=C_TEST, edge="#D69E48", text_size=10,
         ha="left")

    # callout
    _box(ax, (0.5, 0.2), W - 1.0, 1.1,
         "★ train 和 test 的 stats 完全一致 → DT / RF / XGB 只要按 stats 阈值就能秒分对\n"
         "Round 1 结果（8100B / 14 类）：DT = 0.9999,  RF = 1.0000,  XGB = 1.0000,  ANDE = 0.9908",
         face="#FFE3E3", edge=C_BAD, lw=2, text_color=C_BAD, bold=True,
         text_size=11)

    _save(fig, "classroom_round1_leakage")


# ---------------------------------------------------------------------------
# Figure 4: Round 2 over-fix
# ---------------------------------------------------------------------------

def fig_round2_concrete():
    W, H = 15.0, 9.0
    fig, ax = _setup(W, H)
    _title(ax,
           "图 · Round 2 过度修正：pcap 级 split 把任务变成「跨文件泛化」",
           size=14)

    # subtitle
    ax.text(0.5, H - 1.4,
            "为堵 Round 1 的泄漏，改用 pcap-level split——同一 pcap 的所有 session 整体进 train 或 test。",
            fontsize=10.5, color="#444", style="italic")
    ax.text(0.5, H - 1.85,
            "这堵住了 stats 泄漏，但把评测意外变成「跨 pcap 文件的泛化」。下面以 ft-nontor 类别为例：",
            fontsize=10.5, color="#444", style="italic")

    # ---- left: training pcap ----
    tr_x, tr_y, tr_w, tr_h = 0.5, H - 6.0, 6.5, 3.2
    _box(ax, (tr_x, tr_y), tr_w, tr_h, "",
         face="#E8F5E9", edge=C_GOOD, lw=2.5, radius=0.08)
    ax.text(tr_x + 0.3, tr_y + tr_h - 0.4, "TRAIN",
            fontsize=15, weight="bold", color=C_GOOD)
    ax.text(tr_x + 0.3, tr_y + tr_h - 0.95,
            "SFTP_filetransfer.pcap",
            fontsize=12, weight="bold", color="#1F3A68")
    ax.text(tr_x + 0.3, tr_y + tr_h - 1.35,
            "    session 数：仅 12 条",
            fontsize=11, color="#444")
    ax.text(tr_x + 0.3, tr_y + tr_h - 1.70,
            "    label：ft-nontor",
            fontsize=11, color="#444")
    # 12 mini boxes — placed in the lower half so they don't overlap text
    mini = 0.42
    gap_m = 0.12
    grid_w = 6 * mini + 5 * gap_m
    grid_x0 = tr_x + (tr_w - grid_w) / 2
    grid_y0 = tr_y + 0.4
    for i in range(12):
        row = i // 6
        col = i % 6
        ax.add_patch(patches.Rectangle(
            (grid_x0 + col * (mini + gap_m),
             grid_y0 + (1 - row) * (mini + gap_m)),
            mini, mini,
            facecolor=C_GOOD, edgecolor="white", lw=1.2, zorder=3))

    # ---- right: testing pcap ----
    te_x, te_y, te_w, te_h = W - 7.0, H - 6.0, 6.5, 3.2
    _box(ax, (te_x, te_y), te_w, te_h, "",
         face="#FFF3E0", edge="#D69E48", lw=2.5, radius=0.08)
    ax.text(te_x + 0.3, te_y + te_h - 0.4, "TEST",
            fontsize=15, weight="bold", color="#D69E48")
    ax.text(te_x + 0.3, te_y + te_h - 0.95,
            "FTP_filetransfer.pcap",
            fontsize=12, weight="bold", color="#1F3A68")
    ax.text(te_x + 0.3, te_y + te_h - 1.35,
            "    session 数：1,544 条",
            fontsize=11, color="#444")
    ax.text(te_x + 0.3, te_y + te_h - 1.70,
            "    label：ft-nontor",
            fontsize=11, color="#444")
    # 60 mini boxes
    grid_x0 = te_x + 0.3
    grid_y0 = te_y + 0.30
    for r in range(4):
        for c in range(16):
            ax.add_patch(patches.Rectangle(
                (grid_x0 + c * 0.20, grid_y0 + r * 0.20),
                0.15, 0.15,
                facecolor="#D69E48", edgecolor="white", lw=0.5, zorder=3))
    ax.text(grid_x0 + 16 * 0.20 + 0.10, grid_y0 + 2 * 0.20,
            "…", fontsize=16, color="#D69E48", weight="bold",
            va="center")

    # arrow between
    _arrow(ax, (tr_x + tr_w, H - 4.3),
           (te_x, H - 4.3), color=C_LINE, lw=3, mut=20)
    ax.text((tr_x + tr_w + te_x) / 2, H - 3.9,
            "学完 12 条\n→ 去识别 1544 条",
            ha="center", fontsize=10.5, weight="bold", color="#444")

    # explanation
    _box(ax, (0.5, 1.6), W - 1.0, 1.45,
         "★ 关键：SFTP（端口 22 / SSH 加密）和 FTP（端口 21 / 明文）在字节模式、包长分布上差异巨大。\n"
         "「只看过 12 条 SFTP 就要识别 1544 条 FTP」  ≠  论文「相同分布下未见过的 session 能不能分对」\n"
         "→ 任务被定义改了，accuracy 跌到 0.66 是任务变难，不是 ANDE 失效。",
         face="#FFF4DC", edge="#E1B45A", text_color="#7A5200",
         text_size=11, lw=1.5, ha="left")

    _box(ax, (0.5, 0.3), W - 1.0, 1.05,
         "Round 2 结果（8100B / 14 类）：ANDE = 0.6579    "
         "△  不能与论文 0.9820 直接对比",
         face="#FFE8E8", edge=C_BAD, text_color=C_BAD, bold=True,
         text_size=11.5, lw=2)

    _save(fig, "classroom_round2_concrete")


# ---------------------------------------------------------------------------
# Figure 5: Round 3 correct
# ---------------------------------------------------------------------------

def fig_round3_correct():
    W, H = 15.0, 10.0
    fig, ax = _setup(W, H)
    _title(ax,
           "图 · Round 3 修复：Algorithm 2 也按 session 算，每条 session 都有独立的 stats",
           size=14)

    # ---- top: pcap → 5 sessions ----
    _box(ax, (0.5, H - 1.6), 4.5, 1.0,
         "p2p_multipleSpeed2-1.pcap",
         face=C_PCAP, edge=C_PCAP, text_color="white", bold=True,
         text_size=11)
    _arrow(ax, (5.1, H - 1.1), (6.4, H - 1.1), lw=2.0, mut=16)
    ax.text(5.75, H - 0.75,
            "拆出 3,265 条 session\n每条独立算 stats", ha="center",
            fontsize=10, weight="bold", color="#444")

    # 5 sessions row (each gets own stats)
    sess_y = H - 4.7
    sess_w = 2.7
    head_h = 0.55
    body_h = 1.45
    badge_h = 0.45
    gap_s = 0.20
    sessions = [
        ("S1", "55357↔27965",
         "stats_S1\nDur = -0.02\nAvgLen = -0.34",
         C_SESS_A, "train"),
        ("S2", "37070↔8999",
         "stats_S2\nDur = -0.01\nAvgLen = -0.59",
         C_SESS_A, "train"),
        ("S3", "37069↔38553",
         "stats_S3\nDur = -0.01\nAvgLen = +1.15",
         C_SESS_B, "test"),
        ("S4", "9100↔57558",
         "stats_S4\nDur = +0.11\nAvgLen = -0.85",
         C_SESS_B, "test"),
        ("S5", "...",
         "stats_S5\n…  …",
         C_SESS_C, "train"),
    ]
    total = len(sessions) * sess_w + (len(sessions) - 1) * gap_s
    base_x = (W - total) / 2
    for i, (sid, five, stats, color, split) in enumerate(sessions):
        x = base_x + i * (sess_w + gap_s)
        # header
        _box(ax, (x, sess_y + body_h), sess_w, head_h,
             f"{sid}   ·   {five}",
             face=color, edge=color, text_color="white", bold=True,
             text_size=10)
        # body (each different!)
        _box(ax, (x, sess_y), sess_w, body_h,
             stats,
             face="white", edge=color, text_size=10,
             text_color="#333", lw=1.8)
        # split badge
        badge_face = C_TRAIN if split == "train" else C_TEST
        badge_edge = C_GOOD if split == "train" else "#D69E48"
        badge_color = C_GOOD if split == "train" else "#A87000"
        _box(ax, (x + (sess_w - 1.4) / 2, sess_y - badge_h - 0.15),
             1.4, badge_h, split.upper(),
             face=badge_face, edge=badge_edge,
             text_size=11, bold=True, text_color=badge_color, lw=1.5)

    # explanation
    _box(ax, (0.5, 1.95), W - 1.0, 1.7,
         "✓ 每条 session 都有自己独立的 stats（参考论文 Table II："
         "「in one session」、「within a packet window」）\n"
         "✓ 即使 S1（train）和 S3（test）来自同一 pcap，"
         "它们的 stats 完全不同 → 不再泄漏\n"
         "✓ 同时论文 50,905 个样本量级，只有按 session 算才能匹配（按 pcap 算只有 154）",
         face="#E8F5E9", edge=C_GOOD, text_color=C_GOOD, lw=1.5,
         text_size=10.5, ha="left")

    _box(ax, (0.5, 0.3), W - 1.0, 1.4,
         "Round 3 结果（8100B / 14 类）：ANDE = 0.9458   F1 = 0.9454   FPR = 0.0046\n"
         "所有方法 (DT / RF / XGB / CNN1D / ResNet-18 / ANDE) 都落在 0.92~0.95，"
         "符合论文 0.94~0.98 报告区间   →   ★ 最终干净复现",
         face="#D7EBDD", edge=C_GOOD, text_color=C_GOOD, bold=True,
         text_size=11.5, lw=2.5)

    _save(fig, "classroom_round3_correct")


# ---------------------------------------------------------------------------

def main():
    print("Drawing classroom schematic figures...")
    fig_sample_definition()
    fig_algorithm_ande_overview()
    fig_round1_leakage()
    fig_round2_concrete()
    fig_round3_correct()
    print(f"\nAll figures saved to:")
    print(f"  PNG (used by docx): {FIG_DIR}")
    print(f"  SVG (vector source): {SVG_DIR}")


if __name__ == "__main__":
    main()
