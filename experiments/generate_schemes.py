# experiments/generate_schemes.py
"""Generate protocol schemes for Streamlit UI."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_DIR = Path("docs/schemes")


def _setup_figure(title: str, size: tuple[float, float] = (12, 7)):
    fig, ax = plt.subplots(figsize=size, dpi=150)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=16, weight="bold", pad=14)
    return fig, ax


def _box(ax, x: float, y: float, w: float, h: float, text: str, color: str):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.5,
        edgecolor="#2f3e46",
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11, wrap=True)


def _arrow(ax, start: tuple[float, float], end: tuple[float, float], text: str = ""):
    arr = FancyArrowPatch(
        start, end, arrowstyle="->", mutation_scale=14, linewidth=1.8, color="#1d3557"
    )
    ax.add_patch(arr)
    if text:
        tx = (start[0] + end[0]) / 2
        ty = (start[1] + end[1]) / 2 + 0.03
        ax.text(tx, ty, text, ha="center", va="bottom", fontsize=10, color="#1d3557")


def generate_protocol_flow(path: Path) -> None:
    fig, ax = _setup_figure("Поток защищённого инференса")
    _box(
        ax,
        0.05,
        0.62,
        0.25,
        0.22,
        "Клиент\n1) Масштабирование\n2) Кодирование\n3) Шифрование",
        "#d8f3dc",
    )
    _box(ax, 0.38, 0.62, 0.25, 0.22, "API\nЗапрос:\npublic_key_n,\nencrypted_features", "#fff3bf")
    _box(ax, 0.70, 0.62, 0.25, 0.22, "Сервер\nГомоморфный\nлинейный score", "#d0ebff")

    _box(ax, 0.70, 0.24, 0.25, 0.22, "Ответ:\nencrypted_score", "#fff3bf")
    _box(
        ax,
        0.05,
        0.24,
        0.25,
        0.22,
        "Клиент\n4) Дешифрование\n5) Декодирование\n6) Постобработка",
        "#d8f3dc",
    )

    _arrow(ax, (0.30, 0.73), (0.38, 0.73), "HTTP POST")
    _arrow(ax, (0.63, 0.73), (0.70, 0.73), "Enc(x)")
    _arrow(ax, (0.70, 0.35), (0.30, 0.35), "Enc(z)")

    ax.text(
        0.5,
        0.1,
        "Сервер не видит исходные признаки и не имеет закрытого ключа.",
        ha="center",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def generate_threat_model(path: Path) -> None:
    fig, ax = _setup_figure("Модель угроз и границы доверия")
    _box(
        ax,
        0.05,
        0.58,
        0.26,
        0.26,
        "Клиент (доверенная зона)\n• исходные признаки\n• закрытый ключ",
        "#d8f3dc",
    )
    _box(ax, 0.37, 0.58, 0.26, 0.26, "Канал связи\nнаблюдаем\nвнешним нарушителем", "#ffe3e3")
    _box(ax, 0.69, 0.58, 0.26, 0.26, "Сервер (HBC)\nкорректен, но\nлюбопытен", "#d0ebff")

    _box(ax, 0.20, 0.20, 0.22, 0.20, "Передаётся:\nшифртексты\nи открытый ключ", "#fff3bf")
    _box(ax, 0.58, 0.20, 0.22, 0.20, "Не передаётся:\nзакрытый ключ\nи plaintext", "#ffd6a5")

    _arrow(ax, (0.31, 0.70), (0.37, 0.70), "Enc(x)")
    _arrow(ax, (0.63, 0.70), (0.69, 0.70), "Enc(x)")
    _arrow(ax, (0.80, 0.58), (0.42, 0.30), "Наблюдение")

    ax.text(
        0.5,
        0.07,
        "HBC = honest-but-curious: сервер следует протоколу, но пытается извлечь метаданные.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def generate_math_flow(path: Path) -> None:
    fig, ax = _setup_figure("Математический поток вычислений")
    _box(ax, 0.05, 0.64, 0.22, 0.18, "x\n(вектор признаков)", "#d8f3dc")
    _box(ax, 0.33, 0.64, 0.22, 0.18, "x' = scale(x)", "#d8f3dc")
    _box(ax, 0.61, 0.64, 0.30, 0.18, "x_int = round(x'·SCALE)\nEnc(x_int)", "#d8f3dc")

    _box(
        ax,
        0.61,
        0.34,
        0.30,
        0.18,
        "Enc(z_int)=Σ Enc(x_i)^w_i · Enc(b)\n(гомоморфная линейная часть)",
        "#d0ebff",
    )
    _box(ax, 0.33, 0.08, 0.22, 0.18, "z = Dec(Enc(z_int))/SCALE", "#fff3bf")
    _box(ax, 0.05, 0.08, 0.22, 0.18, "ŷ = sigmoid(z)\nили регрессия", "#fff3bf")

    _arrow(ax, (0.27, 0.73), (0.33, 0.73))
    _arrow(ax, (0.55, 0.73), (0.61, 0.73))
    _arrow(ax, (0.76, 0.64), (0.76, 0.52), "на сервер")
    _arrow(ax, (0.61, 0.43), (0.55, 0.17), "Enc(z)")
    _arrow(ax, (0.33, 0.17), (0.27, 0.17))

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def generate_comparison(path: Path) -> None:
    """Сравнение: plaintext vs encoded vs PHE (бар-чарт)."""
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    ax.set_title("Сравнение: plaintext vs encoded vs PHE", fontsize=16, weight="bold", pad=14)

    categories = ["Plaintext", "Encoded\nplaintext", "PHE"]
    privacy = [1, 2, 3]
    latency = [1, 2, 3]
    payload = [1, 2, 3]

    x = range(len(categories))
    width = 0.22

    ax.bar([i - width for i in x], privacy, width, label="Приватность", color="#74c69d")
    ax.bar(x, latency, width, label="Задержка", color="#f4a261")
    ax.bar([i + width for i in x], payload, width, label="Размер payload", color="#577590")

    ax.set_xticks(list(x))
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 3.5)
    ax.set_ylabel("Относительный уровень")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left")

    for i, label in enumerate(["Низкая", "Средняя", "Высокая"]):
        ax.text(i - width, privacy[i] + 0.08, label, ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    generate_protocol_flow(OUT_DIR / "protocol_flow.png")
    generate_threat_model(OUT_DIR / "threat_model.png")
    generate_math_flow(OUT_DIR / "math_flow.png")
    generate_comparison(OUT_DIR / "plaintext_vs_encoded_vs_phe.png")
    print(f"Схемы сохранены в: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
