"""Small printing helpers so every lab looks the same on the projector."""


def title(text):
    line = "=" * 78
    print(f"\n{line}\n  {text}\n{line}")


def section(text):
    print(f"\n--- {text}")


def table(headers, rows, widths=None):
    widths = widths or [max(len(str(h)), *(len(str(r[i])) for r in rows)) + 2 for i, h in enumerate(headers)]
    fmt = "".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print("-" * sum(widths))
    for r in rows:
        print(fmt.format(*[str(x) for x in r]))


def notice(lines):
    print("\nWhat to notice:")
    for ln in lines:
        print(f"  * {ln}")


def note_prompt(lab, text):
    print(f"\nLAB_NOTES.md, Lab {lab}: {text}\n")


def f3(x):
    return f"{x:.3f}"
