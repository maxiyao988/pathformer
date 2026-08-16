from pathlib import Path

path = Path("scripts/python/date.md")
lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

new_lines = []
for idx, line in enumerate(lines, start=1):
    year = "2023" if idx <= 728 else "2022"
    new_lines.append(f"{year}/{line}")

path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")