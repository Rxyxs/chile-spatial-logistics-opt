"""Automatiza `git add` + `git commit` + `git push` para este repositorio.

Uso:
    python scripts/auto_push.py -m "mensaje de commit"
    python scripts/auto_push.py --branch main --yes

Por defecto pide confirmacion interactiva antes de hacer commit y push.
Usa `--yes` para omitirla (por ejemplo, en un pipeline no interactivo).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime


def run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit(result.returncode)


def has_pending_changes() -> bool:
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    return bool(status.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Sincroniza cambios locales con GitHub")
    parser.add_argument("-m", "--message", default=None, help="Mensaje de commit")
    parser.add_argument("--branch", default="main", help="Rama de destino para el push")
    parser.add_argument(
        "--yes", action="store_true", help="Omite la confirmacion interactiva"
    )
    args = parser.parse_args()

    message = args.message or f"Update: {datetime.now():%Y-%m-%d %H:%M}"

    run(["git", "add", "-A"])

    if not has_pending_changes():
        print("No hay cambios para commitear.")
        return

    if not args.yes:
        confirm = input(
            f"Confirmar commit y push a '{args.branch}' con mensaje '{message}'? [y/N] "
        )
        if confirm.strip().lower() != "y":
            print("Cancelado.")
            return

    run(["git", "commit", "-m", message])
    run(["git", "push", "origin", args.branch])
    print("Push completado.")


if __name__ == "__main__":
    main()
