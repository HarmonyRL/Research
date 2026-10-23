"""Serve rendered files on localhost for browsing through an SSH tunnel.

Run here, then from the local machine: ssh -L 8000:localhost:8000 <host> and open
http://localhost:8000 in a browser.
"""

import argparse
from pathlib import Path

from harmonyrl.tools import serve


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--root", default="runs", type=Path)
	parser.add_argument("--port", type=int, default=8000)
	args = parser.parse_args()
	print(f"serving {args.root.resolve()} on http://127.0.0.1:{args.port} (Ctrl+C to stop)")
	serve(args.root, args.port)
	print("stopped")


if __name__ == "__main__":
	main()
