import argparse
import sys
import json
from typing import List, Optional
from lance_bundle import load

def handle_search(args: argparse.Namespace) -> int:
    try:
        bundle = load(args.bundle_path)
        results = bundle.search(args.query, limit=args.limit)

        if args.json:
            print(json.dumps(results, indent=2))
            return 0

        print(f"Search Results for: '{args.query}'")
        for i, res in enumerate(results, start=1):
            text = res.get("text", "<No text found>")
            distance = res.get("_distance", 0.0)
            print(f"\n[{i}] Distance: {distance:.4f}")
            print(f"{repr(text[:100])}...")
                
        return 0
    except Exception as e:
        print(f"Error executing search: {e}", file=sys.stderr)
        return 1

def handle_inspect(args: argparse.Namespace) -> int:
    try:
        bundle = load(args.bundle_path)
        print(f"Bundle Info: {args.bundle_path}")
        print(json.dumps(bundle.metadata(), indent=4))
        return 0
    except Exception as e:
        print(f"Error inspecting bundle: {e}", file=sys.stderr)
        return 1

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lance-bundle",
        description="Tool for searching with and inspecting bundled vectors"
    )
    
    # Subcommand handling
    subparsers = parser.add_subparsers(title="Commands", dest="command", required=True)

    # Search
    search_parser = subparsers.add_parser("search", help="Query against vectors in a bundle")
    search_parser.add_argument("bundle_path", type=str, help="Path to the bundle file")
    search_parser.add_argument("query", type=str, help="The text query to embed and search")
    search_parser.add_argument("--limit", "-l", type=int, default=5, help="Number of results to return (default: 5)")
    search_parser.add_argument("--json", action="store_true", help="Output results as raw JSON")
    search_parser.set_defaults(func=handle_search)

    # Inspect
    inspect_parser = subparsers.add_parser("inspect", help="Inspect a bundle's metadata")
    inspect_parser.add_argument("bundle_path", type=str, help="Path to the bundle file")
    inspect_parser.set_defaults(func=handle_inspect)

    # Parse arguments and dispatch to the correct handler function
    args = parser.parse_args(argv)
    return args.func(args)

if __name__ == "__main__":
    sys.exit(main())
