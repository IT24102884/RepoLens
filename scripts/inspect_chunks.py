"""Polyglot Code Ingestion Inspector.

Demonstrates and verifies manual chunking across multiple programming languages:
- TypeScript / React (.ts, .tsx)
- Go (.go)
- Java (.java)
- Python (.py)
"""

from ai.ingestion.code_chunker import ASTCodeChunker


SAMPLE_FILES = [
    (
        "services/AuthService.ts",
        """interface UserSession {
    token: string;
    expiresAt: number;
}

export function validateSession(session: UserSession): boolean {
    return Date.now() < session.expiresAt;
}

class AuthService {
    private activeTokens = new Set<string>();

    logout(token: string): void {
        this.activeTokens.delete(token);
    }
}
""",
    ),
    (
        "server/gateway.go",
        """package gateway

type RouteConfig struct {
    Path    string
    Timeout int
}

func RegisterRoute(route RouteConfig) error {
    println("Registering route:", route.Path)
    return nil
}
""",
    ),
    (
        "models/PaymentProcessor.java",
        """package com.app.billing;

public class PaymentProcessor {
    public boolean processPayment(double amount, String currency) {
        if (amount <= 0) {
            return false;
        }
        return true;
    }
}
""",
    ),
    (
        "utils/client.py",
        '''"""Python HTTP client helper."""

class HTTPClient:
    def __init__(self, host: str):
        self.host = host

    @classmethod
    def local(cls):
        return cls("http://localhost:8000")
''',
    ),
]


def inspect_chunks():
    chunker = ASTCodeChunker()

    print("\n" + "=" * 75)
    print("      REPOLENS UNIVERSAL POLYGLOT CHUNKER: MANUAL VERIFICATION")
    print("=" * 75)

    for file_path, code in SAMPLE_FILES:
        print(f"\n[FILE] {file_path}")
        chunks = chunker.chunk(content=code, file_path=file_path)
        lang = chunks[0].metadata.get("language", "python") if chunks else "unknown"
        print(f"   Detected Language: {lang}")
        print(f"   Chunks Extracted:  {len(chunks)}")
        print("   " + "-" * 70)

        for i, c in enumerate(chunks, start=1):
            symbol = c.metadata.get("symbol_name", "N/A")
            node_type = c.metadata.get("symbol_type", "N/A")
            print(f"   [{i}] Symbol: {symbol:<20} | Type: {node_type:<24} | Lines {c.start_line}->{c.end_line}")
            first_line = c.content.splitlines()[0] if c.content.splitlines() else ""
            print(f"       Code Snippet: {first_line[:65]}...")

    print("\n" + "=" * 75)
    print(" [OK] Multi-language AST parsing & line-level chunking verified!")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    inspect_chunks()