#!/usr/bin/env bash
# Static (musl) release build of `cadence`.
#
# Produces a fully static binary at:
#   target/x86_64-unknown-linux-musl/release/cadence
# which runs on any x86_64 Linux (glibc or musl distros).
#
# Toolchain resolution (first hit wins):
#   1. $MUSL_CROSS_BIN dir containing x86_64-linux-musl-gcc (e.g. musl.cc cross toolchain)
#   2. ./.scratch/x86_64-linux-musl-cross/bin (rootless Fedora-friendly spot)
#   3. musl-gcc / musl-g++ on PATH (e.g. `sudo dnf install musl-gcc musl-libc-static gcc-c++`)
#
# Also needs: rustup target add x86_64-unknown-linux-musl
set -euo pipefail
cd "$(dirname "$0")"

PREFIX=""
if [[ -n "${MUSL_CROSS_BIN:-}" && -x "$MUSL_CROSS_BIN/x86_64-linux-musl-gcc" ]]; then
    PREFIX="$MUSL_CROSS_BIN/x86_64-linux-musl"
elif [[ -x .scratch/x86_64-linux-musl-cross/bin/x86_64-linux-musl-gcc ]]; then
    PREFIX="$PWD/.scratch/x86_64-linux-musl-cross/bin/x86_64-linux-musl"
elif command -v musl-gcc >/dev/null 2>&1 && command -v musl-g++ >/dev/null 2>&1; then
    PREFIX="musl"
fi

if [[ -z "$PREFIX" ]]; then
    echo "error: no musl C/C++ toolchain found." >&2
    echo "  option A (rootless): unpack https://musl.cc/x86_64-linux-musl-cross.tgz" >&2
    echo "               into ./.scratch/x86_64-linux-musl-cross" >&2
    echo "  option B (Fedora):   sudo dnf install musl-gcc musl-libc-static gcc-c++" >&2
    echo "  then: rustup target add x86_64-unknown-linux-musl" >&2
    exit 1
fi

# Plain `musl-gcc` on PATH wraps system gcc; the cross toolchain ships real binaries.
if [[ "$PREFIX" == "musl" ]]; then
    CC_BIN="musl-gcc"
    CXX_BIN="musl-g++"
    AR_BIN="ar"
else
    CC_BIN="$PREFIX-gcc"
    CXX_BIN="$PREFIX-g++"
    AR_BIN="$PREFIX-ar"
fi

export CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_LINKER="$CC_BIN"
export CC_x86_64_unknown_linux_musl="$CC_BIN"
export CXX_x86_64_unknown_linux_musl="$CXX_BIN"
export AR_x86_64_unknown_linux_musl="$AR_BIN"

# Size over speed: this app is LLM- and I/O-bound, so optimize for a small
# portable binary. Rust side only; the C/C++ deps (MuPDF, SQLite) follow
# CFLAGS below (appended after cc-crate defaults, so -Os wins).
# DEBUG=false: our release profile keeps line tables for benchmark fidelity,
# but the portable binary ships without them (saves ~30MB; matches the old
# learning-app/cadence artifact, which had no debug info).
export CARGO_PROFILE_RELEASE_OPT_LEVEL="s"
export CARGO_PROFILE_RELEASE_DEBUG="false"
export CFLAGS_x86_64_unknown_linux_musl="-Os"
export CXXFLAGS_x86_64_unknown_linux_musl="-Os"

cargo build --release --target x86_64-unknown-linux-musl "$@"

BIN="target/x86_64-unknown-linux-musl/release/cadence"
echo "---"
file "$BIN"
ls -lh "$BIN"
