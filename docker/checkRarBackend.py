"""Build-time check: can this image read RAR5 *correctly*?

Run from the Dockerfile. It exists because "a rar tool is installed" and "a rar
tool works" turned out to be very different claims, and the gap between them is
silent. Measured against a real 200-file RAR5 corpus:

    unrar    200/200 members read correctly
    unar     195/200 -- five members raise, with no size pattern
             (2KB, 6.5KB, 7KB, 21KB, 26KB), so it is a solid-block streaming
             problem rather than anything a fixture can be shaped to trigger
    bsdtar   0/200 -- returns 51 bytes for a 39KB member and reports success

``app.ingestion.documents._archiveText`` deliberately skips a member it cannot
read rather than losing the whole document, which is the right call and also
means a truncating backend builds a RAG database quietly missing files. Nobody
notices; the answers are just worse.

**Two checks, because one is not enough.** The probe below round-trips a small
RAR5 archive, which catches "no backend at all" and a broken unrar build. It
does *not* catch ``unar`` -- a two-member fixture reads fine there, and the real
failures follow no pattern a synthetic archive reproduces. So the backend is
also checked by name. Naming a tool is cruder than testing behaviour, and it is
used here precisely because the behaviour cannot be tested cheaply: the only
honest summary of the measurements above is "use unrar".
"""

import shutil
import sys

import rarfile

PROBE = "/app/docker/rar5Probe.rar"


def main() -> int:
    if shutil.which("unrar") is None:
        print(
            "FAIL: `unrar` is not on PATH. It is the only backend measured to read "
            "RAR5 correctly -- unar and bsdtar both truncate members silently. "
            "Install it from Debian non-free; see the Dockerfile.",
            file=sys.stderr,
        )
        return 1

    try:
        rarfile.tool_setup(force=True)
    except rarfile.RarCannotExec as exc:
        print(f"FAIL: rarfile found no usable backend ({exc})", file=sys.stderr)
        return 1

    try:
        with rarfile.RarFile(PROBE) as archive:
            members = archive.infolist()
            if not members:
                print("FAIL: the probe archive listed no members", file=sys.stderr)
                return 1

            for info in members:
                body = archive.read(info)
                if len(body) != info.file_size:
                    print(
                        f"FAIL: '{info.filename}' read back {len(body)} bytes, but the "
                        f"archive header says {info.file_size}. This backend truncates "
                        f"RAR5 members.",
                        file=sys.stderr,
                    )
                    return 1
    except Exception as exc:
        print(
            f"FAIL: could not read the RAR5 probe: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    print(f"rar backend ok: unrar, {len(members)} probe members verified byte-exact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
