# Continuum v1.0.0 — Commercial Release Package

- **Company / Publisher:** `ahkscriptlerim`
- **Author / Architect:** `gtaking`
- **Executable:** `Continuum.exe` (Signed Win32 x64 Native AVX2+FMA Release)
- **First-Boot Auto-Calibration:** `Continuum.ini` is intentionally omitted from releases so every end-user PC automatically probes its own GPU VRAM, CPU thread pool, and Per-Monitor V2 DPI on first launch.

## Distribution Artifacts (`manual release\win\shipment`)
1. **`Continuum_v1.0.0_Setup_x64.exe`** — Signed Inno Setup 6 (`lzma2/ultra64`) commercial installer (supports zero-UAC Per-User install & Admin All-Users install).
2. **`Continuum.exe`** — Clean zero-install Win64 portable exe.
3. **`SHA256SUMS.txt` & `SHIPMENT_MANIFEST.md`** — Cryptographic SHA-256 & PE/Authenticode audit telemetry.
