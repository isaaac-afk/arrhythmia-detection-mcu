# Firmware — ecg-mcu

STM32CubeIDE project. Nucleo-F411RE, Cortex-M4F @ **84 MHz** (`CPU_MHZ=84`).

Replays a recorded MIT-BIH slice from flash through the Stage 1.2 detector,
printing R-peaks and DWT cycle counts over USART2.

## Layout

| Path | |
|---|---|
| `Core/Src/app_ecg.c` | replay loop, DWT timing, reporting |
| `Core/Src/detector.c` | Pan-Tompkins, identical to `c-reference/` |
| `Core/Inc/ecg_data.h` | generated — do not edit by hand |
| `board_peaks_rec100.txt` | captured device output, record 100 |

## Regenerating the ECG data

    cd stage13
    C:\Users\isaac\anaconda3\envs\ecg\python.exe export_to_c.py 100 --seconds 60
    Copy-Item ecg_data.h ..\firmware\ecg-mcu\Core\Inc\ecg_data.h -Force

Writes `ecg_data.h` **and** `expected_rpeaks.txt` (the golden reference for
that exact slice). Refresh the project in CubeIDE (F5) afterwards or Eclipse
will rebuild the stale array.

## Flashing (Windows on ARM)

**Drag `Debug/ecg-mcu.bin` onto the `NOD_F411RE` drive, then right-click ->
Eject.** Never yank the cable.

Three things that do NOT work on this machine, all for the same root cause —
no ARM64-signed ST-LINK driver exists:

- CubeIDE's green Run: fails `DEV_NO_STLINK`
- STM32CubeProgrammer: same
- The community `stsw-link009_v3_arm64` unsigned driver: will not install
  (Secure Boot ignores the F7 signature-disable option)

Device Manager shows "ST-Link Debug" with **Code 28**. That is expected here.

**Flash `.bin`, never `.hex`.** The board's 2019 ST-LINK MSD firmware
(DETAILS.TXT: Version 0221, Build Jan 7 2019) silently discards `.hex` files
written by Windows 11 — the file vanishes from the drive, no `FAIL.TXT` is
produced, and the chip is unchanged. There is no error of any kind. This cost
two days to diagnose.

Enable `.bin` output: Project Properties -> C/C++ Build -> Settings ->
Tool Settings -> MCU Post build outputs -> "Convert to binary file".

**Confirming a flash actually landed:** toggle LD2 at a distinctive rate
(`HAL_GPIO_TogglePin(LD2_GPIO_Port, LD2_Pin)`). Nothing else on the board
drives LD2, so a changed blink rate proves new code is running.

**Untried fallback** if MSD flashing ever fails: the F411 ROM UART
bootloader. Jumper CN7 pin5 <-> pin7 to pull BOOT0 high (spare jumpers live
on CN11/CN12), reset, then CubeProgrammer in UART mode on COM3 @ 115200
**EVEN parity**. Bypasses the ST-LINK debug interface entirely.

## Reading serial

    python stage13\capture_serial.py     # captures to run.log, then diff

Or interactively:

    python -m serial.tools.miniterm COM3 115200

Only one process can hold COM3 — quit miniterm (Ctrl+]) before running the
capture script, or you get `PermissionError(13)`.

PuTTY's Open button does not work on this machine; the in-box `usbser` CDC
driver is fine, so miniterm and pyserial both work.

## Verifying a run

    python stage13\capture_serial.py
    # press RESET
    Select-String -Path run.log -Pattern "R-peak @ (\d+)" |
      ForEach-Object { $_.Matches[0].Groups[1].Value } |
      Set-Content firmware\board_peaks_rec100.txt
    $diff = Compare-Object (Get-Content firmware\board_peaks_rec100.txt) `
                           (Get-Content stage13\expected_rpeaks.txt)
    if (-not $diff) { "MATCH" } else { $diff }

See `docs/results.md` for measured latency and footprint.

---

**Not a medical device.**
