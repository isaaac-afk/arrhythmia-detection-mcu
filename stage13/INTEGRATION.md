# Integrating the detector into your CubeIDE project

Do this once the project exists. We'll create the project together (screenshots),
but here's the map so you know where everything goes.

## 1. Create the project
File → New → STM32 Project → **Board Selector** tab → search **NUCLEO-F411RE**
→ select it → Next → name it `ecg-mcu` → Finish.
When asked "Initialize all peripherals with default mode?" → **Yes**.
(This auto-configures USART2 to the ST-LINK virtual COM port and LD2 on PA5.)

## 2. Configure in the Device Configuration Tool (.ioc)
- **USART2**: should already be enabled (Asynchronous). Set baud rate **115200**,
  8N1. This is your serial link over the same USB cable.
- **Clock Configuration** tab: set **HCLK (SYSCLK) to 100 MHz** (max for F411).
  If it's already 84 or 100, note the number — it sets `CPU_MHZ`.
- Save the `.ioc` → it regenerates code.

## 3. Add the source files
Copy into the project:
- `detector.h`, `app_ecg.h`, `ecg_data.h`  → `Core/Inc/`
- `detector.c`, `app_ecg.c`                → `Core/Src/`
(Right-click the folder → Import, or just drag them in. "Copy files" = yes.)

If your `CPU_MHZ` isn't 100, add to the project's C defines (Project → Properties
→ C/C++ Build → Settings → Preprocessor) `CPU_MHZ=84` (or whatever your clock is).

## 4. Edit main.c (only inside the USER CODE blocks)

**Includes** — find `/* USER CODE BEGIN Includes */`:
```c
/* USER CODE BEGIN Includes */
#include <stdio.h>
#include "app_ecg.h"
/* USER CODE END Includes */
```

**printf retarget** — find `/* USER CODE BEGIN 4 */` (near the bottom):
```c
/* USER CODE BEGIN 4 */
int __io_putchar(int ch) {
    HAL_UART_Transmit(&huart2, (uint8_t *)&ch, 1, HAL_MAX_DELAY);
    return ch;
}
/* USER CODE END 4 */
```
This makes `printf` go out USART2 → your PC serial terminal.
(If nothing prints, try `_write` instead — ask me and I'll give that variant.)

**Run the app once** — find `/* USER CODE BEGIN 2 */` (after the init calls,
before the while(1) loop):
```c
/* USER CODE BEGIN 2 */
app_ecg_run();
/* USER CODE END 2 */
```

## 5. Build
Click the hammer (Build). It should compile with 0 errors — no board needed yet.
Fix any missing-include path issues by confirming the files are in Core/Inc,Src.

## 6. Flash + view (when the board arrives)
- Plug in the Nucleo (mini-USB). Click the green **Run** ▶.
- Open a serial terminal on the ST-LINK COM port @ **115200 8N1**. Options:
  CubeIDE has no built-in terminal by default — use PuTTY, or the VS Code
  "Serial Monitor" extension, or Tera Term. (I'll help you pick.)
- Press the black RESET button to replay. You should see:
  ```
  === Stage 1.3: replay 3600 samples @ 360 Hz ===
  R-peak @ 765
  R-peak @ 1048
  ...
  --- done: N R-peaks ---
  compute/sample: avg ... cyc (... ns), worst ... cyc (... ns)
  real-time budget @ 360 Hz = 2777 us/sample -> PASS (fits)
  ```

## 7. Verify the checkpoint
Compare the printed `R-peak @ ...` indices against `expected_rpeaks.txt`.
They should match exactly. That + the PASS timing line closes Stage 1.3.
