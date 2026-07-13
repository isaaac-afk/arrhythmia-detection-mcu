/* app_ecg.h — Stage 1.3 application: replay a recorded ECG through the
 * detector on-device, report R-peaks + per-sample compute time over UART.
 * Call app_ecg_run() once from main() after peripherals are initialised.
 */
#ifndef APP_ECG_H
#define APP_ECG_H
void app_ecg_run(void);
#endif
