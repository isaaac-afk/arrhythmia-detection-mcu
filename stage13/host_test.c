/* Host test: compile app_ecg with the real detector + synthetic ecg_data.h,
 * capture the R-peaks it prints, and (separately) diff against the golden
 * reference list. Proves the on-device processing loop is correct before
 * it ever touches hardware. */
#define HOST_TEST
#include "detector.c"
#include "app_ecg.c"
int main(void) { app_ecg_run(); return 0; }
