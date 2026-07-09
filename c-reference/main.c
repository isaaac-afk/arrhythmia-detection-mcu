/* main.c — run the detector over a CSV of ECG samples (one value per line).
 * Usage: ./detector input.csv rpeaks_out.txt [integrated_out.txt]
 */
#include <stdio.h>
#include <stdlib.h>
#include "detector.h"

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s input.csv rpeaks_out.txt [integ_out.txt]\n", argv[0]);
        return 1;
    }
    FILE *in = fopen(argv[1], "r");
    if (!in) { perror("input"); return 1; }
    FILE *rout = fopen(argv[2], "w");
    FILE *iout = (argc >= 4) ? fopen(argv[3], "w") : NULL;

    pt_detector d;
    pt_init(&d);

    double sample; long ridx;
    while (fscanf(in, "%lf", &sample) == 1) {
        int fired = pt_process(&d, sample, &ridx);
        if (iout) fprintf(iout, "%.17g\n", pt_last_integrated(&d));
        if (fired) fprintf(rout, "%ld\n", ridx);
    }
    fclose(in); fclose(rout); if (iout) fclose(iout);
    return 0;
}
