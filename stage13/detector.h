/* detector.h — streaming, causal Pan-Tompkins R-peak detector.
 * Portable C99, no platform headers. Feed one sample at a time via
 * pt_process(); it returns 1 on the sample where an R-peak is confirmed,
 * writing the R-peak's absolute sample index to *rpeak_index.
 *
 * This is the exact algorithm that ports to the STM32: no future samples,
 * bounded memory, integer/double arithmetic only.
 */
#ifndef DETECTOR_H
#define DETECTOR_H

#define PT_FS            360
#define PT_MWI_WIN       54     /* round(0.150 * fs)  moving-window integrator */
#define PT_REFRACTORY    72     /* round(0.200 * fs)  physiological refractory */
#define PT_LEARN         720    /* 2 * fs  threshold learning phase            */
#define PT_REFINE_WIN    90     /* round(0.250 * fs)  R-peak look-back window  */
#define PT_BP_HIST       128    /* ring buffer >= PT_REFINE_WIN + margin       */

typedef struct {
    /* bandpass biquad (2 sections, transposed direct form II) states */
    double z1[2], z2[2];
    /* causal derivative buffer: last 5 bandpassed samples */
    double dbuf[5];
    /* moving-window integrator */
    double mwi_buf[PT_MWI_WIN];
    double mwi_sum;
    int    mwi_pos;
    /* bandpassed history ring (for R-peak refinement) */
    double bp_hist[PT_BP_HIST];
    /* integrated-signal local-max detection state */
    double i_prev1, i_prev2;
    /* adaptive thresholds */
    double spki, npki;
    double learn_max, learn_sum;
    int    learned;
    /* bookkeeping */
    long   n;          /* absolute sample counter */
    long   last_qrs;   /* absolute index of last accepted QRS */
} pt_detector;

void pt_init(pt_detector *d);
/* returns 1 if an R-peak was confirmed at this step (index in *rpeak_index) */
int  pt_process(pt_detector *d, double sample, long *rpeak_index);
/* exposed for bit-comparison: the integrated-signal value at the current step */
double pt_last_integrated(const pt_detector *d);

#endif
