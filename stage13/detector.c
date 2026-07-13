#include "detector.h"

/* 5-15 Hz Butterworth bandpass @360 Hz, second-order sections (from scipy). */
static const double SOS[2][6] = {
    /* b0, b1, b2, a0, a1, a2 */
    {0.0067654132571125444, 0.013530826514225089, 0.0067654132571125444,
     1.0, -1.7919338331533654, 0.84122205760775526},
    {1.0, -2.0, 1.0,
     1.0, -1.9193726170801984, 0.92874464520573263},
};

void pt_init(pt_detector *d) {
    for (int i = 0; i < 2; i++) { d->z1[i] = 0.0; d->z2[i] = 0.0; }
    for (int i = 0; i < 5; i++) d->dbuf[i] = 0.0;
    for (int i = 0; i < PT_MWI_WIN; i++) d->mwi_buf[i] = 0.0;
    for (int i = 0; i < PT_BP_HIST; i++) d->bp_hist[i] = 0.0;
    d->mwi_sum = 0.0; d->mwi_pos = 0;
    d->i_prev1 = 0.0; d->i_prev2 = 0.0;
    d->spki = 0.0; d->npki = 0.0;
    d->learn_max = 0.0; d->learn_sum = 0.0; d->learned = 0;
    d->n = 0; d->last_qrs = -PT_REFRACTORY - 1;
}

/* transposed direct form II biquad cascade (matches scipy sosfilt order) */
static double bandpass(pt_detector *d, double x) {
    for (int s = 0; s < 2; s++) {
        double b0 = SOS[s][0], b1 = SOS[s][1], b2 = SOS[s][2];
        double a1 = SOS[s][4], a2 = SOS[s][5];
        double y = b0 * x + d->z1[s];
        d->z1[s] = b1 * x - a1 * y + d->z2[s];
        d->z2[s] = b2 * x - a2 * y;
        x = y;
    }
    return x;
}

static double last_integrated_val = 0.0;

double pt_last_integrated(const pt_detector *d) { (void)d; return last_integrated_val; }

int pt_process(pt_detector *d, double sample, long *rpeak_index) {
    long n = d->n;

    /* 1. causal bandpass */
    double bp = bandpass(d, sample);
    d->bp_hist[n % PT_BP_HIST] = bp;

    /* 2. causal 5-point derivative: (2x[n]+x[n-1]-x[n-3]-2x[n-4]) / 8 */
    d->dbuf[4] = d->dbuf[3]; d->dbuf[3] = d->dbuf[2];
    d->dbuf[2] = d->dbuf[1]; d->dbuf[1] = d->dbuf[0]; d->dbuf[0] = bp;
    double deriv = (2.0*d->dbuf[0] + d->dbuf[1] - d->dbuf[3] - 2.0*d->dbuf[4]) / 8.0;

    /* 3. square */
    double sq = deriv * deriv;

    /* 4. causal moving-window integration (trailing average) */
    d->mwi_sum -= d->mwi_buf[d->mwi_pos];
    d->mwi_buf[d->mwi_pos] = sq;
    d->mwi_sum += sq;
    d->mwi_pos = (d->mwi_pos + 1) % PT_MWI_WIN;
    double integ = d->mwi_sum / (double)PT_MWI_WIN;
    last_integrated_val = integ;

    int fired = 0;

    /* 5. threshold learning over the first PT_LEARN samples */
    if (n < PT_LEARN) {
        if (integ > d->learn_max) d->learn_max = integ;
        d->learn_sum += integ;
        if (n == PT_LEARN - 1) {
            d->spki = 0.25 * d->learn_max;
            d->npki = 0.5  * (d->learn_sum / (double)PT_LEARN);
            d->learned = 1;
        }
    } else {
        /* 6. local-max detection on the integrated signal at index n-1 */
        long p = n - 1;
        if (d->i_prev1 > integ && d->i_prev1 >= d->i_prev2 && p >= PT_LEARN) {
            double v = d->i_prev1;
            double thr = d->npki + 0.25 * (d->spki - d->npki);
            if (v > thr && (p - d->last_qrs) > PT_REFRACTORY) {
                /* refine: argmax of bandpassed signal over [p-REFINE_WIN, p] */
                long lo = p - PT_REFINE_WIN; if (lo < 0) lo = 0;
                long best = lo; double bestv = d->bp_hist[lo % PT_BP_HIST];
                for (long k = lo; k <= p; k++) {
                    double val = d->bp_hist[k % PT_BP_HIST];
                    if (val > bestv) { bestv = val; best = k; }
                }
                *rpeak_index = best;
                d->last_qrs = p;
                d->spki = 0.125 * v + 0.875 * d->spki;
                fired = 1;
            } else {
                d->npki = 0.125 * v + 0.875 * d->npki;
            }
        }
    }

    d->i_prev2 = d->i_prev1;
    d->i_prev1 = integ;
    d->n = n + 1;
    return fired;
}
