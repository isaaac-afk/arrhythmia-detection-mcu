/* biquad.h — Stage 1.4d real-time noise filters (portable C).
 *
 * Two second-order sections, direct-form II transposed, single-precision
 * (runs on the F411's hardware FPU — cheap). Coefficients come straight from
 * pipeline/noise_filters.py (scipy), so the C and Python filters match.
 *
 *   section 0: baseline-wander high-pass, 0.5 Hz, 2nd-order Butterworth
 *   section 1: mains notch, 60 Hz, Q=30
 *
 * Usage (LIVE path only — do NOT filter clean replay data, it would break the
 * bit-exact golden-reference check):
 *
 *     ecg_filter_reset();
 *     for each sample:
 *         float clean = ecg_filter_sample((float)raw);
 *         pt_process(&d, (double)clean, &ridx);   // detector still takes double
 *
 * Let the filters settle ~0.5 s (a causal IIR has a startup transient) before
 * trusting the output.
 */
#ifndef BIQUAD_H
#define BIQUAD_H

typedef struct {
    float b0, b1, b2, a1, a2;   /* coefficients (a0 normalised to 1) */
    float z1, z2;               /* state */
} biquad_t;

/* one sample through one section, direct-form II transposed */
static inline float biquad_step(biquad_t *s, float x)
{
    float y = s->b0 * x + s->z1;
    s->z1   = s->b1 * x - s->a1 * y + s->z2;
    s->z2   = s->b2 * x - s->a2 * y;
    return y;
}

void  ecg_filter_reset(void);        /* zero all filter state */
float ecg_filter_sample(float x);    /* HP -> notch, returns filtered sample */

#endif /* BIQUAD_H */
