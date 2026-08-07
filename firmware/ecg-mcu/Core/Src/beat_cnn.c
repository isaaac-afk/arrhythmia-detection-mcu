#include "beat_cnn.h"
#include "model_weights.h"
#include <string.h>
#include <math.h>

/* generic 1-D conv, "same" padding, stride 1, optional fused ReLU.
 * in/out laid out [t*C + c]; weights [(kk*Cin + c)*Cout + o] (Keras k,in,out). */
static void conv1d_same(const float *in, float *out, const float *w, const float *b,
                        int k, int T, int Cin, int Cout, int relu) {
    int pad = (k - 1) / 2;
    for (int t = 0; t < T; t++)
        for (int o = 0; o < Cout; o++) {
            float acc = b[o];
            for (int kk = 0; kk < k; kk++) {
                int ti = t + kk - pad;
                if (ti < 0 || ti >= T) continue;
                const float *xin = in + ti * Cin;
                const float *wk = w + (kk * Cin) * Cout + o;
                for (int c = 0; c < Cin; c++) acc += wk[c * Cout] * xin[c];
            }
            if (relu && acc < 0.0f) acc = 0.0f;
            out[t * Cout + o] = acc;
        }
}
static void bn_affine(float *x, const float *scale, const float *shift, int T, int C) {
    for (int t = 0; t < T; t++)
        for (int c = 0; c < C; c++) x[t * C + c] = x[t * C + c] * scale[c] + shift[c];
}
static void maxpool2(const float *in, float *out, int T, int C) {
    int To = T / 2;
    for (int t = 0; t < To; t++)
        for (int c = 0; c < C; c++) {
            float a = in[(2 * t) * C + c], b = in[(2 * t + 1) * C + c];
            out[t * C + c] = a > b ? a : b;
        }
}
static void gap(const float *in, float *out, int T, int C) {
    for (int c = 0; c < C; c++) {
        float s = 0.0f;
        for (int t = 0; t < T; t++) s += in[t * C + c];
        out[c] = s / (float)T;
    }
}
static void dense(const float *x, float *out, const float *w, const float *b,
                  int In, int Out, int relu) {
    for (int j = 0; j < Out; j++) {
        float acc = b[j];
        for (int i = 0; i < In; i++) acc += w[i * Out + j] * x[i];
        if (relu && acc < 0.0f) acc = 0.0f;
        out[j] = acc;
    }
}

/* activation buffers (max element counts) */
static float A[252 * 32];
static float B[252 * 32];

void beat_cnn_predict(const float beat[BEAT_LEN], const float rr[2], float out[N_CLASSES]) {
    /* beat branch */
    conv1d_same(beat, A, conv_w, conv_b, 7, 252, 1, 16, 1);   /* -> A[252*16] */
    bn_affine(A, bn0_scale, bn0_shift, 252, 16);
    maxpool2(A, B, 252, 16);                                  /* -> B[126*16] */
    conv1d_same(B, A, conv1_w, conv1_b, 5, 126, 16, 32, 1);   /* -> A[126*32] */
    bn_affine(A, bn1_scale, bn1_shift, 126, 32);
    maxpool2(A, B, 126, 32);                                  /* -> B[63*32] */
    conv1d_same(B, A, conv2_w, conv2_b, 3, 63, 32, 32, 1);    /* -> A[63*32] */
    bn_affine(A, bn2_scale, bn2_shift, 63, 32);
    float feat[48];
    gap(A, feat, 63, 32);                                     /* feat[0..31] */
    /* rr branch */
    dense(rr, feat + 32, rr_w, rr_b, 2, 16, 1);               /* feat[32..47] */
    /* head */
    float h[32];
    dense(feat, h, d1_w, d1_b, 48, 32, 1);
    float logits[5];
    dense(h, logits, d2_w, d2_b, 32, 5, 0);
    /* softmax */
    float mx = logits[0];
    for (int i = 1; i < 5; i++) if (logits[i] > mx) mx = logits[i];
    float s = 0.0f;
    for (int i = 0; i < 5; i++) { out[i] = expf(logits[i] - mx); s += out[i]; }
    for (int i = 0; i < 5; i++) out[i] /= s;
}
int beat_cnn_argmax(const float beat[BEAT_LEN], const float rr[2]) {
    float p[5]; beat_cnn_predict(beat, rr, p);
    int a = 0; for (int i = 1; i < 5; i++) if (p[i] > p[a]) a = i; return a;
}

#ifdef BEATCNN_TEST
#include <stdio.h>
int main(void) {
    float beat[252], rr[2], out[5];
    /* read test_in.txt: each line = 252 beat + 2 rr floats */
    while (1) {
        for (int i = 0; i < 252; i++) if (scanf("%f", &beat[i]) != 1) return 0;
        for (int i = 0; i < 2; i++) scanf("%f", &rr[i]);
        beat_cnn_predict(beat, rr, out);
        printf("%.8e %.8e %.8e %.8e %.8e\n", out[0], out[1], out[2], out[3], out[4]);
    }
}
#endif
