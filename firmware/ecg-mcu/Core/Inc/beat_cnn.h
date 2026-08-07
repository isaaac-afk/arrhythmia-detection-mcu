#ifndef BEAT_CNN_H
#define BEAT_CNN_H
// Float inference for the RR-augmented beat classifier (portable C, uses the FPU).
// Inputs are ALREADY preprocessed: beat = z-normalized 252-sample window,
// rr = standardized {prev_RR, RR_ratio}. Returns class probabilities and argmax.
#define BEAT_LEN   252
#define N_CLASSES  5
void beat_cnn_predict(const float beat[BEAT_LEN], const float rr[2], float out[N_CLASSES]);
int  beat_cnn_argmax(const float beat[BEAT_LEN], const float rr[2]);
#endif
