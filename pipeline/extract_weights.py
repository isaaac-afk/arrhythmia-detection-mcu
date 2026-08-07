import numpy as np, keras
m = keras.models.load_model("beat_cnn_rr.keras")
g = {L.name: L for L in m.layers}

def bn_scale_shift(L):
    gamma, beta, mean, var = L.get_weights()
    eps = L.get_config().get("epsilon", 1e-3)
    scale = gamma / np.sqrt(var + eps)
    shift = beta - mean * scale
    return scale.astype(np.float32), shift.astype(np.float32)

def carr(name, a):
    a = np.asarray(a, dtype=np.float32).ravel()
    s = ", ".join(f"{v:.8e}f" for v in a)
    return f"static const float {name}[{a.size}] = {{ {s} }};\n"

out = ["#ifndef MODEL_WEIGHTS_H\n#define MODEL_WEIGHTS_H\n",
       "// Auto-generated float weights for beat_cnn_rr (BN kept separate: conv->relu->BN).\n\n"]

# conv layers: keras weight shape (k, in, out); keep that layout, index [(k*in+c)*out+o]
for cname, kk, ic, oc in [("conv1d",7,1,16), ("conv1d_1",5,16,32), ("conv1d_2",3,32,32)]:
    W, b = g[cname].get_weights()
    tag = cname.replace("conv1d","conv").replace("_","")
    out.append(carr(f"{tag}_w", W))          # (k,in,out) row-major
    out.append(carr(f"{tag}_b", b))
for bname, tag in [("batch_normalization","bn0"),
                   ("batch_normalization_1","bn1"),
                   ("batch_normalization_2","bn2")]:
    sc, sh = bn_scale_shift(g[bname])
    out.append(carr(f"{tag}_scale", sc))
    out.append(carr(f"{tag}_shift", sh))

Wr, br = g["dense"].get_weights()          # (2,16)
out.append(carr("rr_w", Wr)); out.append(carr("rr_b", br))
W1, b1 = g["dense_1"].get_weights()        # (48,32)
out.append(carr("d1_w", W1)); out.append(carr("d1_b", b1))
W2, b2 = g["dense_2"].get_weights()        # (32,5)
out.append(carr("d2_w", W2)); out.append(carr("d2_b", b2))

out.append("\n#endif\n")
open("model_weights.h","w").writelines(out)
print("wrote model_weights.h")
# also dump a few random test vectors + keras outputs for the C test
rng = np.random.default_rng(0)
N=20
beat = rng.standard_normal((N,252,1)).astype(np.float32)
rr   = rng.standard_normal((N,2)).astype(np.float32)
pred = m.predict({"beat":beat,"rr":rr}, verbose=0)  # softmax probs
np.savetxt("test_in.txt", np.hstack([beat.reshape(N,-1), rr]), fmt="%.8e")
np.savetxt("test_out.txt", pred, fmt="%.8e")
print("wrote test_in.txt / test_out.txt  (N=%d)"%N)
print("keras argmax:", pred.argmax(1))
