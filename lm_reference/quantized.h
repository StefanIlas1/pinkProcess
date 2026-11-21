/* Fixed-point forward pass for the pinned TinyStories checkpoint. */

enum { PP_DIM = 64, PP_HIDDEN = 172, PP_LAYERS = 5, PP_HEADS = 8, PP_KV = 32,
       PP_HEAD = 8, PP_VOCAB = 512, PP_SEQ = 512, PP_ONE = 65536,
       PP_LIMIT = 1 << 24, PP_EXP_INTERVAL = 1024, PP_EXP_LAST = 1024 };

#define PP_MATRIX(name, rows, cols) int32_t name##_scale[rows]; int8_t name[(rows) * (cols)]
static struct {
    int32_t embedding[PP_VOCAB * PP_DIM], rms_att[PP_LAYERS * PP_DIM];
    PP_MATRIX(wq, PP_LAYERS * PP_DIM, PP_DIM);
    PP_MATRIX(wk, PP_LAYERS * PP_KV, PP_DIM);
    PP_MATRIX(wv, PP_LAYERS * PP_KV, PP_DIM);
    PP_MATRIX(wo, PP_LAYERS * PP_DIM, PP_DIM);
    int32_t rms_ffn[PP_LAYERS * PP_DIM];
    PP_MATRIX(w1, PP_LAYERS * PP_HIDDEN, PP_DIM);
    PP_MATRIX(w2, PP_LAYERS * PP_DIM, PP_HIDDEN);
    PP_MATRIX(w3, PP_LAYERS * PP_HIDDEN, PP_DIM);
    int32_t rms_final[PP_DIM];
    int8_t classifier[PP_VOCAB * PP_DIM];
    int32_t rotary[PP_SEQ * PP_HEAD], exponential[PP_EXP_LAST + 1];
} pp_model;
#undef PP_MATRIX

static int32_t pp_keys[PP_LAYERS][PP_SEQ][PP_KV];
static int32_t pp_values[PP_LAYERS][PP_SEQ][PP_KV];
static int pp_enabled;
static int pp_score_fd = -1;

static void pp_load(Config *config, const char *path) {
    unsigned char header[8];
    FILE *file = fopen(path, "rb");
    if (!file || fread(header, sizeof(header), 1, file) != 1 ||
        memcmp(header, "PPQ16V1", sizeof(header)) != 0) {
        fprintf(stderr, "Expected a pinkProcess quantized checkpoint.\n");
        exit(EXIT_FAILURE);
    }
    /* Read tensors individually so C struct padding is never part of the file. */
    int count = 0;
#define PP_READ(field) count += fread(pp_model.field, sizeof(pp_model.field), 1, file)
#define PP_READ_MATRIX(field) PP_READ(field##_scale); PP_READ(field)
    PP_READ(embedding);
    PP_READ(rms_att);
    PP_READ_MATRIX(wq);
    PP_READ_MATRIX(wk);
    PP_READ_MATRIX(wv);
    PP_READ_MATRIX(wo);
    PP_READ(rms_ffn);
    PP_READ_MATRIX(w1);
    PP_READ_MATRIX(w2);
    PP_READ_MATRIX(w3);
    PP_READ(rms_final);
    PP_READ(classifier);
    PP_READ(rotary);
    PP_READ(exponential);
#undef PP_READ_MATRIX
#undef PP_READ
    if (count != 21 || fgetc(file) != EOF || ferror(file)) {
        fprintf(stderr, "Quantized checkpoint has an unexpected tensor size.\n");
        exit(EXIT_FAILURE);
    }
    fclose(file);
    *config = (Config){PP_DIM, PP_HIDDEN, PP_LAYERS, PP_HEADS, PP_KV / PP_HEAD, PP_VOCAB, PP_SEQ};
    load_replacement_row(pp_model.classifier, PP_DIM, PP_VOCAB);
    const char *fd = getenv("PINKPROCESS_SCORE_FD");
    if (fd && (pp_score_fd = atoi(fd)) < 3) {
        fprintf(stderr, "Invalid live classifier descriptor.\n");
        exit(EXIT_FAILURE);
    }
    fprintf(stderr, "weights integer");
    for (int i = 0; i < PP_VOCAB * PP_DIM; i++) {
        fprintf(stderr, " %d", pp_model.classifier[i]);
    }
    fputc('\n', stderr);
}

static int64_t pp_round_div(int64_t value, int64_t divisor) {
    return value >= 0 ? (value + divisor / 2) / divisor : -((-value + divisor / 2) / divisor);
}

static int32_t pp_clip(int64_t value) {
    return value < -PP_LIMIT ? -PP_LIMIT : value > PP_LIMIT ? PP_LIMIT : (int32_t)value;
}

/* Restoring integer square root, rounded down. */
static uint64_t pp_sqrt(uint64_t value) {
    uint64_t root = 0;
    uint64_t bit = UINT64_C(1) << 62;
    while (bit > value) bit >>= 2;
    while (bit != 0) {
        if (value >= root + bit) {
            value -= root + bit;
            root = (root >> 1) + bit;
        } else {
            root >>= 1;
        }
        bit >>= 2;
    }
    return root;
}

static void pp_rms(int32_t *out, const int32_t *x, const int32_t *gain) {
    uint64_t squares = 0;
    for (int i = 0; i < PP_DIM; i++) squares += (int64_t)x[i] * x[i];
    int64_t rms = pp_sqrt(squares / PP_DIM + 42950);  /* 1e-5 in Q32 */
    for (int i = 0; i < PP_DIM; i++) out[i] = pp_clip(pp_round_div((int64_t)x[i] * gain[i], rms));
}

static void pp_project(int32_t *out, const int32_t *x, const int8_t *weights, const int32_t *scale,
                       int inputs, int outputs) {
    for (int row = 0; row < outputs; row++) {
        int64_t sum = 0;
        for (int col = 0; col < inputs; col++) sum += (int64_t)x[col] * weights[row * inputs + col];
        out[row] = pp_clip(pp_round_div(sum * scale[row], INT64_C(1) << 24));
    }
}

static void pp_rotate(int32_t *x, int width, int position) {
    for (int i = 0; i < width; i += 2) {
        int pair = position * PP_HEAD + i % PP_HEAD;
        int32_t c = pp_model.rotary[pair], s = pp_model.rotary[pair + 1];
        int32_t a = x[i], b = x[i + 1];
        x[i] = pp_clip(pp_round_div((int64_t)a * c - (int64_t)b * s, PP_ONE));
        x[i + 1] = pp_clip(pp_round_div((int64_t)a * s + (int64_t)b * c, PP_ONE));
    }
}

static int32_t pp_exp_negative(int32_t magnitude) {
    int index = magnitude / PP_EXP_INTERVAL;
    if (index >= PP_EXP_LAST) return 0;
    int fraction = magnitude % PP_EXP_INTERVAL;
    return (int32_t)pp_round_div((int64_t)pp_model.exponential[index] * (PP_EXP_INTERVAL - fraction)
                         + (int64_t)pp_model.exponential[index + 1] * fraction, PP_EXP_INTERVAL);
}

static void pp_attention(int32_t *out, const int32_t *q, int layer, int position) {
    for (int head = 0; head < PP_HEADS; head++) {
        int32_t weights[PP_SEQ], largest = -PP_LIMIT;
        int group = (head / (PP_HEADS / (PP_KV / PP_HEAD))) * PP_HEAD;
        for (int t = 0; t <= position; t++) {
            int64_t dot = 0;
            for (int i = 0; i < PP_HEAD; i++) {
                dot += (int64_t)q[head * PP_HEAD + i] * pp_keys[layer][t][group + i];
            }
            /* sqrt(8) in Q16 is 185364. The product is already Q32. */
            weights[t] = pp_clip(pp_round_div(dot, 185364));
            if (weights[t] > largest) largest = weights[t];
        }
        int32_t total = 0;
        for (int t = 0; t <= position; t++) {
            weights[t] = pp_exp_negative(largest - weights[t]);
            total += weights[t];
        }
        for (int i = 0; i < PP_HEAD; i++) {
            int64_t sum = 0;
            for (int t = 0; t <= position; t++) sum += (int64_t)weights[t] * pp_values[layer][t][group + i];
            out[head * PP_HEAD + i] = pp_clip(pp_round_div(sum, total));
        }
    }
}

static const int32_t *pp_forward(int token, int position) {
    int32_t x[PP_DIM], xb[PP_DIM], projected[PP_DIM], q[PP_DIM];
    int32_t gate[PP_HIDDEN], up[PP_HIDDEN];
    memcpy(x, pp_model.embedding + token * PP_DIM, sizeof(x));
#define PP_PROJECT(out, in, field, cols, rows) \
    pp_project(out, in, pp_model.field + layer * (rows) * (cols), \
               pp_model.field##_scale + layer * (rows), cols, rows)
    for (int layer = 0; layer < PP_LAYERS; layer++) {
        int32_t *k = pp_keys[layer][position], *v = pp_values[layer][position];
        pp_rms(xb, x, pp_model.rms_att + layer * PP_DIM);
        PP_PROJECT(q, xb, wq, PP_DIM, PP_DIM);
        PP_PROJECT(k, xb, wk, PP_DIM, PP_KV);
        PP_PROJECT(v, xb, wv, PP_DIM, PP_KV);
        pp_rotate(q, PP_DIM, position);
        pp_rotate(k, PP_KV, position);
        pp_attention(xb, q, layer, position);
        PP_PROJECT(projected, xb, wo, PP_DIM, PP_DIM);
        for (int i = 0; i < PP_DIM; i++) x[i] = pp_clip((int64_t)x[i] + projected[i]);

        pp_rms(xb, x, pp_model.rms_ffn + layer * PP_DIM);
        PP_PROJECT(gate, xb, w1, PP_DIM, PP_HIDDEN);
        PP_PROJECT(up, xb, w3, PP_DIM, PP_HIDDEN);
        for (int i = 0; i < PP_HIDDEN; i++) {
            int32_t e = pp_exp_negative(abs(gate[i]));
            int32_t numerator = gate[i] >= 0 ? PP_ONE : e;
            int64_t silu = pp_round_div((int64_t)gate[i] * numerator, PP_ONE + e);
            gate[i] = pp_clip(pp_round_div(silu * up[i], PP_ONE));
        }
        PP_PROJECT(xb, gate, w2, PP_HIDDEN, PP_DIM);
        for (int i = 0; i < PP_DIM; i++) x[i] = pp_clip((int64_t)x[i] + xb[i]);
    }
#undef PP_PROJECT
    pp_rms(xb, x, pp_model.rms_final);
    int peak = 1;
    for (int i = 0; i < PP_DIM; i++) if (abs(xb[i]) > peak) peak = abs(xb[i]);
    int8_t encoded[PP_DIM];
    fprintf(stderr, "input %d q16:%d", position, peak);
    for (int i = 0; i < PP_DIM; i++) {
        encoded[i] = (int8_t)pp_round_div((int64_t)xb[i] * 127, peak);
        fprintf(stderr, " %d", encoded[i]);
    }
    fprintf(stderr, "\nscores %d", position);
    static int32_t scores[PP_VOCAB];
    if (pp_score_fd >= 0) {
        transfer_scores(pp_score_fd, encoded, sizeof(encoded), 1);
        transfer_scores(pp_score_fd, scores, sizeof(scores), 0);
    } else {
        for (int row = 0; row < PP_VOCAB; row++) {
            int32_t sum = 0;
            for (int i = 0; i < PP_DIM; i++) sum += (int32_t)encoded[i] * pp_model.classifier[row * PP_DIM + i];
            scores[row] = sum;
        }
    }
    for (int row = 0; row < PP_VOCAB; row++) fprintf(stderr, " %d", scores[row]);
    fputc('\n', stderr);
    return scores;
}

static int pp_argmax(const int32_t *scores) {
    int best = 0;
    for (int i = 1; i < PP_VOCAB; i++) if (scores[i] > scores[best]) best = i;
    return best;
}
