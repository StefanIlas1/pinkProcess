/* Included after matmul() in the pinned llama2.c source. */
#include <stdint.h>
#include <errno.h>

static void transfer_scores(int fd, void *buffer, size_t bytes, int send) {
    unsigned char *cursor = buffer;
    while (bytes > 0) {
        ssize_t count = send ? write(fd, cursor, bytes) : read(fd, cursor, bytes);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            fprintf(stderr, "Classifier connection closed during score transfer\n");
            exit(EXIT_FAILURE);
        }
        cursor += count;
        bytes -= (size_t)count;
    }
}

static float int8_scale(const float *values, size_t count) {
    float peak = 0.0f;
    for (size_t i = 0; i < count; i++) {
        peak = fmaxf(peak, fabsf(values[i]));
    }
    return peak == 0.0f ? 1.0f : peak / 127.0f;
}

static void quantize_int8(const float *values, int8_t *encoded, size_t count, float scale) {
    for (size_t i = 0; i < count; i++) {
        float rounded = roundf(values[i] / scale);
        encoded[i] = (int8_t)fmaxf(-127.0f, fminf(127.0f, rounded));
    }
}

static void load_replacement_row(int8_t *weights, int width, int rows) {
    const char *path = getenv("PINKPROCESS_REFERENCE_ROW");
    if (!path) {
        return;
    }
    FILE *file = fopen(path, "rb");
    unsigned char index[2];
    if (!file || fread(index, 1, 2, file) != 2) {
        fprintf(stderr, "Cannot read the replacement row header\n");
        exit(EXIT_FAILURE);
    }
    unsigned int row = index[0] | ((unsigned int)index[1] << 8);
    if (row >= (unsigned int)rows ||
        fread(weights + (size_t)row * width, 1, (size_t)width, file) != (size_t)width ||
        fgetc(file) != EOF || ferror(file)) {
        fprintf(stderr, "Replacement row index or size is invalid\n");
        exit(EXIT_FAILURE);
    }
    fclose(file);
    for (int column = 0; column < width; column++) {
        if (weights[(size_t)row * width + column] == INT8_MIN) {
            fprintf(stderr, "Replacement row must use the symmetric INT8 range\n");
            exit(EXIT_FAILURE);
        }
    }
}

static void pinkprocess_classifier(float *logits, float *x, float *weights,
                                   int width, int rows, int position) {
    /* The CLI loads one model per process. Own the classifier separately
       because this checkpoint shares its source weights with embeddings. */
    static float *output_weights;
    static int8_t *encoded_weights;
    static int8_t *encoded_input;
    static float weight_scale;
    static int integer_mode;
    static int score_fd = -1;
    if (output_weights == NULL) {
        size_t count = (size_t)rows * width;
        output_weights = malloc(count * sizeof(*output_weights));
        encoded_weights = malloc(count * sizeof(*encoded_weights));
        encoded_input = malloc((size_t)width * sizeof(*encoded_input));
        if (!output_weights || !encoded_weights || !encoded_input) {
            fprintf(stderr, "Cannot allocate output classifier buffers\n");
            exit(EXIT_FAILURE);
        }
        memcpy(output_weights, weights, count * sizeof(*output_weights));

        const char *mode = getenv("PINKPROCESS_REFERENCE_MODE");
        integer_mode = mode && strcmp(mode, "int8") == 0;
        if (integer_mode) {
            const char *fd = getenv("PINKPROCESS_SCORE_FD");
            if (fd) {
                score_fd = atoi(fd);
                if (width != 64 || rows != 512 || score_fd < 3) {
                    fprintf(stderr, "Invalid live classifier dimensions or descriptor\n");
                    exit(EXIT_FAILURE);
                }
            }
            weight_scale = int8_scale(output_weights, count);
            quantize_int8(output_weights, encoded_weights, count, weight_scale);
            load_replacement_row(encoded_weights, width, rows);
            fprintf(stderr, "weights %a", (double)weight_scale);
            for (size_t i = 0; i < count; i++) {
                fprintf(stderr, " %d", encoded_weights[i]);
            }
            fputc('\n', stderr);
        }
    }

    if (!integer_mode) {
        matmul(logits, x, output_weights, width, rows);
        return;
    }

    float input_scale = int8_scale(x, width);
    quantize_int8(x, encoded_input, (size_t)width, input_scale);
    fprintf(stderr, "input %d %a", position, (double)input_scale);
    for (int i = 0; i < width; i++) {
        fprintf(stderr, " %d", encoded_input[i]);
    }
    fprintf(stderr, "\nscores %d", position);
    int32_t live_scores[512];
    if (score_fd >= 0) {
        transfer_scores(score_fd, encoded_input, (size_t)width, 1);
        transfer_scores(score_fd, live_scores, sizeof(live_scores), 0);
    }
    for (int row = 0; row < rows; row++) {
        int32_t sum = 0;
        if (score_fd >= 0) {
            sum = live_scores[row];
        } else {
            for (int column = 0; column < width; column++) {
                sum += (int32_t)encoded_weights[(size_t)row * width + column] * encoded_input[column];
            }
        }
        logits[row] = (float)sum * (weight_scale * input_scale);
        fprintf(stderr, " %d", (int)sum);
    }
    fputc('\n', stderr);
}
