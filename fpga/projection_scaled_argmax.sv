// Experimental stream reducer for FLAN's signed scores and positive float32 scales.
// Each accepted input is one real vocabulary row. The caller supplies increasing
// indices and asserts in_last on the final row. Row and activation scales are
// IEEE binary32, and each multiplication rounds independently to nearest-even.
module projection_scaled_argmax (
    input  wire        clk,
    input  wire        rst,
    input  wire        in_valid,
    output wire        in_ready,
    input  wire        in_last,
    input  wire [31:0] in_index,
    input  wire signed [31:0] in_score,
    input  wire [31:0] in_activation_scale,
    input  wire [31:0] in_row_scale,
    output reg         out_valid,
    input  wire        out_ready,
    output reg  [31:0] out_index,
    output reg  [31:0] out_score,
    output reg         out_invalid
);
    function automatic invalid_scale(input [31:0] scale);
        invalid_scale = (scale[31] && |scale[30:0]) || scale[30:23] == 8'hff;
    endfunction

    function automatic greater(input [31:0] a, input [31:0] b);
        begin
            // +0 and -0 compare equal. For equal scores the earliest index wins.
            if (a[30:0] == 0 && b[30:0] == 0)
                greater = 1'b0;
            else if (a[31] != b[31])
                greater = !a[31];
            else if (a[31])
                greater = a[30:0] < b[30:0];
            else
                greater = a[30:0] > b[30:0];
        end
    endfunction

    // Each arithmetic unit has one operand/product register and six
    // normalization registers. Keep row metadata and scales synchronized.
    localparam UNIT_LATENCY = 7, SCORE_LATENCY = 3 * UNIT_LATENCY;
    reg [SCORE_LATENCY-1:0] valid_pipe, last_pipe, invalid_pipe;
    reg [31:0] index_pipe [0:SCORE_LATENCY-1];
    reg [31:0] activation_pipe [0:UNIT_LATENCY-1];
    reg [31:0] scale_pipe [0:2*UNIT_LATENCY-1];

    reg sign;
    reg [31:0] magnitude;
    wire [31:0] integer_float, activation_scaled, row_scaled;
    wire clock_enable = !out_valid || out_ready;
    assign in_ready = clock_enable;

    always @(posedge clk) if (clock_enable) begin
        sign <= in_score[31];
        magnitude <= in_score[31] ? 32'd0 - in_score : in_score;
    end

    projection_fp32_normalize int_convert (
        .clk(clk), .enable(clock_enable), .in_sign(sign),
        .in_magnitude({16'd0, magnitude}), .in_base(10'sd0), .in_special(2'd0),
        .out_float(integer_float)
    );
    projection_fp32_multiply activation_multiply (
        .clk(clk), .enable(clock_enable), .a(integer_float),
        .b(activation_pipe[UNIT_LATENCY-1]), .product(activation_scaled)
    );
    projection_fp32_multiply row_multiply (
        .clk(clk), .enable(clock_enable), .a(activation_scaled),
        .b(scale_pipe[2*UNIT_LATENCY-1]), .product(row_scaled)
    );

    reg [31:0] best_index, best_score;
    reg have_best, bad_frame;
    wire nan_score = row_scaled[30:23] == 8'hff && |row_scaled[22:0];
    wire invalid_row = invalid_pipe[SCORE_LATENCY-1] || nan_score;
    wire new_best = !invalid_row && (!have_best || greater(row_scaled, best_score));
    integer stage;

    // Stalling the output also freezes every pipeline stage.
    always @(posedge clk) begin
        if (rst) begin
            valid_pipe <= 0;
            have_best <= 0;
            bad_frame <= 0;
            out_valid <= 0;
        end else if (clock_enable) begin
            out_valid <= 0;
            valid_pipe <= {valid_pipe[SCORE_LATENCY-2:0], in_valid};
            last_pipe <= {last_pipe[SCORE_LATENCY-2:0], in_last};
            invalid_pipe <= {invalid_pipe[SCORE_LATENCY-2:0],
                             invalid_scale(in_activation_scale) || invalid_scale(in_row_scale)};
            index_pipe[0] <= in_index;
            activation_pipe[0] <= in_activation_scale;
            scale_pipe[0] <= in_row_scale;
            for (stage = 1; stage < SCORE_LATENCY; stage = stage + 1)
                index_pipe[stage] <= index_pipe[stage-1];
            for (stage = 1; stage < UNIT_LATENCY; stage = stage + 1)
                activation_pipe[stage] <= activation_pipe[stage-1];
            for (stage = 1; stage < 2*UNIT_LATENCY; stage = stage + 1)
                scale_pipe[stage] <= scale_pipe[stage-1];

            if (valid_pipe[SCORE_LATENCY-1]) begin
                if (last_pipe[SCORE_LATENCY-1]) begin
                    out_valid <= 1;
                    out_invalid <= bad_frame || invalid_row;
                    out_index <= new_best ? index_pipe[SCORE_LATENCY-1] : have_best ? best_index : 32'd0;
                    out_score <= new_best ? row_scaled : have_best ? best_score : 32'd0;
                    have_best <= 0;
                    bad_frame <= 0;
                end else begin
                    bad_frame <= bad_frame || invalid_row;
                    if (new_best) begin
                        have_best <= 1;
                        best_index <= index_pipe[SCORE_LATENCY-1];
                        best_score <= row_scaled;
                    end
                end
            end
        end
    end
endmodule

module projection_fp32_multiply (
    input  wire        clk,
    input  wire        enable,
    input  wire [31:0] a,
    input  wire [31:0] b,
    output wire [31:0] product
);
    wire [23:0] a_mantissa = {|a[30:23], a[22:0]};
    wire [23:0] b_mantissa = {|b[30:23], b[22:0]};
    wire signed [9:0] a_base = a[30:23] == 0 ? -10'sd149 : $signed({2'd0, a[30:23]}) - 10'sd150;
    wire signed [9:0] b_base = b[30:23] == 0 ? -10'sd149 : $signed({2'd0, b[30:23]}) - 10'sd150;
    reg [47:0] magnitude;
    reg signed [9:0] base;
    reg sign;
    reg [1:0] special;
    always @(posedge clk) if (enable) begin
        magnitude <= a_mantissa * b_mantissa;
        base <= a_base + b_base;
        sign <= a[31] ^ b[31];
        special <= a[30:23] != 8'hff ? 2'd0 : b[30:0] == 0 ? 2'd2 : 2'd1;
    end

    projection_fp32_normalize normalize (
        .clk(clk), .enable(enable), .in_sign(sign),
        .in_magnitude(magnitude), .in_base(base), .in_special(special),
        .out_float(product)
    );
endmodule

// Convert a 48-bit unsigned magnitude with the unit 2^in_base to binary32.
// The carries, priority encoding and variable shifts sit in separate stages.
module projection_fp32_normalize (
    input  wire              clk,
    input  wire              enable,
    input  wire              in_sign,
    input  wire [47:0]       in_magnitude,
    input  wire signed [9:0] in_base,
    input  wire [1:0]        in_special,
    output reg  [31:0]       out_float
);
    reg [47:0] magnitude [0:2];
    reg signed [9:0] base [0:1];
    reg [4:0] signs;
    reg [4:1] zeros;
    reg [1:0] special [0:4];

    // First encode the leading position within each group of eight bits.
    wire [5:0] group_nonzero;
    reg [2:0] group_top [0:5];
    for (genvar group = 0; group < 6; group = group + 1) begin : leading_groups
        integer bit_index;
        reg [2:0] leading;
        wire [7:0] bits = in_magnitude[8*group +: 8];
        always @* begin
            leading = 0;
            for (bit_index = 0; bit_index < 8; bit_index = bit_index + 1)
                if (bits[bit_index])
                    leading = bit_index[2:0];
        end
        assign group_nonzero[group] = |bits;
        always @(posedge clk) if (enable)
            group_top[group] <= leading;
    end
    reg [5:0] group_live;
    reg [5:0] top, leading_top;
    integer group_index;
    always @* begin
        leading_top = 0;
        for (group_index = 0; group_index < 6; group_index = group_index + 1)
            if (group_live[group_index])
                leading_top = {group_index[2:0], group_top[group_index]};
    end

    // Choose the exponent and amount of precision to discard. In the
    // subnormal range, the target least-significant bit is fixed at 2^-149.
    wire signed [9:0] leading_exponent = base[1] + $signed({4'd0, top});
    reg signed [9:0] exponent_3, exponent_4, exponent_5, discard;
    reg [23:0] truncated;
    reg guard_bit, sticky;
    wire [5:0] right_shift = discard[5:0];
    wire [5:0] left_shift = 6'(-discard);
    wire [47:0] sticky_mask = 48'hffffffffffff >> (6'd49 - right_shift);
    wire [47:0] unrounded = discard <= 0 ? magnitude[2] << left_shift : magnitude[2] >> right_shift;
    reg [24:0] rounded;

    wire signed [9:0] final_exponent = exponent_5 + $signed({9'd0, rounded[24]});
    wire [7:0] biased = 8'(final_exponent + 10'sd127);
    integer stage;
    always @(posedge clk) if (enable) begin
        group_live <= group_nonzero;
        top <= leading_top;
        signs <= {signs[3:0], in_sign};
        zeros <= {zeros[3:1], group_live == 0};
        base[0] <= in_base;
        base[1] <= base[0];
        magnitude[0] <= in_magnitude;
        magnitude[1] <= magnitude[0];
        magnitude[2] <= magnitude[1];
        special[0] <= in_special;
        for (stage = 1; stage < 5; stage = stage + 1)
            special[stage] <= special[stage-1];

        exponent_3 <= leading_exponent;
        discard <= leading_exponent < -10'sd126 ? -10'sd149 - base[1] : $signed({4'd0, top}) - 10'sd23;

        exponent_4 <= exponent_3;
        truncated <= discard > 10'sd48 ? 24'd0 : unrounded[23:0];
        guard_bit <= discard > 0 && discard <= 10'sd48 && magnitude[2][right_shift - 6'd1];
        sticky <= |(magnitude[2] & sticky_mask);

        exponent_5 <= exponent_4;
        rounded <= {1'b0, truncated} + {24'd0, guard_bit && (truncated[0] || sticky)};

        if (special[4] == 2'd2)
            out_float <= 32'h7fc00000;
        else if (special[4] == 2'd1)
            out_float <= {signs[4], 8'hff, 23'd0};
        else if (zeros[4])
            out_float <= {signs[4], 31'd0};
        else if (exponent_5 < -10'sd126)
            // Rounding the largest subnormal can produce the smallest normal.
            out_float <= {signs[4], 7'd0, rounded[23:0]};
        else if (final_exponent > 10'sd127)
            out_float <= {signs[4], 8'hff, 23'd0};
        else
            out_float <= {signs[4], biased, rounded[24] ? rounded[23:1] : rounded[22:0]};
    end
endmodule
