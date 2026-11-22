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
    // The magnitude is an unsigned integer with least-significant bit 2^base.
    function automatic [31:0] rounded_float(
        input sign, input [47:0] magnitude, input integer base
    );
        integer top, exponent, discard, bit_index;
        reg [48:0] rounded;
        reg [47:0] sticky_mask;
        reg [7:0] biased;
        begin
            top = 0;
            for (bit_index = 0; bit_index < 48; bit_index = bit_index + 1)
                if (magnitude[bit_index])
                    top = bit_index;
            exponent = top + base;
            discard = (exponent < -126) ? -149 - base : top - 23;
            if (discard <= 0)
                rounded = {1'b0, magnitude} << -discard;
            else
                rounded = {1'b0, magnitude} >> discard;
            if (discard > 0 && discard <= 48) begin
                sticky_mask = 48'hffffffffffff >> (49 - discard);
                if (magnitude[discard-1] && (rounded[0] || |(magnitude & sticky_mask)))
                    rounded = rounded + 49'd1;
            end
            if (magnitude == 0)
                rounded_float = {sign, 31'b0};
            else if (exponent < -126)
                // A rounded subnormal can carry into the smallest normal value.
                rounded_float = {sign, 7'd0, rounded[23:0]};
            else begin
                if (rounded[24]) begin
                    rounded = rounded >> 1;
                    exponent = exponent + 1;
                end
                biased = 8'(exponent + 127);
                rounded_float = exponent > 127 ? {sign, 8'hff, 23'd0} :
                                                  {sign, biased, rounded[22:0]};
            end
        end
    endfunction

    function automatic [31:0] int_to_float(input [31:0] value);
        reg [31:0] magnitude;
        begin
            magnitude = value[31] ? 32'd0 - value : value;
            int_to_float = rounded_float(value[31], {16'd0, magnitude}, 0);
        end
    endfunction

    function automatic [31:0] multiply_float(input [31:0] a, input [31:0] b);
        reg [23:0] a_mantissa, b_mantissa;
        integer a_base, b_base;
        begin
            a_mantissa = {|a[30:23], a[22:0]};
            b_mantissa = {|b[30:23], b[22:0]};
            a_base = a[30:23] == 0 ? -149 : int'(a[30:23]) - 150;
            b_base = b[30:23] == 0 ? -149 : int'(b[30:23]) - 150;
            if (a[30:23] == 8'hff)
                multiply_float = b[30:0] == 0 ? 32'h7fc00000 : a;
            else
                multiply_float = rounded_float(a[31] ^ b[31], a_mantissa * b_mantissa,
                                               a_base + b_base);
        end
    endfunction

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

    reg [2:0] valid_pipe, last_pipe, invalid_pipe;
    reg [31:0] index_pipe [0:2];
    reg [31:0] activation_scale, first_row_scale, second_row_scale;
    reg [31:0] integer_float, activation_scaled, row_scaled;
    reg [31:0] best_index, best_score;
    reg have_best, bad_frame;
    wire nan_score = row_scaled[30:23] == 8'hff && |row_scaled[22:0];
    wire invalid_row = invalid_pipe[2] || nan_score;
    wire new_best = !invalid_row && (!have_best || greater(row_scaled, best_score));

    // Stalling the output also freezes every pipeline stage.
    assign in_ready = !out_valid || out_ready;
    always @(posedge clk) begin
        if (rst) begin
            valid_pipe <= 0;
            have_best <= 0;
            bad_frame <= 0;
            out_valid <= 0;
        end else if (in_ready) begin
            out_valid <= 0;
            valid_pipe <= {valid_pipe[1:0], in_valid};
            if (in_valid) begin
                last_pipe[0] <= in_last;
                index_pipe[0] <= in_index;
                invalid_pipe[0] <= invalid_scale(in_activation_scale) || invalid_scale(in_row_scale);
                integer_float <= int_to_float(in_score);
                activation_scale <= in_activation_scale;
                first_row_scale <= in_row_scale;
            end
            if (valid_pipe[0]) begin
                last_pipe[1] <= last_pipe[0];
                index_pipe[1] <= index_pipe[0];
                invalid_pipe[1] <= invalid_pipe[0];
                activation_scaled <= multiply_float(integer_float, activation_scale);
                second_row_scale <= first_row_scale;
            end
            if (valid_pipe[1]) begin
                last_pipe[2] <= last_pipe[1];
                index_pipe[2] <= index_pipe[1];
                invalid_pipe[2] <= invalid_pipe[1];
                row_scaled <= multiply_float(activation_scaled, second_row_scale);
            end
            if (valid_pipe[2]) begin
                if (last_pipe[2]) begin
                    out_valid <= 1;
                    out_invalid <= bad_frame || invalid_row;
                    out_index <= new_best ? index_pipe[2] : have_best ? best_index : 32'd0;
                    out_score <= new_best ? row_scaled : have_best ? best_score : 32'd0;
                    have_best <= 0;
                    bad_frame <= 0;
                end else begin
                    bad_frame <= bad_frame || invalid_row;
                    if (new_best) begin
                        have_best <= 1;
                        best_index <= index_pipe[2];
                        best_score <= row_scaled;
                    end
                end
            end
        end
    end
endmodule
