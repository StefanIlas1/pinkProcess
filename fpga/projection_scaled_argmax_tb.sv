`timescale 1ns/1ps
module projection_scaled_argmax_tb;
    reg clk = 0;
    always #5 clk = ~clk;
    reg rst = 1;
    reg in_valid = 0, in_last = 0, out_ready = 0;
    reg [31:0] in_index, in_score, in_activation_scale, in_row_scale;
    wire in_ready, out_valid, out_invalid;
    wire [31:0] out_index, out_score;
    projection_scaled_argmax dut (.*);

    reg [64:0] expected [0:16383];
    integer sent = 0, consumed = 0, cycles = 0;
    integer vectors, fields;
    reg expected_invalid;
    reg [31:0] expected_index, expected_score;
    reg prior_stall = 0;
    reg [64:0] prior_result;

    always @(posedge clk) begin
        if (!rst) begin
            cycles <= cycles + 1;
            if (cycles > 60000)
                $fatal(1, "Timed out.");
            if (prior_stall && (!out_valid ||
                {out_invalid, out_index, out_score} !== prior_result))
                $fatal(1, "Result changed while output was stalled.");
            prior_stall <= out_valid && !out_ready;
            prior_result <= {out_invalid, out_index, out_score};
            if (out_valid && out_ready) begin
                if (consumed >= sent ||
                    {out_invalid, out_index, out_score} !== expected[consumed])
                    $fatal(1, "Result %0d got %h expected %h", consumed,
                           {out_invalid, out_index, out_score}, expected[consumed]);
                consumed <= consumed + 1;
            end
        end
    end

    always @(negedge clk)
        out_ready <= cycles % 9 != 2 && cycles % 9 != 3 && cycles % 9 != 4;

    initial begin
        vectors = $fopen("argmax.txt", "r");
        if (!vectors)
            $fatal(1, "Cannot open argmax.txt.");
        repeat (2) @(posedge clk);
        @(negedge clk);
        rst = 0;
        fields = $fscanf(vectors, "%h %h %h %h %b %b %h %h",
                         in_index, in_score, in_activation_scale, in_row_scale,
                         in_last, expected_invalid, expected_index, expected_score);
        while (fields == 8) begin
            if (in_last) begin
                expected[sent] = {expected_invalid, expected_index, expected_score};
                sent = sent + 1;
            end
            in_valid = 1;
            @(posedge clk);
            while (!in_ready) @(posedge clk);
            @(negedge clk);
            if (cycles % 7 == 0) begin
                in_valid = 0;
                @(negedge clk);
            end
            fields = $fscanf(vectors, "%h %h %h %h %b %b %h %h",
                             in_index, in_score, in_activation_scale, in_row_scale,
                             in_last, expected_invalid, expected_index, expected_score);
        end
        if (!$feof(vectors))
            $fatal(1, "Malformed argmax vector.");
        in_valid = 0;
        $fclose(vectors);
        wait (consumed == sent);
        $display("Verified %d signed FP32 argmax frames.", consumed);
        $finish;
    end
endmodule
