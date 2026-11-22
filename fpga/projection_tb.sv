`timescale 1ns/1ps
module projection_tb;
    reg clk = 0;
    always #5 clk = ~clk;

    reg rst = 0, clear = 0, step = 0, decode = 0;
    reg [63:0] activations = 0;
    reg [511:0] weights = 0;
    wire [2047:0] sums;

    projection_tile dut (
        .clk(clk), .rst(rst), .clear(clear), .step(step), .decode(decode),
        .activations(activations), .weights(weights), .sums(sums)
    );

    // Commands drive one clock edge. Compare leaves the tile idle.
    integer file_handle, scans, operation, lane, line_number;
    reg [2047:0] expected, before_stall;
    initial begin
        file_handle = $fopen("projection.txt", "r");
        if (!file_handle) $fatal(1, "Could not open projection.txt");
        line_number = 0;
        scans = $fscanf(file_handle, "%d", operation);
        while (scans == 1 && operation != 6) begin
            line_number = line_number + 1;
            @(negedge clk);
            rst = 0;
            clear = 0;
            step = 0;
            case (operation)
                0: begin
                    rst = 1;
                    step = 1;
                end
                1: begin
                    if ($fscanf(file_handle, "%b", decode) != 1)
                        $fatal(1, "Missing mode at line %0d", line_number);
                    clear = 1;
                    step = 1;
                end
                2, 3: begin
                    if ($fscanf(file_handle, "%h %h", activations, weights) != 2)
                        $fatal(1, "Missing inputs at line %0d", line_number);
                    step = operation == 2;
                    before_stall = sums;
                end
                4: begin
                    step = 1;
                    activations = 0;
                    weights = 0;
                end
                5: begin
                    if ($fscanf(file_handle, "%h", expected) != 1)
                        $fatal(1, "Missing sums at line %0d", line_number);
                    if (sums !== expected) begin
                        for (lane = 0; lane < 64; lane = lane + 1)
                            if (sums[32*lane +: 32] !== expected[32*lane +: 32])
                                $fatal(1, "Line %0d, lane %0d: got %h, expected %h",
                                       line_number, lane, sums[32*lane +: 32], expected[32*lane +: 32]);
                    end
                end
                default: $fatal(1, "Unknown operation %0d at line %0d", operation, line_number);
            endcase
            @(posedge clk);
            #1;
            if (operation == 3 && sums !== before_stall)
                $fatal(1, "Sums changed while stalled at line %0d", line_number);
            scans = $fscanf(file_handle, "%d", operation);
        end
        if (scans != 1) $fatal(1, "Missing end marker");
        $fclose(file_handle);
        $finish;
    end
endmodule
