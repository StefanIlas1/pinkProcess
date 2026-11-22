// Signed INT8 projection tile with 64 shared multiply-accumulate lanes.
module projection_tile (
    input  wire          clk,
    input  wire          rst,
    input  wire          clear,
    input  wire          step,
    input  wire          decode,
    input  wire [63:0]   activations,
    input  wire [511:0]  weights,
    output wire [2047:0] sums
);
    wire signed [7:0] x_path [0:7][0:8];
    wire signed [7:0] w_path [0:8][0:7];

    // Skew edge lane i by i steps so matching terms reach cell (r,c)
    // together at step r+c after entering the tile.
    for (genvar lane = 0; lane < 8; lane = lane + 1) begin : edge_lane
        if (lane == 0) begin : direct
            assign x_path[lane][0] = activations[8*lane +: 8];
            assign w_path[0][lane] = weights[8*lane +: 8];
        end else begin : skewed
            reg [8*lane-1:0] x_delay;
            reg [8*lane-1:0] w_delay;
            always @(posedge clk) begin
                if (rst || clear) begin
                    x_delay <= '0;
                    w_delay <= '0;
                end else if (step && !decode) begin
                    x_delay <= (x_delay << 8) | {{(8*lane-8){1'b0}}, activations[8*lane +: 8]};
                    w_delay <= (w_delay << 8) | {{(8*lane-8){1'b0}}, weights[8*lane +: 8]};
                end
            end
            assign x_path[lane][0] = x_delay[8*(lane-1) +: 8];
            assign w_path[0][lane] = w_delay[8*(lane-1) +: 8];
        end
    end

    for (genvar row = 0; row < 8; row = row + 1) begin : tile_row
        for (genvar col = 0; col < 8; col = col + 1) begin : tile_col
            localparam integer INDEX = 8*row + col;
            reg signed [7:0] x_next;
            reg signed [7:0] w_next;
            reg signed [31:0] accumulator;

            // Decode reuses the encoder multipliers with one shared activation
            // and one independent output weight per cell.
            wire signed [7:0] x = decode ? activations[7:0] : x_path[row][col];
            wire signed [7:0] w = decode ? weights[8*INDEX +: 8] : w_path[row][col];
            wire signed [15:0] product = x * w;

            always @(posedge clk) begin
                if (rst || clear) begin
                    x_next <= '0;
                    w_next <= '0;
                    accumulator <= '0;
                end else if (step) begin
                    if (!decode) begin
                        x_next <= x_path[row][col];
                        w_next <= w_path[row][col];
                    end
                    accumulator <= accumulator + product;
                end
            end

            assign x_path[row][col+1] = x_next;
            assign w_path[row+1][col] = w_next;
            assign sums[32*INDEX +: 32] = accumulator;
        end
    end
endmodule
