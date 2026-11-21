// AXI4-Lite slave for the 512 x 64 TinyStories INT8 output classifier.
module pinkprocess_classifier (
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF S_AXI, ASSOCIATED_RESET s_axi_aresetn" *)
    input wire s_axi_aclk,
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input wire s_axi_aresetn,
    input wire [15:0] s_axi_awaddr,
    input wire [2:0] s_axi_awprot,
    input wire s_axi_awvalid,
    output wire s_axi_awready,
    input wire [31:0] s_axi_wdata,
    input wire [3:0] s_axi_wstrb,
    input wire s_axi_wvalid,
    output wire s_axi_wready,
    output reg [1:0] s_axi_bresp,
    output reg s_axi_bvalid,
    input wire s_axi_bready,
    input wire [15:0] s_axi_araddr,
    input wire [2:0] s_axi_arprot,
    input wire s_axi_arvalid,
    output wire s_axi_arready,
    output reg [31:0] s_axi_rdata,
    output reg [1:0] s_axi_rresp,
    output reg s_axi_rvalid,
    input wire s_axi_rready
);
    localparam [15:0] INPUT_BASE = 16'h8000, SCORE_BASE = 16'h8040;
    localparam [15:0] STATUS = 16'h8840, COMMAND = 16'h8844;
    localparam [1:0] IDLE = 0, FETCH = 1, ACCUMULATE = 2;

    (* ram_style = "block" *) reg [31:0] weights [0:8191];
    (* ram_style = "block" *) reg signed [31:0] scores [0:511];
    reg [31:0] activations [0:15];
    reg [31:0] weight_word;
    reg signed [31:0] accumulator;
    reg [8:0] row;
    reg [3:0] group_index;
    reg [1:0] state;
    reg done;

    reg aw_pending, w_pending;
    reg [15:0] write_address;
    reg [31:0] write_data;
    reg [3:0] write_strobes;
    wire complete_write = aw_pending && w_pending && !s_axi_bvalid;
    wire busy = state != IDLE;
    assign s_axi_awready = !aw_pending && !s_axi_bvalid;
    assign s_axi_wready = !w_pending && !s_axi_bvalid;
    assign s_axi_arready = !s_axi_rvalid;

    function signed [31:0] dot4;
        input [31:0] w;
        input [31:0] x;
        reg signed [15:0] p0, p1, p2, p3;
        begin
            p0 = $signed(w[7:0]) * $signed(x[7:0]);
            p1 = $signed(w[15:8]) * $signed(x[15:8]);
            p2 = $signed(w[23:16]) * $signed(x[23:16]);
            p3 = $signed(w[31:24]) * $signed(x[31:24]);
            dot4 = p0 + p1 + p2 + p3;
        end
    endfunction

    wire signed [31:0] updated_sum = accumulator + dot4(weight_word, activations[group_index]);
    integer byte_index;
    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            aw_pending <= 0;
            w_pending <= 0;
            s_axi_rvalid <= 0;
            s_axi_bvalid <= 0;
            state <= IDLE;
            done <= 0;
        end else begin
            if (s_axi_awvalid && s_axi_awready) begin
                write_address <= s_axi_awaddr;
                aw_pending <= 1;
            end
            if (s_axi_wvalid && s_axi_wready) begin
                write_data <= s_axi_wdata;
                write_strobes <= s_axi_wstrb;
                w_pending <= 1;
            end
            if (s_axi_bvalid && s_axi_bready)
                s_axi_bvalid <= 0;
            if (complete_write) begin
                aw_pending <= 0;
                w_pending <= 0;
                s_axi_bvalid <= 1;
                s_axi_bresp <= 0;
                if (busy || write_address[1:0] != 0) begin
                    s_axi_bresp <= 2'b10;
                end else if (write_address < INPUT_BASE) begin
                    for (byte_index = 0; byte_index < 4; byte_index = byte_index + 1)
                        if (write_strobes[byte_index])
                            weights[write_address[14:2]][8*byte_index +: 8] <= write_data[8*byte_index +: 8];
                end else if (write_address < SCORE_BASE) begin
                    for (byte_index = 0; byte_index < 4; byte_index = byte_index + 1)
                        if (write_strobes[byte_index])
                            activations[write_address[5:2]][8*byte_index +: 8] <= write_data[8*byte_index +: 8];
                end else if (write_address == COMMAND) begin
                    if (write_strobes[0] && write_data[0]) begin
                        row <= 0;
                        group_index <= 0;
                        accumulator <= 0;
                        done <= 0;
                        state <= FETCH;
                    end
                end else begin
                    s_axi_bresp <= 2'b10;
                end
            end

            if (s_axi_rvalid && s_axi_rready)
                s_axi_rvalid <= 0;
            if (s_axi_arvalid && s_axi_arready) begin
                s_axi_rvalid <= 1;
                s_axi_rresp <= 0;
                s_axi_rdata <= 0;
                if (s_axi_araddr == STATUS) begin
                    s_axi_rdata <= {30'b0, done, busy};
                end else if (s_axi_araddr[1:0] == 0 && s_axi_araddr >= SCORE_BASE &&
                             s_axi_araddr < STATUS && done) begin
                    s_axi_rdata <= scores[(s_axi_araddr - SCORE_BASE) >> 2];
                end else begin
                    s_axi_rresp <= 2'b10;
                end
            end

            case (state)
                FETCH: begin
                    weight_word <= weights[{row, group_index}];
                    state <= ACCUMULATE;
                end
                ACCUMULATE: begin
                    if (group_index == 15) begin
                        scores[row] <= updated_sum;
                        group_index <= 0;
                        accumulator <= 0;
                        if (row == 511) begin
                            done <= 1;
                            state <= IDLE;
                        end else begin
                            row <= row + 1'b1;
                            state <= FETCH;
                        end
                    end else begin
                        group_index <= group_index + 1'b1;
                        accumulator <= updated_sum;
                        state <= FETCH;
                    end
                end
                default: begin end
            endcase
        end
    end
endmodule
