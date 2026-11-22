// Optional read-only AXI DMA front end for decoder vocabulary reduction.
module pinkprocess_argmax (
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF S_AXI:M_AXI, ASSOCIATED_RESET s_axi_aresetn" *)
    input wire s_axi_aclk, input wire s_axi_aresetn,
    input wire [11:0] s_axi_awaddr, input wire [2:0] s_axi_awprot,
    input wire s_axi_awvalid, output wire s_axi_awready,
    input wire [31:0] s_axi_wdata, input wire [3:0] s_axi_wstrb,
    input wire s_axi_wvalid, output wire s_axi_wready,
    output reg [1:0] s_axi_bresp, output reg s_axi_bvalid, input wire s_axi_bready,
    input wire [11:0] s_axi_araddr, input wire [2:0] s_axi_arprot,
    input wire s_axi_arvalid, output wire s_axi_arready,
    output reg [31:0] s_axi_rdata, output reg [1:0] s_axi_rresp,
    output reg s_axi_rvalid, input wire s_axi_rready,

    output wire [31:0] m_axi_araddr, output wire [7:0] m_axi_arlen,
    output wire [2:0] m_axi_arsize, output wire [1:0] m_axi_arburst,
    output wire [3:0] m_axi_arcache, output wire [2:0] m_axi_arprot,
    output wire m_axi_arlock, output wire m_axi_arvalid, input wire m_axi_arready,
    input wire [63:0] m_axi_rdata, input wire [1:0] m_axi_rresp,
    input wire m_axi_rlast, input wire m_axi_rvalid, output wire m_axi_rready
);
    wire clk = s_axi_aclk;
    wire rst = !s_axi_aresetn;
    assign m_axi_arsize = 3;
    assign m_axi_arburst = 1;
    assign m_axi_arcache = 3;
    assign m_axi_arprot = 0;
    assign m_axi_arlock = 0;

    localparam [2:0] IDLE = 0, SCORE_CMD = 1, SCORE_DATA = 2,
                     SCALE_CMD = 3, SCALE_DATA = 4, FEED = 5, RESULT = 6;
    reg [2:0] state;
    wire busy = state != IDLE;
    reg complete, failed;
    reg [31:0] score_addr, scale_addr, activation_scale;
    reg [15:0] rows;
    reg [31:0] read_score_addr, read_scale_addr;
    reg [15:0] remaining, index;
    reg [31:0] result_index, result_score;

    reg aw_pending, w_pending;
    reg [11:0] write_addr;
    reg [31:0] write_value;
    reg [3:0] write_strobes;
    assign s_axi_awready = !aw_pending && !s_axi_bvalid;
    assign s_axi_wready = !w_pending && !s_axi_bvalid;
    assign s_axi_arready = !s_axi_rvalid;
    wire write_event = aw_pending && w_pending;
    wire launch = write_event && write_addr == 0 && write_strobes[0] && write_value[0] && !busy;

    wire valid_config = rows != 0 && (score_addr[2:0] | scale_addr[2:0]) == 0;
    wire read_command = state == SCORE_CMD || state == SCALE_CMD;
    wire read_ready = state == SCORE_DATA || state == SCALE_DATA;
    wire command_ready, data_valid, dma_error;
    wire [63:0] data;
    wire accept_word = read_ready && data_valid;
    wire [16:0] rounded_remaining = {1'b0, remaining} + 17'd1;
    wire [4:0] beats = remaining > 32 ? 5'd16 : rounded_remaining[5:1];
    reg [3:0] word_index;
    reg [4:0] row_in_chunk;
    reg [63:0] scores [0:15];
    reg [63:0] scales [0:15];
    wire [63:0] score_word = scores[row_in_chunk[4:1]];
    wire [63:0] scale_word = scales[row_in_chunk[4:1]];
    wire [31:0] row_score = row_in_chunk[0] ? score_word[63:32] : score_word[31:0];
    wire [31:0] row_scale = row_in_chunk[0] ? scale_word[63:32] : scale_word[31:0];
    wire reducer_ready, result_valid, result_invalid;
    wire [31:0] winner_index, winner_score;

    projection_dma dma (
        .clk(clk), .rst(rst), .clear_error(launch), .error(dma_error),
        .rd_cmd_valid(read_command), .rd_cmd_ready(command_ready),
        .rd_cmd_addr(state == SCORE_CMD ? read_score_addr : read_scale_addr),
        .rd_cmd_beats({4'd0, beats}), .rd_data_valid(data_valid),
        .rd_data_ready(read_ready), .rd_data(data),
        .wr_cmd_valid(1'b0), .wr_cmd_addr(32'd0), .wr_cmd_beats(9'd0),
        .wr_data_valid(1'b0), .wr_data(64'd0), .wr_data_last(1'b0),
        .m_axi_araddr(m_axi_araddr), .m_axi_arlen(m_axi_arlen),
        .m_axi_arvalid(m_axi_arvalid), .m_axi_arready(m_axi_arready),
        .m_axi_rdata(m_axi_rdata), .m_axi_rresp(m_axi_rresp),
        .m_axi_rlast(m_axi_rlast), .m_axi_rvalid(m_axi_rvalid), .m_axi_rready(m_axi_rready),
        .m_axi_awready(1'b0), .m_axi_wready(1'b0), .m_axi_bresp(2'd0), .m_axi_bvalid(1'b0)
    );
    projection_scaled_argmax reducer (
        .clk(clk), .rst(rst),
        .in_valid(state == FEED), .in_ready(reducer_ready), .in_last(remaining == 1),
        .in_index({16'd0, index}), .in_score(row_score),
        .in_activation_scale(activation_scale), .in_row_scale(row_scale),
        .out_valid(result_valid), .out_ready(state == RESULT),
        .out_index(winner_index), .out_score(winner_score), .out_invalid(result_invalid)
    );

    integer byte_lane;
    always @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            complete <= 0;
            failed <= 0;
            aw_pending <= 0;
            w_pending <= 0;
            s_axi_bvalid <= 0;
            s_axi_rvalid <= 0;
            score_addr <= 0;
            scale_addr <= 0;
            activation_scale <= 0;
            rows <= 0;
            result_index <= 0;
            result_score <= 0;
        end else begin
            if (s_axi_awvalid && s_axi_awready) begin
                aw_pending <= 1;
                write_addr <= s_axi_awaddr;
            end
            if (s_axi_wvalid && s_axi_wready) begin
                w_pending <= 1;
                write_value <= s_axi_wdata;
                write_strobes <= s_axi_wstrb;
            end
            if (s_axi_bvalid && s_axi_bready)
                s_axi_bvalid <= 0;
            if (write_event) begin
                aw_pending <= 0;
                w_pending <= 0;
                s_axi_bvalid <= 1;
                s_axi_bresp <= 0;
                if (!busy) begin
                    for (byte_lane = 0; byte_lane < 4; byte_lane = byte_lane + 1) begin
                        if (write_strobes[byte_lane]) begin
                            case (write_addr)
                                12'h004: score_addr[8*byte_lane +: 8] <= write_value[8*byte_lane +: 8];
                                12'h008: scale_addr[8*byte_lane +: 8] <= write_value[8*byte_lane +: 8];
                                12'h00c: if (byte_lane < 2)
                                    rows[8*byte_lane +: 8] <= write_value[8*byte_lane +: 8];
                                12'h010: activation_scale[8*byte_lane +: 8] <= write_value[8*byte_lane +: 8];
                                default: begin end
                            endcase
                        end
                    end
                end
            end
            if (s_axi_rvalid && s_axi_rready)
                s_axi_rvalid <= 0;
            if (s_axi_arvalid && s_axi_arready) begin
                s_axi_rvalid <= 1;
                s_axi_rresp <= 0;
                case (s_axi_araddr)
                    12'h000: s_axi_rdata <= {29'd0, failed, complete, busy};
                    12'h004: s_axi_rdata <= score_addr;
                    12'h008: s_axi_rdata <= scale_addr;
                    12'h00c: s_axi_rdata <= {16'd0, rows};
                    12'h010: s_axi_rdata <= activation_scale;
                    12'h014: s_axi_rdata <= result_index;
                    12'h018: s_axi_rdata <= result_score;
                    12'h01c: s_axi_rdata <= 32'h50415231;
                    default: s_axi_rdata <= 0;
                endcase
            end

            case (state)
                IDLE: if (launch) begin
                    complete <= !valid_config;
                    failed <= !valid_config;
                    if (valid_config) begin
                        read_score_addr <= score_addr;
                        read_scale_addr <= scale_addr;
                        remaining <= rows;
                        index <= 0;
                        state <= SCORE_CMD;
                    end
                end
                SCORE_CMD: if (command_ready) begin
                    word_index <= 0;
                    state <= SCORE_DATA;
                end
                SCORE_DATA: if (accept_word) begin
                    scores[word_index] <= data;
                    word_index <= word_index + 1'b1;
                    if ({1'b0, word_index} == beats - 1'b1)
                        state <= SCALE_CMD;
                end
                SCALE_CMD: if (command_ready) begin
                    word_index <= 0;
                    state <= SCALE_DATA;
                end
                SCALE_DATA: if (accept_word) begin
                    scales[word_index] <= data;
                    word_index <= word_index + 1'b1;
                    if ({1'b0, word_index} == beats - 1'b1) begin
                        row_in_chunk <= 0;
                        state <= FEED;
                    end
                end
                FEED: if (reducer_ready) begin
                    remaining <= remaining - 1'b1;
                    index <= index + 1'b1;
                    row_in_chunk <= row_in_chunk + 1'b1;
                    if (remaining == 1)
                        state <= RESULT;
                    else if (row_in_chunk == 31) begin
                        read_score_addr <= read_score_addr + 32'd128;
                        read_scale_addr <= read_scale_addr + 32'd128;
                        state <= SCORE_CMD;
                    end
                end
                RESULT: if (result_valid) begin
                    result_index <= winner_index;
                    result_score <= winner_score;
                    failed <= dma_error || result_invalid;
                    complete <= 1;
                    state <= IDLE;
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
