// Streams packed INT8 operands through a projection tile and writes INT32 scores.
module projection_engine (
    input  wire         clk,
    input  wire         rst,
    input  wire         start,
    input  wire         decode,
    input  wire [15:0]  k,
    input  wire [15:0]  pos_tiles,
    input  wire [15:0]  out_tiles,
    input  wire [31:0]  a_base,
    input  wire [31:0]  w_base,
    input  wire [31:0]  y_base,
    output wire         busy,
    output reg          done,

    output wire         rd_cmd_valid,
    input  wire         rd_cmd_ready,
    output wire [31:0]  rd_cmd_addr,
    output wire [8:0]   rd_cmd_beats,
    input  wire         rd_data_valid,
    output wire         rd_data_ready,
    input  wire [63:0]  rd_data,

    output wire         wr_cmd_valid,
    input  wire         wr_cmd_ready,
    output wire [31:0]  wr_cmd_addr,
    output wire [8:0]   wr_cmd_beats,
    output wire         wr_data_valid,
    input  wire         wr_data_ready,
    output wire [63:0]  wr_data,
    output wire         wr_data_last,
    input  wire         wr_done
);
    localparam [2:0] IDLE = 0, PREPARE = 1, WAIT_CHUNK = 2, FEED = 3,
                     FLUSH = 4, WRITE_CMD = 5, WRITE_DATA = 6, WRITE_RESPONSE = 7;
    localparam [2:0] LOAD_IDLE = 0, READ_A_CMD = 1, READ_A = 2,
                     READ_W_CMD = 3, READ_W = 4;
    localparam [5:0] CHUNK_SIZE = 32;

    reg [2:0] state;
    reg [2:0] load_state;
    reg mode_decode;
    reg [15:0] depth, positions_left, outputs_left, output_tiles;
    reg [31:0] a_tile_base, w_tile_base, w_first_base, y_ptr;

    reg [31:0] a_read_ptr, w_read_ptr;
    reg [15:0] load_left, feed_left;
    reg load_bank, feed_bank;
    reg [1:0] bank_full;
    reg [5:0] bank_length [0:1];
    reg [5:0] load_length;
    reg [7:0] read_index;
    reg [4:0] feed_index;
    reg [3:0] flush_index;
    reg [4:0] write_index;

    wire [8:0] a_beats = {3'b0, load_length};
    wire [8:0] w_beats = mode_decode ? {load_length, 3'b0} : a_beats;
    wire accept_data = rd_data_valid && rd_data_ready;
    wire finish_load = load_state == READ_W && accept_data &&
                       {1'b0, read_index} == w_beats - 9'd1;
    wire [5:0] feed_length = bank_length[feed_bank];
    wire finish_chunk = state == FEED && {1'b0, feed_index} == feed_length - 6'd1;
    wire last_chunk = feed_left == 16'd1;
    wire next_bank = !feed_bank;
    wire issue_read = (state == WAIT_CHUNK && bank_full[feed_bank]) ||
                      (state == FEED && (!finish_chunk ||
                       (!last_chunk && bank_full[next_bank])));
    wire read_other_bank = finish_chunk;
    wire [5:0] buffer_read_addr = {
        feed_bank ^ read_other_bank,
        (state == WAIT_CHUNK || read_other_bank) ? 5'd0 : feed_index + 5'd1
    };
    wire [4:0] weight_row = mode_decode ? read_index[7:3] : read_index[4:0];
    wire [2:0] weight_lane = mode_decode ? read_index[2:0] : 3'd0;

    (* ram_style = "block" *) reg [63:0] activation_buffer [0:63];
    reg [63:0] tile_activations;
    wire [511:0] tile_weights;
    wire [2047:0] tile_sums;

    always @(posedge clk) begin
        if (load_state == READ_A && accept_data)
            activation_buffer[{load_bank, read_index[4:0]}] <= rd_data;
        if (issue_read)
            tile_activations <= activation_buffer[buffer_read_addr];
    end

    for (genvar lane = 0; lane < 8; lane = lane + 1) begin : weight_buffer
        (* ram_style = "block" *) reg [63:0] words [0:63];
        reg [63:0] tile_word;
        always @(posedge clk) begin
            if (load_state == READ_W && accept_data && weight_lane == lane)
                words[{load_bank, weight_row}] <= rd_data;
            if (issue_read)
                tile_word <= words[buffer_read_addr];
        end
        assign tile_weights[64*lane +: 64] = tile_word;
    end

    projection_tile tile (
        .clk(clk),
        .rst(rst),
        .clear(state == PREPARE),
        .step(state == FEED || state == FLUSH),
        .decode(mode_decode),
        .activations(state == FLUSH ? 64'd0 : tile_activations),
        .weights(state == FLUSH ? 512'd0 : tile_weights),
        .sums(tile_sums)
    );

    assign busy = state != IDLE;
    assign rd_cmd_valid = load_state == READ_A_CMD || load_state == READ_W_CMD;
    assign rd_cmd_addr = load_state == READ_A_CMD ? a_read_ptr : w_read_ptr;
    assign rd_cmd_beats = load_state == READ_A_CMD ? a_beats : w_beats;
    assign rd_data_ready = load_state == READ_A || load_state == READ_W;
    assign wr_cmd_valid = state == WRITE_CMD;
    assign wr_cmd_addr = y_ptr;
    assign wr_cmd_beats = 9'd32;
    assign wr_data_valid = state == WRITE_DATA;
    assign wr_data = tile_sums[{write_index, 6'b0} +: 64];
    assign wr_data_last = write_index == 5'd31;

    // The loader owns a bank until its last weight beat. The feeder holds
    // that bank until its last MAC and can already prime the opposite bank.
    always @(posedge clk) begin
        if (rst || state == PREPARE) begin
            bank_full <= '0;
        end else begin
            if (finish_load)
                bank_full[load_bank] <= 1'b1;
            if (finish_chunk)
                bank_full[feed_bank] <= 1'b0;
        end
    end

    always @(posedge clk) begin
        if (rst) begin
            load_state <= LOAD_IDLE;
        end else if (state == PREPARE) begin
            load_state <= LOAD_IDLE;
            load_left <= depth;
            load_bank <= 1'b0;
            a_read_ptr <= a_tile_base;
            w_read_ptr <= w_tile_base;
        end else begin
            case (load_state)
                LOAD_IDLE: begin
                    if (load_left != 0 && !bank_full[load_bank] &&
                        (state == WAIT_CHUNK || state == FEED)) begin
                        load_length <= load_left > {10'b0, CHUNK_SIZE} ?
                                       CHUNK_SIZE : load_left[5:0];
                        load_state <= READ_A_CMD;
                    end
                end
                READ_A_CMD: begin
                    if (rd_cmd_ready) begin
                        read_index <= 0;
                        load_state <= READ_A;
                    end
                end
                READ_A: begin
                    if (accept_data) begin
                        if ({1'b0, read_index} == a_beats - 9'd1)
                            load_state <= READ_W_CMD;
                        read_index <= read_index + 8'd1;
                    end
                end
                READ_W_CMD: begin
                    if (rd_cmd_ready) begin
                        read_index <= 0;
                        load_state <= READ_W;
                    end
                end
                READ_W: begin
                    if (accept_data) begin
                        read_index <= read_index + 8'd1;
                        if (finish_load) begin
                            bank_length[load_bank] <= load_length;
                            load_left <= load_left - {10'b0, load_length};
                            a_read_ptr <= a_read_ptr + {20'b0, a_beats, 3'b0};
                            w_read_ptr <= w_read_ptr + {20'b0, w_beats, 3'b0};
                            load_bank <= !load_bank;
                            load_state <= LOAD_IDLE;
                        end
                    end
                end
                default: load_state <= LOAD_IDLE;
            endcase
        end
    end

    always @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            done <= 1'b0;
        end else begin
            done <= 1'b0;
            case (state)
                IDLE: begin
                    if (start) begin
                        mode_decode <= decode;
                        depth <= k;
                        positions_left <= pos_tiles;
                        outputs_left <= out_tiles;
                        output_tiles <= out_tiles;
                        a_tile_base <= a_base;
                        w_tile_base <= w_base;
                        w_first_base <= w_base;
                        y_ptr <= y_base;
                        state <= PREPARE;
                    end
                end
                PREPARE: begin
                    feed_left <= depth;
                    feed_bank <= 1'b0;
                    feed_index <= 0;
                    state <= WAIT_CHUNK;
                end
                WAIT_CHUNK: begin
                    if (bank_full[feed_bank])
                        state <= FEED;
                end
                FEED: begin
                    feed_left <= feed_left - 16'd1;
                    feed_index <= feed_index + 5'd1;
                    if (finish_chunk) begin
                        feed_bank <= next_bank;
                        feed_index <= 0;
                        if (last_chunk) begin
                            flush_index <= 0;
                            // Decode MACs consume the broadcast input directly.
                            state <= mode_decode ? WRITE_CMD : FLUSH;
                        end else if (!bank_full[next_bank]) begin
                            state <= WAIT_CHUNK;
                        end
                    end
                end
                FLUSH: begin
                    flush_index <= flush_index + 4'd1;
                    if (flush_index == 4'd13)
                        state <= WRITE_CMD;
                end
                WRITE_CMD: begin
                    if (wr_cmd_ready) begin
                        write_index <= 0;
                        state <= WRITE_DATA;
                    end
                end
                WRITE_DATA: begin
                    if (wr_data_ready) begin
                        write_index <= write_index + 5'd1;
                        if (wr_data_last)
                            state <= WRITE_RESPONSE;
                    end
                end
                WRITE_RESPONSE: begin
                    if (wr_done) begin
                        y_ptr <= y_ptr + 32'd256;
                        if (outputs_left > 16'd1) begin
                            outputs_left <= outputs_left - 16'd1;
                            w_tile_base <= w_read_ptr;
                            state <= PREPARE;
                        end else if (positions_left > 16'd1) begin
                            positions_left <= positions_left - 16'd1;
                            outputs_left <= output_tiles;
                            a_tile_base <= a_read_ptr;
                            w_tile_base <= w_first_base;
                            state <= PREPARE;
                        end else begin
                            done <= 1'b1;
                            state <= IDLE;
                        end
                    end
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
