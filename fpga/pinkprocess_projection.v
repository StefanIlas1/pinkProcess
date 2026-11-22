// Host writes the physical addresses of three contiguous packed buffers, then
// starts a complete projection. Kernel status remains latched until the next start.
module pinkprocess_projection (
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
    input wire m_axi_rlast, input wire m_axi_rvalid, output wire m_axi_rready,
    output wire [31:0] m_axi_awaddr, output wire [7:0] m_axi_awlen,
    output wire [2:0] m_axi_awsize, output wire [1:0] m_axi_awburst,
    output wire [3:0] m_axi_awcache, output wire [2:0] m_axi_awprot,
    output wire m_axi_awlock, output wire m_axi_awvalid, input wire m_axi_awready,
    output wire [63:0] m_axi_wdata, output wire [7:0] m_axi_wstrb,
    output wire m_axi_wlast, output wire m_axi_wvalid, input wire m_axi_wready,
    input wire [1:0] m_axi_bresp, input wire m_axi_bvalid, output wire m_axi_bready
);
    wire clk = s_axi_aclk;
    wire rst = !s_axi_aresetn;
    assign m_axi_arsize=3; assign m_axi_awsize=3;
    assign m_axi_arburst=1; assign m_axi_awburst=1;
    assign m_axi_arcache=3; assign m_axi_awcache=3;
    assign m_axi_arprot=0; assign m_axi_awprot=0;
    assign m_axi_arlock=0; assign m_axi_awlock=0;
    assign m_axi_wstrb=8'hff;

    reg aw_seen, w_seen;
    reg [11:0] address;
    reg [31:0] value;
    reg [3:0] strobes;
    assign s_axi_awready = !aw_seen && !s_axi_bvalid;
    assign s_axi_wready = !w_seen && !s_axi_bvalid;
    assign s_axi_arready = !s_axi_rvalid;

    reg start, completed, invalid, decode;
    reg [15:0] k, pos_tiles, out_tiles;
    reg [31:0] a_base, w_base, y_base;
    wire busy, done, dma_error, mac_feed;
    reg [31:0] elapsed_cycles, feed_cycles, read_beats, write_beats;
    // Descriptors are staged while idle; one doorbell runs up to eight in order.
    // Keep the legacy configuration readable throughout execution.
    reg [144:0] descriptors [0:7];
    reg [3:0] queued, queue_index;
    reg queue_active;
    wire [144:0] config_words = {decode, k, pos_tiles, out_tiles, a_base, w_base, y_base};
    wire [144:0] current_words = queue_active ? descriptors[queue_index[2:0]] : config_words;
    wire engine_decode;
    wire [15:0] engine_k, engine_positions, engine_outputs;
    wire [31:0] engine_a, engine_w, engine_y;
    assign {engine_decode, engine_k, engine_positions, engine_outputs,
            engine_a, engine_w, engine_y} = current_words;
    wire running = start || busy || queue_active;
    wire valid_config = k != 0 && pos_tiles != 0 && out_tiles != 0 &&
                        (!decode || pos_tiles == 1) &&
                        (a_base[2:0] | w_base[2:0] | y_base[2:0]) == 0;
    // Counters span the whole batch and remain readable until the next launch.
    wire first_launch = start && (!queue_active || queue_index == 0);
    always @(posedge clk) begin
        if (rst || first_launch) begin
            elapsed_cycles <= 0; feed_cycles <= 0; read_beats <= 0; write_beats <= 0;
        end else if (running) begin
            elapsed_cycles <= elapsed_cycles + 1'b1;
            if (mac_feed) feed_cycles <= feed_cycles + 1'b1;
            if (m_axi_rvalid && m_axi_rready) read_beats <= read_beats + 1'b1;
            if (m_axi_wvalid && m_axi_wready) write_beats <= write_beats + 1'b1;
        end
    end
    always @(posedge clk) begin
        if (rst) begin
            start <= 0; completed <= 0; invalid <= 0;
            queued <= 0; queue_index <= 0; queue_active <= 0;
            aw_seen <= 0; w_seen <= 0;
            s_axi_bvalid <= 0; s_axi_rvalid <= 0;
            k <= 0; pos_tiles <= 0; out_tiles <= 0; decode <= 0;
            a_base <= 0; w_base <= 0; y_base <= 0;
        end else begin
            start <= 0;
            if (done) begin
                if (queue_active && !dma_error && queue_index + 1 < queued) begin
                    queue_index <= queue_index + 1'b1;
                    start <= 1;
                end else begin
                    completed <= 1;
                    if (queue_active) begin
                        queue_active <= 0;
                        queued <= 0;
                    end
                end
            end
            if (s_axi_awvalid && s_axi_awready) begin
                aw_seen <= 1;
                address <= s_axi_awaddr;
            end
            if (s_axi_wvalid && s_axi_wready) begin
                w_seen <= 1;
                value <= s_axi_wdata;
                strobes <= s_axi_wstrb;
            end
            if (s_axi_bvalid && s_axi_bready) s_axi_bvalid <= 0;
            if (aw_seen && w_seen) begin
                aw_seen <= 0; w_seen <= 0; s_axi_bvalid <= 1;
                s_axi_bresp <= 0;
                if (running || strobes != 4'hf) s_axi_bresp <= 2;
                else case (address)
                    0: if (value == 1) begin
                        invalid <= !valid_config;
                        completed <= !valid_config;
                        start <= valid_config;
                    end else s_axi_bresp <= 2;
                    4: decode <= value[0];
                    8: k <= value[15:0];
                    12: pos_tiles <= value[15:0];
                    16: out_tiles <= value[15:0];
                    20: a_base <= value;
                    24: w_base <= value;
                    28: y_base <= value;
                    36: case (value)
                        0: queued <= 0;
                        1: if (valid_config && queued < 8) begin
                            descriptors[queued[2:0]] <= config_words;
                            queued <= queued + 1'b1;
                        end else s_axi_bresp <= 2;
                        2: if (queued != 0) begin
                            queue_index <= 0;
                            queue_active <= 1;
                            completed <= 0;
                            invalid <= 0;
                            start <= 1;
                        end else s_axi_bresp <= 2;
                        default: s_axi_bresp <= 2;
                    endcase
                    default: s_axi_bresp <= 2;
                endcase
            end
            if (s_axi_rvalid && s_axi_rready) s_axi_rvalid <= 0;
            if (s_axi_arvalid && s_axi_arready) begin
                s_axi_rvalid <= 1;
                s_axi_rresp <= 0;
                case (s_axi_araddr)
                    0: s_axi_rdata <= {29'b0,(invalid || dma_error),(completed || (done && !queue_active)),running};
                    4: s_axi_rdata <= decode;
                    8: s_axi_rdata <= k;
                    12: s_axi_rdata <= pos_tiles;
                    16: s_axi_rdata <= out_tiles;
                    20: s_axi_rdata <= a_base;
                    24: s_axi_rdata <= w_base;
                    28: s_axi_rdata <= y_base;
                    32: s_axi_rdata <= 32'h50504a31;
                    36: s_axi_rdata <= {24'b0, queue_index, queued};
                    40: s_axi_rdata <= 32'h51554538; // QUE8 capability
                    44: s_axi_rdata <= elapsed_cycles;
                    48: s_axi_rdata <= feed_cycles;
                    52: s_axi_rdata <= read_beats;
                    56: s_axi_rdata <= write_beats;
                    default: begin s_axi_rdata <= 0; s_axi_rresp <= 2; end
                endcase
            end
        end
    end

    wire rd_cmd_valid, rd_cmd_ready, rd_data_valid, rd_data_ready;
    wire wr_cmd_valid, wr_cmd_ready, wr_data_valid, wr_data_ready, wr_data_last, wr_done;
    wire [31:0] rd_cmd_addr, wr_cmd_addr;
    wire [8:0] rd_cmd_beats, wr_cmd_beats;
    wire [63:0] rd_data, wr_data;
    projection_engine engine (
        .clk(clk), .rst(rst), .start(start),
        .decode(engine_decode), .k(engine_k), .pos_tiles(engine_positions),
        .out_tiles(engine_outputs), .a_base(engine_a), .w_base(engine_w),
        .y_base(engine_y), .busy(busy), .mac_feed(mac_feed), .done(done),
        .rd_cmd_valid(rd_cmd_valid), .rd_cmd_ready(rd_cmd_ready), .rd_cmd_addr(rd_cmd_addr),
        .rd_cmd_beats(rd_cmd_beats), .rd_data_valid(rd_data_valid), .rd_data_ready(rd_data_ready),
        .rd_data(rd_data), .wr_cmd_valid(wr_cmd_valid), .wr_cmd_ready(wr_cmd_ready),
        .wr_cmd_addr(wr_cmd_addr), .wr_cmd_beats(wr_cmd_beats), .wr_data_valid(wr_data_valid),
        .wr_data_ready(wr_data_ready), .wr_data(wr_data), .wr_data_last(wr_data_last),
        .wr_done(wr_done)
    );
    projection_dma dma (
        .clk(clk), .rst(rst), .rd_cmd_valid(rd_cmd_valid),
        .rd_cmd_ready(rd_cmd_ready), .rd_cmd_addr(rd_cmd_addr), .rd_cmd_beats(rd_cmd_beats),
        .rd_data_valid(rd_data_valid), .rd_data_ready(rd_data_ready), .rd_data(rd_data),
        .wr_cmd_valid(wr_cmd_valid), .wr_cmd_ready(wr_cmd_ready), .wr_cmd_addr(wr_cmd_addr),
        .wr_cmd_beats(wr_cmd_beats), .wr_data_valid(wr_data_valid), .wr_data_ready(wr_data_ready),
        .wr_data(wr_data), .wr_data_last(wr_data_last), .wr_done(wr_done),
        .m_axi_araddr(m_axi_araddr), .m_axi_arlen(m_axi_arlen), .m_axi_arvalid(m_axi_arvalid),
        .m_axi_arready(m_axi_arready), .m_axi_rdata(m_axi_rdata), .m_axi_rresp(m_axi_rresp),
        .m_axi_rlast(m_axi_rlast), .m_axi_rvalid(m_axi_rvalid), .m_axi_rready(m_axi_rready),
        .m_axi_awaddr(m_axi_awaddr), .m_axi_awlen(m_axi_awlen), .m_axi_awvalid(m_axi_awvalid),
        .m_axi_awready(m_axi_awready), .m_axi_wdata(m_axi_wdata), .m_axi_wlast(m_axi_wlast),
        .m_axi_wvalid(m_axi_wvalid), .m_axi_wready(m_axi_wready), .m_axi_bresp(m_axi_bresp),
        .m_axi_bvalid(m_axi_bvalid), .m_axi_bready(m_axi_bready), .clear_error(start),
        .error(dma_error)
    );
endmodule
