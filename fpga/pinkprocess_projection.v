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
    wire busy, done, dma_error;
    wire running = start || busy;
    wire valid_config = k != 0 && pos_tiles != 0 && out_tiles != 0 &&
                        (!decode || pos_tiles == 1) &&
                        (a_base[2:0] | w_base[2:0] | y_base[2:0]) == 0;
    always @(posedge clk) begin
        if (rst) begin
            start <= 0; completed <= 0; invalid <= 0;
            aw_seen <= 0; w_seen <= 0;
            s_axi_bvalid <= 0; s_axi_rvalid <= 0;
            k <= 0; pos_tiles <= 0; out_tiles <= 0; decode <= 0;
            a_base <= 0; w_base <= 0; y_base <= 0;
        end else begin
            start <= 0;
            if (done) completed <= 1;
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
                    default: s_axi_bresp <= 2;
                endcase
            end
            if (s_axi_rvalid && s_axi_rready) s_axi_rvalid <= 0;
            if (s_axi_arvalid && s_axi_arready) begin
                s_axi_rvalid <= 1;
                s_axi_rresp <= 0;
                case (s_axi_araddr)
                    0: s_axi_rdata <= {29'b0,(invalid || dma_error),(completed || done),running};
                    4: s_axi_rdata <= decode;
                    8: s_axi_rdata <= k;
                    12: s_axi_rdata <= pos_tiles;
                    16: s_axi_rdata <= out_tiles;
                    20: s_axi_rdata <= a_base;
                    24: s_axi_rdata <= w_base;
                    28: s_axi_rdata <= y_base;
                    32: s_axi_rdata <= 32'h50504a31;
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
        .decode(decode), .k(k), .pos_tiles(pos_tiles),
        .out_tiles(out_tiles), .a_base(a_base), .w_base(w_base),
        .y_base(y_base), .busy(busy), .done(done),
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
