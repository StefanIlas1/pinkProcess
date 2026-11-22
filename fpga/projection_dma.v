// 64-bit aligned DDR transfers. Split at both AXI3's 16-beat limit and 4 KiB boundaries.
module projection_dma (
    input wire clk, rst, clear_error,
    output wire error,
    input wire rd_cmd_valid, output wire rd_cmd_ready,
    input wire [31:0] rd_cmd_addr, input wire [8:0] rd_cmd_beats,
    output wire rd_data_valid, input wire rd_data_ready, output wire [63:0] rd_data,
    input wire wr_cmd_valid, output wire wr_cmd_ready,
    input wire [31:0] wr_cmd_addr, input wire [8:0] wr_cmd_beats,
    input wire wr_data_valid, output wire wr_data_ready, input wire [63:0] wr_data,
    input wire wr_data_last, output reg wr_done,

    output reg [31:0] m_axi_araddr, output wire [7:0] m_axi_arlen,
    output wire m_axi_arvalid, input wire m_axi_arready,
    input wire [63:0] m_axi_rdata, input wire [1:0] m_axi_rresp, input wire m_axi_rlast,
    input wire m_axi_rvalid, output wire m_axi_rready,
    output reg [31:0] m_axi_awaddr, output wire [7:0] m_axi_awlen,
    output wire m_axi_awvalid, input wire m_axi_awready,
    output wire [63:0] m_axi_wdata, output wire m_axi_wlast,
    output wire m_axi_wvalid, input wire m_axi_wready,
    input wire [1:0] m_axi_bresp, input wire m_axi_bvalid, output wire m_axi_bready
);
    function [4:0] burst_length;
        input [8:0] remaining;
        input [31:0] address;
        reg [9:0] before_boundary;
        reg [4:0] limited;
        begin
            before_boundary = 10'd512 - {1'b0,address[11:3]};
            limited = remaining > 16 ? 5'd16 : remaining[4:0];
            burst_length = before_boundary < limited ? before_boundary[4:0] : limited;
        end
    endfunction

    localparam IDLE=0, ADDRESS=1, DATA=2, RESPONSE=3;
    reg [1:0] read_state, write_state;
    reg [8:0] read_remaining, write_remaining;
    reg [4:0] read_left, write_left;
    reg read_error, write_error;
    assign error = read_error || write_error;

    wire [4:0] read_length = burst_length(read_remaining, m_axi_araddr);
    assign m_axi_arlen = {3'b0,read_length} - 8'd1;
    assign m_axi_arvalid = read_state == ADDRESS;
    assign rd_cmd_ready = read_state == IDLE;
    assign m_axi_rready = read_state == DATA && rd_data_ready;
    assign rd_data_valid = read_state == DATA && m_axi_rvalid;
    assign rd_data = m_axi_rdata;
    always @(posedge clk) begin
        if (rst) begin
            read_state <= IDLE;
            read_error <= 0;
        end else begin
            if (clear_error) read_error <= 0;
            case (read_state)
                IDLE: if (rd_cmd_valid) begin
                    m_axi_araddr <= rd_cmd_addr;
                    read_remaining <= rd_cmd_beats;
                    read_state <= ADDRESS;
                end
                ADDRESS: if (m_axi_arready) begin
                    read_left <= read_length;
                    read_state <= DATA;
                end
                DATA: if (m_axi_rvalid && m_axi_rready) begin
                    if (m_axi_rresp != 0 || m_axi_rlast != (read_left == 1))
                        read_error <= 1;
                    read_left <= read_left - 1'b1;
                    read_remaining <= read_remaining - 1'b1;
                    m_axi_araddr <= m_axi_araddr + 8;
                    if (read_left == 1)
                        read_state <= read_remaining == 1 ? IDLE : ADDRESS;
                end
            endcase
        end
    end

    wire [4:0] write_length = burst_length(write_remaining, m_axi_awaddr);
    reg aw_pending, burst_data_done;
    reg [7:0] awlen_q;
    assign m_axi_awlen = awlen_q;
    assign m_axi_awvalid = write_state == DATA && aw_pending;
    assign wr_cmd_ready = write_state == IDLE;
    assign m_axi_wvalid = write_state == DATA && !burst_data_done && wr_data_valid;
    assign wr_data_ready = write_state == DATA && !burst_data_done && m_axi_wready;
    assign m_axi_wdata = wr_data;
    assign m_axi_wlast = write_left == 1;
    assign m_axi_bready = write_state == RESPONSE;
    always @(posedge clk) begin
        if (rst) begin
            write_state <= IDLE;
            write_error <= 0;
            wr_done <= 0;
        end else begin
            wr_done <= 0;
            if (clear_error) write_error <= 0;
            case (write_state)
                IDLE: if (wr_cmd_valid) begin
                    m_axi_awaddr <= wr_cmd_addr;
                    write_remaining <= wr_cmd_beats;
                    write_state <= ADDRESS;
                end
                ADDRESS: begin
                    aw_pending <= 1;
                    burst_data_done <= 0;
                    awlen_q <= {3'b0,write_length} - 8'd1;
                    write_left <= write_length;
                    write_state <= DATA;
                end
                DATA: begin
                    if (m_axi_awvalid && m_axi_awready) aw_pending <= 0;
                    if (m_axi_wvalid && m_axi_wready) begin
                        if (wr_data_last != (write_remaining == 1)) write_error <= 1;
                        write_left <= write_left - 1'b1;
                        write_remaining <= write_remaining - 1'b1;
                        if (write_left == 1) burst_data_done <= 1;
                    end
                    if (!aw_pending && burst_data_done) write_state <= RESPONSE;
                end
                RESPONSE: if (m_axi_bvalid) begin
                    if (m_axi_bresp != 0) write_error <= 1;
                    if (write_remaining == 0) begin
                        wr_done <= 1;
                        write_state <= IDLE;
                    end else begin
                        m_axi_awaddr <= m_axi_awaddr + (({24'b0, awlen_q} + 1) << 3);
                        write_state <= ADDRESS;
                    end
                end
            endcase
        end
    end
endmodule
