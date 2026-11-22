`timescale 1ns/1ps
module projection_argmax_dma_tb;
    reg clk = 0;
    always #5 clk = ~clk;
    reg rst_n = 0;
    reg [11:0] awaddr = 0, araddr = 0;
    reg awvalid = 0, arvalid = 0, wvalid = 0;
    wire awready, arready, wready, bvalid, rvalid;
    reg [31:0] wdata = 0;
    wire [31:0] rdata;
    wire [31:0] read_addr;
    wire [7:0] read_len;
    wire request, data_ready;
    reg [63:0] data = 0;
    reg data_valid = 0, data_last = 0;
    reg [1:0] data_response = 0;
    reg inject_error = 0;
    integer tick = 0, beats_left = 0, next_word = 0, commands = 0;
    reg [63:0] memory [0:2047];
    wire request_ready = beats_left == 0 && !data_valid && tick % 4 == 2;
    reg stalled = 0;
    reg [39:0] previous;
    pinkprocess_argmax dut (
        .s_axi_aclk(clk), .s_axi_aresetn(rst_n),
        .s_axi_awaddr(awaddr), .s_axi_awprot(3'd0), .s_axi_awvalid(awvalid),
        .s_axi_awready(awready), .s_axi_wdata(wdata), .s_axi_wstrb(4'hf),
        .s_axi_wvalid(wvalid), .s_axi_wready(wready), .s_axi_bvalid(bvalid), .s_axi_bready(1'b1),
        .s_axi_araddr(araddr), .s_axi_arprot(3'd0), .s_axi_arvalid(arvalid),
        .s_axi_arready(arready), .s_axi_rdata(rdata), .s_axi_rvalid(rvalid), .s_axi_rready(1'b1),
        .m_axi_araddr(read_addr), .m_axi_arlen(read_len), .m_axi_arvalid(request),
        .m_axi_arready(request_ready), .m_axi_rdata(data), .m_axi_rresp(data_response),
        .m_axi_rlast(data_last), .m_axi_rvalid(data_valid), .m_axi_rready(data_ready)
    );

    always @(posedge clk) if (rst_n) begin
        tick <= tick + 1;
        if (tick == 5000)
            $fatal(1, "DMA test timed out.");
        if (stalled && (!request || {read_addr, read_len} !== previous))
            $fatal(1, "Read command changed under backpressure.");
        stalled <= request && !request_ready;
        previous <= {read_addr, read_len};
        if (request && request_ready) begin
            if (read_addr[2:0] != 0 || read_len >= 16 ||
                int'(read_addr[11:0]) + 8 * (int'(read_len) + 1) > 4096)
                $fatal(1, "Illegal AXI burst.");
            next_word <= int'(read_addr / 8);
            beats_left <= int'(read_len) + 1;
            commands <= commands + 1;
        end
        if (!data_valid || data_ready) begin
            data_valid <= beats_left != 0 && tick % 5 != 3;
            if (beats_left && tick % 5 != 3) begin
                data <= memory[next_word];
                data_last <= beats_left == 1;
                data_response <= inject_error && next_word == 512 ? 2'd2 : 2'd0;
                next_word <= next_word + 1;
                beats_left <= beats_left - 1;
            end
        end
    end

    task write_register(input [11:0] address, input [31:0] value);
        begin
            @(negedge clk);
            awaddr = address;
            awvalid = 1;
            @(posedge clk);
            if (!awready) $fatal(1, "Expected AWREADY while idle.");
            @(negedge clk);
            awvalid = 0;
            wdata = value;
            wvalid = 1;
            @(posedge clk);
            if (!wready) $fatal(1, "Expected WREADY while idle.");
            @(negedge clk);
            wvalid = 0;
            while (!bvalid) @(negedge clk);
            @(posedge clk);
        end
    endtask

    task read_register(input [11:0] address, output [31:0] value);
        begin
            @(negedge clk);
            araddr = address;
            arvalid = 1;
            @(posedge clk);
            if (!arready) $fatal(1, "Expected ARREADY while idle.");
            @(negedge clk);
            arvalid = 0;
            if (!rvalid) $fatal(1, "Expected RVALID.");
            value = rdata;
            @(posedge clk);
        end
    endtask

    task run(input [31:0] wanted_status, input [31:0] wanted_index, input [31:0] wanted_score);
        reg [31:0] status, index, score;
        begin
            write_register(0, 1);
            read_register(0, status);
            while (status[0]) read_register(0, status);
            if (status !== wanted_status)
                $fatal(1, "Got status %h expected %h", status, wanted_status);
            if (wanted_status == 2) begin
                read_register('h14, index);
                read_register('h18, score);
                if (index !== wanted_index || score !== wanted_score)
                    $fatal(1, "Got index %0d and score %h", index, score);
            end
        end
    endtask

    integer word;
    reg [31:0] core_id;
    initial begin
        // Score buffer crosses 4 KiB in its first command.
        for (word = 0; word < 17; word = word + 1) begin
            memory[511 + word] = {32'd8, 32'd8};
            memory[1024 + word] = {32'h3f800000, 32'h3f800000};
        end
        // Score 8 × (0.5 × 4) wins. The padded lane is even larger and is ignored.
        memory[511 + 16][63:32] = 125;
        memory[1024 + 16][31:0] = 32'h40800000;
        repeat (2) @(posedge clk);
        @(negedge clk);
        rst_n = 1;
        read_register('h1c, core_id);
        if (core_id !== 32'h50415231) $fatal(1, "Wrong core ID.");
        run(6, 0, 0);
        write_register('h04, 32'h0ff8);
        write_register('h08, 32'h2000);
        write_register('h0c, 33);
        write_register('h10, 32'h3f000000);
        run(2, 32, 32'h41800000);
        if (commands != 5) $fatal(1, "Expected one split and four logical bursts.");
        inject_error = 1;
        run(6, 0, 0);
        inject_error = 0;
        // An earlier row ties the last row, proving first-index tie-breaking.
        memory[1024][63:32] = 32'h40800000;
        run(2, 1, 32'h41800000);
        $display("Verified AXI argmax: split reads, odd tail, stalls, error recovery and ties.");
        $finish;
    end
endmodule
