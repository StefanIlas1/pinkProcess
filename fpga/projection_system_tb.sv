`timescale 1ns/1ps
module projection_system_tb;
    localparam integer K = 37;
    localparam [31:0] A = 32'h00000ff0, W = 32'h00080ff8, Y = 32'h00100ff8;
    localparam integer DDR_WORDS = 180000;
    localparam [63:0] UNWRITTEN = 64'hdeadbeef_deadbeef;

    reg clk = 0;
    always #5 clk = ~clk;
    reg resetn = 0;
    reg [11:0] awaddr = 0, araddr = 0;
    reg [31:0] wdata = 0;
    reg [3:0] wstrb = 0;
    reg awvalid = 0, wvalid = 0, arvalid = 0, bready = 0, rready = 0;
    wire awready, wready, arready, bvalid, rvalid;
    wire [1:0] bresp, rresp;
    wire [31:0] rdata;

    wire [31:0] dma_awaddr, dma_araddr;
    wire [7:0] dma_awlen, dma_arlen;
    wire [2:0] dma_awsize, dma_arsize;
    wire [1:0] dma_awburst, dma_arburst;
    wire dma_awvalid, dma_arvalid, dma_wvalid, dma_wlast, dma_rready, dma_bready;
    wire [63:0] dma_wdata;
    wire [7:0] dma_wstrb;
    reg [63:0] dma_rdata = 0;
    reg [1:0] dma_rresp = 0, dma_bresp = 0;
    reg dma_rlast = 0, dma_rvalid = 0, dma_bvalid = 0;
    reg [63:0] ddr [0:DDR_WORDS-1];
    reg [63:0] oracle [0:DDR_WORDS-Y/8-1];

    integer tick = 0, read_addr = 0, read_left = 0;
    integer write_addr = 0, write_beats = 0, write_count = 0, aw_delay = 0;
    reg write_has_addr = 0, write_has_last = 0;
    reg [63:0] write_buffer [0:15];
    reg write_last [0:15];
    reg inject_write_error = 0, injected_error = 0;
    integer read_crossing = 0, write_crossing = 0, full_read = 0, full_write = 0;
    integer read_stalls = 0, write_stalls = 0, early_write = 0;
    integer command_tick = 0, last_response_tick = 0;
    reg stalled_ar = 0, stalled_aw = 0, stalled_w = 0;
    reg [39:0] held_ar, held_aw;
    reg [64:0] held_w;
    integer index;

    wire dma_arready = read_left == 0 && !dma_rvalid && tick % 5 == 2;
    wire dma_awready = !write_has_addr && !dma_bvalid && aw_delay >= 4;
    wire dma_wready = !write_has_last && !dma_bvalid && tick % 4 != 2;

    pinkprocess_projection dut (
        .s_axi_aclk(clk), .s_axi_aresetn(resetn),
        .s_axi_awaddr(awaddr), .s_axi_awprot(3'b0), .s_axi_awvalid(awvalid), .s_axi_awready(awready),
        .s_axi_wdata(wdata), .s_axi_wstrb(wstrb), .s_axi_wvalid(wvalid), .s_axi_wready(wready),
        .s_axi_bresp(bresp), .s_axi_bvalid(bvalid), .s_axi_bready(bready),
        .s_axi_araddr(araddr), .s_axi_arprot(3'b0), .s_axi_arvalid(arvalid), .s_axi_arready(arready),
        .s_axi_rdata(rdata), .s_axi_rresp(rresp), .s_axi_rvalid(rvalid), .s_axi_rready(rready),
        .m_axi_araddr(dma_araddr), .m_axi_arlen(dma_arlen),
        .m_axi_arsize(dma_arsize), .m_axi_arburst(dma_arburst),
        .m_axi_arvalid(dma_arvalid), .m_axi_arready(dma_arready),
        .m_axi_rdata(dma_rdata), .m_axi_rresp(dma_rresp), .m_axi_rlast(dma_rlast),
        .m_axi_rvalid(dma_rvalid), .m_axi_rready(dma_rready),
        .m_axi_awaddr(dma_awaddr), .m_axi_awlen(dma_awlen),
        .m_axi_awsize(dma_awsize), .m_axi_awburst(dma_awburst),
        .m_axi_awvalid(dma_awvalid), .m_axi_awready(dma_awready),
        .m_axi_wdata(dma_wdata), .m_axi_wstrb(dma_wstrb), .m_axi_wlast(dma_wlast),
        .m_axi_wvalid(dma_wvalid), .m_axi_wready(dma_wready),
        .m_axi_bresp(dma_bresp), .m_axi_bvalid(dma_bvalid), .m_axi_bready(dma_bready)
    );

    task check_burst(input [31:0] address, input [7:0] len);
        integer end_byte;
        begin
            end_byte = (int'(address[11:0]) + 8 * (int'(len) + 1));
            if (address[2:0] != 0 || len > 15 || end_byte > 4096 ||
                int'(address / 8) + int'(len) >= DDR_WORDS)
                $fatal(1, "Invalid DMA burst at %h, length %0d", address, int'(len) + 1);
        end
    endtask

    always @(posedge clk) if (resetn) begin
        tick <= tick + 1;
        if (awvalid && awready && awaddr == 0) command_tick <= tick;
        if (stalled_ar && {dma_arvalid, dma_araddr, dma_arlen} !== {1'b1, held_ar})
            $fatal(1, "DMA read address changed under backpressure");
        if (stalled_aw && {dma_awvalid, dma_awaddr, dma_awlen} !== {1'b1, held_aw})
            $fatal(1, "DMA write address changed under backpressure");
        if (stalled_w && {dma_wvalid, dma_wdata, dma_wlast} !== {1'b1, held_w})
            $fatal(1, "DMA write data changed under backpressure");
        stalled_ar <= dma_arvalid && !dma_arready;
        stalled_aw <= dma_awvalid && !dma_awready;
        stalled_w <= dma_wvalid && !dma_wready;
        held_ar <= {dma_araddr, dma_arlen};
        held_aw <= {dma_awaddr, dma_awlen};
        held_w <= {dma_wdata, dma_wlast};

        if (dma_arvalid && !dma_arready) read_stalls <= read_stalls + 1;
        if (dma_arvalid && dma_arready) begin
            if ({dma_arsize, dma_arburst} !== {3'd3, 2'b01})
                $fatal(1, "DMA reads must use 64-bit incrementing bursts");
            check_burst(dma_araddr, dma_arlen);
            read_addr <= dma_araddr >> 3;
            read_left <= int'(dma_arlen) + 1;
            if (dma_araddr[11:0] >= 12'hff0) read_crossing <= read_crossing + 1;
            if (dma_arlen == 15) full_read <= full_read + 1;
        end
        if (!dma_rvalid || dma_rready) begin
            dma_rvalid <= read_left != 0 && tick % 7 != 3;
            if (read_left != 0 && tick % 7 != 3) begin
                dma_rdata <= ddr[read_addr];
                dma_rlast <= read_left == 1;
                read_addr <= read_addr + 1;
                read_left <= read_left - 1;
            end
        end

        if (dma_awvalid && !dma_awready) aw_delay <= aw_delay + 1;
        if (dma_awvalid && dma_awready) begin
            if ({dma_awsize, dma_awburst} !== {3'd3, 2'b01})
                $fatal(1, "DMA writes must use 64-bit incrementing bursts");
            check_burst(dma_awaddr, dma_awlen);
            write_addr <= dma_awaddr >> 3;
            write_beats <= int'(dma_awlen) + 1;
            write_has_addr <= 1;
            aw_delay <= 0;
            if (dma_awaddr[11:0] >= 12'hff0) write_crossing <= write_crossing + 1;
            if (dma_awlen == 15) full_write <= full_write + 1;
        end
        if (dma_wvalid && !dma_wready) write_stalls <= write_stalls + 1;
        if (dma_wvalid && dma_wready) begin
            if (write_count >= 16 || dma_wstrb !== 8'hff)
                $fatal(1, "Invalid DMA write beat %0d, strobe %h", write_count, dma_wstrb);
            if (!write_has_addr && !dma_awready) early_write <= early_write + 1;
            write_buffer[write_count] <= dma_wdata;
            write_last[write_count] <= dma_wlast;
            write_count <= write_count + 1;
            write_has_last <= dma_wlast;
        end
        if (dma_bvalid && dma_bready) begin
            last_response_tick <= tick;
            dma_bvalid <= 0;
            write_has_addr <= 0;
            write_has_last <= 0;
            write_count <= 0;
        end else if (write_has_addr && write_has_last && !dma_bvalid && tick % 6 == 1) begin
            if (write_count != write_beats)
                $fatal(1, "DMA sent %0d write beats, expected %0d", write_count, write_beats);
            for (index = 0; index < write_count; index = index + 1) begin
                if (write_last[index] !== (index == write_count - 1))
                    $fatal(1, "Invalid WLAST at write beat %0d", index);
                ddr[write_addr + index] <= write_buffer[index];
            end
            dma_bvalid <= 1;
            dma_bresp <= inject_write_error && !injected_error ? 2'b10 : 2'b00;
            injected_error <= injected_error || inject_write_error;
        end
    end

    task control_address(input [11:0] address);
        @(negedge clk);
        awaddr = address;
        awvalid = 1;
        do @(posedge clk); while (!awready);
        @(negedge clk);
        awvalid = 0;
    endtask

    task control_data(input [31:0] data, input [3:0] mask);
        @(negedge clk);
        wdata = data;
        wstrb = mask;
        wvalid = 1;
        do @(posedge clk); while (!wready);
        @(negedge clk);
        wvalid = 0;
    endtask

    task control_write(input [11:0] address, input [31:0] data, input [3:0] mask = 4'hf,
                       input [1:0] expected_response = 2'b00);
        begin
            if (address[2]) begin
                control_address(address);
                control_data(data, mask);
            end else begin
                control_data(data, mask);
                control_address(address);
            end
            repeat (2) @(negedge clk);
            bready = 1;
            do @(posedge clk); while (!bvalid);
            if (bresp !== expected_response)
                $fatal(1, "Control write %h returned %b, expected %b", address, bresp, expected_response);
            @(negedge clk);
            bready = 0;
        end
    endtask

    task control_read(input [11:0] address, output [31:0] data);
        @(negedge clk);
        araddr = address;
        arvalid = 1;
        do @(posedge clk); while (!arready);
        @(negedge clk);
        arvalid = 0;
        repeat (2) @(negedge clk);
        rready = 1;
        do @(posedge clk); while (!rvalid);
        data = rdata;
        if (rresp !== 2'b00) $fatal(1, "Control read failed at %h: %b", address, rresp);
        @(negedge clk);
        rready = 0;
    endtask

    function automatic signed [7:0] activation(input integer position, reduction);
        activation = 8'((position * 17 + reduction * 31 + 128) % 256 - 128);
    endfunction

    function automatic signed [7:0] weight(input integer column, reduction);
        weight = 8'((column * 43 + reduction * 7 + 255) % 256 - 128);
    endfunction

    task execute(input bit decode, fault, input integer depth, positions, outputs);
        reg [31:0] status;
        begin
            injected_error = 0;
            inject_write_error = fault;
            control_write('h04, {31'b0, decode});
            control_write('h08, depth);
            control_write('h08, 32'hffff_ff00, 4'b0100, 2'b10);
            control_read('h08, status);
            if (status[15:0] !== depth[15:0]) $fatal(1, "Unexpected depth readback: %h", status);
            control_write('h0c, positions);
            control_write('h10, outputs);
            control_write('h14, A);
            control_write('h18, W);
            control_write('h1c, Y);
            control_write('h00, 32'd1);
            control_read('h00, status);
            if (depth >= K) begin
                if (status[2:0] !== 3'b001) $fatal(1, "Expected busy status, got %h", status);
                control_write('h08, 32'd1, 4'hf, 2'b10);
            end
            while (status[0]) control_read('h00, status);
            if (status[2:0] !== {fault, 2'b10})
                $fatal(1, "Expected done with error=%b, status=%h", fault, status);
            if (fault && !injected_error) $fatal(1, "DMA error never injected");
        end
    endtask

    task prepare_output(input integer words);
        integer word_index;
        for (word_index = -1; word_index <= words; word_index = word_index + 1)
            ddr[Y/8 + word_index] = UNWRITTEN;
    endtask

    task compare_output(input bit decode, fault, input integer positions, outputs);
        integer words, word_index;
        begin
            words = positions*outputs*32;
            for (word_index = 0; word_index < words; word_index = word_index + 1)
                if (ddr[Y/8 + word_index] !== oracle[word_index])
                    $fatal(1, "Mode %0d output word %0d: got %h, expected %h",
                           decode, word_index, ddr[Y/8 + word_index], oracle[word_index]);
            if (ddr[Y/8 - 1] !== UNWRITTEN || ddr[Y/8 + words] !== UNWRITTEN)
                $fatal(1, "DMA wrote beyond output tile region");
            $display("Matched %0d DMA outputs, decode=%0d, injected write error=%0d, %0d clocks from start through final DDR write response",
                     2*words, decode, fault, last_response_tick - command_tick + 1);
        end
    endtask

    task run_projection(input bit decode, fault, input integer positions, outputs);
        integer p, o, k, lane, length, output_word;
        reg signed [31:0] expected;
        begin
            length = decode ? 64 : 8;
            for (p = 0; p < positions; p = p + 1)
                for (k = 0; k < K; k = k + 1)
                    for (lane = 0; lane < 8; lane = lane + 1)
                        ddr[A/8 + p*K + k][8*lane +: 8] =
                            activation(p*(decode ? 1 : 8) + lane, k);
            for (o = 0; o < outputs; o = o + 1)
                for (k = 0; k < K; k = k + 1)
                    for (lane = 0; lane < length; lane = lane + 1)
                        ddr[W/8 + (o*K + k)*(length/8) + lane/8][8*(lane%8) +: 8] =
                            weight(o*length + lane, k);
            prepare_output(positions*outputs*32);

            for (p = 0; p < positions; p = p + 1)
                for (o = 0; o < outputs; o = o + 1)
                    for (lane = 0; lane < 64; lane = lane + 1) begin
                        expected = 0;
                        for (k = 0; k < K; k = k + 1)
                            expected = expected +
                                activation(p*(64/length) + lane/length, k) *
                                weight(o*length + lane%length, k);
                        output_word = (p*outputs + o)*32 + lane/2;
                        oracle[output_word][32*(lane%2) +: 32] = expected;
                    end
            execute(decode, fault, K, positions, outputs);
            compare_output(decode, fault, positions, outputs);
        end
    endtask

    task run_capture;
        integer mode, depth, positions, outputs, a_words, w_words, y_words;
        begin
            if (!$value$plusargs("mode=%d", mode) || !$value$plusargs("depth=%d", depth) ||
                !$value$plusargs("positions=%d", positions) || !$value$plusargs("outputs=%d", outputs))
                $fatal(1, "Missing capture dimensions");
            a_words = positions*depth;
            w_words = outputs*depth*(mode ? 8 : 1);
            y_words = positions*outputs*32;
            if (A/8 + a_words > W/8 || W/8 + w_words > Y/8 || Y/8 + y_words >= DDR_WORDS)
                $fatal(1, "Capture exceeds the simulated DDR address regions");
            $readmemh("activation.hex", ddr, A/8, A/8 + a_words - 1);
            $readmemh("weights.hex", ddr, W/8, W/8 + w_words - 1);
            $readmemh("expected.hex", oracle, 0, y_words - 1);
            prepare_output(y_words);
            execute(mode[0], 0, depth, positions, outputs);
            compare_output(mode[0], 0, positions, outputs);
        end
    endtask

    initial begin
        reg [31:0] id;
        repeat (4) @(negedge clk);
        resetn = 1;
        control_read('h20, id);
        if (id !== 32'h50504a31) $fatal(1, "Projection ID returned %h", id);
        if ($test$plusargs("capture")) begin
            run_capture();
        end else begin
            run_projection(0, 0, 2, 2);
            run_projection(1, 1, 1, 2);
            run_projection(1, 0, 1, 2);
            if (read_crossing == 0 || write_crossing == 0 || full_read == 0 || full_write == 0 ||
                read_stalls == 0 || write_stalls == 0 || early_write == 0)
                $fatal(1, "Required DMA boundary or backpressure case was not exercised");
        end
        $finish;
    end

    initial begin
        #100_000_000;
        $fatal(1, "System simulation timed out");
    end
endmodule
