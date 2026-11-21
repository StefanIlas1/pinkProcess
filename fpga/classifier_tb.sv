`timescale 1ns/1ps
module classifier_tb;
    reg clk = 0;
    always #5 clk = ~clk;
    reg resetn = 0;
    reg [15:0] awaddr, araddr;
    reg [31:0] wdata;
    reg awvalid = 0, wvalid = 0, arvalid = 0;
    wire awready, wready, arready, bvalid, rvalid;
    wire [1:0] bresp, rresp;
    wire [31:0] rdata;

    pinkprocess_classifier dut (
        .s_axi_aclk(clk), .s_axi_aresetn(resetn),
        .s_axi_awaddr(awaddr), .s_axi_awprot(3'b0),
        .s_axi_awvalid(awvalid), .s_axi_awready(awready),
        .s_axi_wdata(wdata), .s_axi_wstrb(4'hf), .s_axi_wvalid(wvalid), .s_axi_wready(wready),
        .s_axi_bresp(bresp), .s_axi_bvalid(bvalid), .s_axi_bready(1'b1),
        .s_axi_araddr(araddr), .s_axi_arprot(3'b0),
        .s_axi_arvalid(arvalid), .s_axi_arready(arready),
        .s_axi_rdata(rdata), .s_axi_rresp(rresp), .s_axi_rvalid(rvalid), .s_axi_rready(1'b1)
    );

    task send_address(input [15:0] address);
        @(negedge clk);
        awaddr = address;
        awvalid = 1;
        do @(posedge clk); while (!awready);
        @(negedge clk);
        awvalid = 0;
    endtask

    task send_data(input [31:0] value);
        @(negedge clk);
        wdata = value;
        wvalid = 1;
        do @(posedge clk); while (!wready);
        @(negedge clk);
        wvalid = 0;
    endtask

    task write_word(input [15:0] address, input [31:0] value);
        // Exercise both legal orders for the independent AW and W channels.
        if (address[2]) begin
            send_data(value);
            send_address(address);
        end else begin
            send_address(address);
            send_data(value);
        end
        do @(posedge clk); while (!bvalid);
        if (bresp !== 0) $fatal(1, "Write rejected at %h", address);
    endtask

    task read_word(input [15:0] address, output [31:0] value);
        @(negedge clk);
        araddr = address;
        arvalid = 1;
        do @(posedge clk); while (!arready);
        @(negedge clk);
        arvalid = 0;
        do @(posedge clk); while (!rvalid);
        if (rresp !== 0) $fatal(1, "Read rejected at %h", address);
        value = rdata;
    endtask

    integer file_handle, operation, scans, cycle;
    reg [15:0] address;
    reg [31:0] expected, actual;
    initial begin
        repeat (4) @(negedge clk);
        resetn = 1;
        file_handle = $fopen("stimulus.txt", "r");
        if (!file_handle) $fatal(1, "Could not open stimulus.txt");
        scans = $fscanf(file_handle, "%d %h %h", operation, address, expected);
        while (scans == 3) begin
            case (operation)
                0: write_word(address, expected);
                1: begin
                    actual = 0;
                    for (cycle = 0; cycle < 18000 && actual != expected; cycle = cycle + 1)
                        read_word(address, actual);
                    if (actual !== expected) $fatal(1, "Classifier did not finish");
                end
                2: begin
                    read_word(address, actual);
                    if (actual !== expected)
                        $fatal(1, "Score at %h: %h != %h", address, actual, expected);
                end
                default: $fatal(1, "Unknown operation %d", operation);
            endcase
            scans = $fscanf(file_handle, "%d %h %h", operation, address, expected);
        end
        if (!$feof(file_handle)) $fatal(1, "Malformed stimulus line");
        $fclose(file_handle);
        $finish;
    end

    initial begin
        #20_000_000;
        $fatal(1, "Simulation timed out");
    end
endmodule
