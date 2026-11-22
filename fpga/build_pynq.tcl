# Run with vivado -mode batch -source fpga/build_pynq.tcl -tclargs <build-dir>
if {$argc != 1} {
    error "Pass a new build directory with -tclargs."
}
set project_dir [file normalize [lindex $argv 0]]
set source_dir [file dirname [file normalize [info script]]]
set boards [lsort -dictionary [get_board_parts -quiet *pynq-z2*]]
if {[llength $boards] == 0} {
    error "Install the PYNQ-Z2 board preset in Vivado before building."
}
set board [lindex $boards end]
create_project pinkprocess $project_dir -part [get_property PART_NAME $board]
set_property board_part $board [current_project]
foreach rtl {classifier.v projection_tile.sv projection_engine.sv projection_dma.v pinkprocess_projection.v} {
    add_files [file join $source_dir $rtl]
}
update_compile_order -fileset sources_1
set block_design pinkprocess_bd
create_bd_design $block_design
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable"} \
    [get_bd_cells ps7]
set_property -dict [list CONFIG.PCW_USE_M_AXI_GP0 {1} CONFIG.PCW_USE_S_AXI_HP0 {1} \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {50}] [get_bd_cells ps7]

create_bd_cell -type module -reference pinkprocess_classifier classifier
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config {Master {/ps7/M_AXI_GP0} Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto}} \
    [get_bd_intf_pins classifier/S_AXI]
create_bd_cell -type module -reference pinkprocess_projection projection
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config {Master {/ps7/M_AXI_GP0} Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto}} \
    [get_bd_intf_pins projection/S_AXI]

# SmartConnect adapts the burst master to the Zynq AXI3 HP port.
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 projection_ddr
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] [get_bd_cells projection_ddr]
connect_bd_intf_net [get_bd_intf_pins projection/M_AXI] [get_bd_intf_pins projection_ddr/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins projection_ddr/M00_AXI] [get_bd_intf_pins ps7/S_AXI_HP0]
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins projection_ddr/aclk] [get_bd_pins ps7/S_AXI_HP0_ACLK]
connect_bd_net [get_bd_pins rst_ps7_50M/peripheral_aresetn] [get_bd_pins projection_ddr/aresetn]
assign_bd_address
validate_bd_design
save_bd_design

set design_file [get_files ${block_design}.bd]
generate_target all $design_file
add_files [make_wrapper -files $design_file -top]
set_property top ${block_design}_wrapper [current_fileset]
update_compile_order -fileset sources_1
launch_runs impl_1 -to_step write_bitstream
wait_on_run impl_1

open_run impl_1
report_timing_summary -file [file join $project_dir timing.rpt]
report_utilization -file [file join $project_dir utilization.rpt]
foreach delay_type {max min} {
    set worst [get_timing_paths -delay_type $delay_type -max_paths 1]
    if {![llength $worst] || [get_property SLACK $worst] < 0} {
        error "Projection overlay failed $delay_type timing; inspect timing.rpt."
    }
}
set bitstream [file join [get_property DIRECTORY [get_runs impl_1]] ${block_design}_wrapper.bit]
set handoffs [glob -nocomplain -directory $project_dir */sources_1/bd/$block_design/hw_handoff/${block_design}.hwh]
if {![file isfile $bitstream] || [llength $handoffs] == 0} {
    error "Vivado did not produce the bitstream and hardware handoff. Inspect the implementation log."
}
file copy -force $bitstream [file join $project_dir pinkprocess.bit]
file copy -force [lindex $handoffs 0] [file join $project_dir pinkprocess.hwh]
