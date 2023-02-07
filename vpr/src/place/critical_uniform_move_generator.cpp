#include "critical_uniform_move_generator.h"
#include "globals.h"
#include "place_constraints.h"
#include "move_utils.h"

e_create_move CriticalUniformMoveGenerator::propose_move(t_pl_blocks_to_be_moved& blocks_affected, e_move_type& /*move_type*/, t_logical_block_type& blk_type, float rlim, const t_placer_opts& /*placer_opts*/, const PlacerCriticalities* /*criticalities*/) {
    auto& place_ctx = g_vpr_ctx.placement();
    auto& cluster_ctx = g_vpr_ctx.clustering();

    ClusterNetId net_from;
    ClusterBlockId b_from;
    int pin_from;

    if (blk_type.index == -1) { //If the block type is unspecified, choose any random highly critical block to be swapped with another random block
        b_from = pick_from_highly_critical_block(net_from, pin_from);
        if (!b_from) {
            return e_create_move::ABORT; //No movable block found
        }
        blk_type.index = convert_logical_to_agent_block_type(cluster_ctx.clb_nlist.block_type(b_from)->index);
    } else { //If the block type is specified, choose a random highly critical with blk_type to be swapped with another random block
        b_from = pick_from_highly_critical_block(net_from, pin_from, blk_type);
    }

    t_pl_loc from = place_ctx.block_locs[b_from].loc;
    auto cluster_from_type = cluster_ctx.clb_nlist.block_type(b_from);
    auto grid_from_type = g_vpr_ctx.device().grid[from.x][from.y].type;
    VTR_ASSERT(is_tile_compatible(grid_from_type, cluster_from_type));

    t_pl_loc to;

    if (!find_to_loc_uniform(cluster_from_type, rlim, from, to, b_from)) {
        return e_create_move::ABORT;
    }

    e_create_move create_move = ::create_move(blocks_affected, b_from, to);

    //Check that all of the blocks affected by the move would still be in a legal floorplan region after the swap
    if (!floorplan_legal(blocks_affected)) {
        return e_create_move::ABORT;
    }

    return create_move;
}
