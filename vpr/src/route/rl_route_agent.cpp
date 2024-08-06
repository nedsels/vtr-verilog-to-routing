//
// Created by shrevena on 15/07/24.
//

#include "globals.h"
#include "rl_route_agent.h"

RLRouteAgent::RLRouteAgent(const t_router_opts& router_opts)
    : itry_(0)
    , pres_fac_(router_opts.first_iter_pres_fac)
    , initial_pres_fac_(router_opts.initial_pres_fac)
    , pres_fac_mult_(router_opts.pres_fac_mult)
    , k_step_size_(0.1f)
    , length_fac_(2.0f) {
    prev_cong_.resize(g_vpr_ctx.device().rr_graph.num_nodes(), 0.0);
    prev_length_.resize(g_vpr_ctx.device().rr_graph.num_nodes(), 0.0);

    update_probability_distribution();
}

void RLRouteAgent::update_per_iteration(float pres_fac, int itry) {
    itry_ = itry;

    if (itry_ <= 1) {
        return;
    }

    // Update pres_fac
    pres_fac_ = pres_fac;
    // VTR_LOG("initial pres_fac for this iteration: %.3f\n", pres_fac);

    // How much pres_fac would grow normally per iteration
    pres_growth_fac_ = (pres_fac * pres_fac_mult_) - pres_fac;

    // Set upper and lower bounds for pres_fac for this iteration
    const size_t k_expansion_factor = 5; // arbitrary
    const float pres_fac_range = pres_growth_fac_ * k_expansion_factor;

    max_pres_fac_ = pres_fac + pres_fac_range;
    min_pres_fac_ = std::max(pres_fac - pres_fac_range, initial_pres_fac_);
}

size_t RLRouteAgent::do_action() {
    if (itry_ <= 1) {
        return -1;
    }

    // Determine action
    size_t action_index = probability_distribution_(generator_);
    // VTR_LOG("Action %d selected. Changing pres_fac from %.5f to ", action_index, pres_fac_);

    // Do action
    switch (action_index) {
        case 0:
            pres_fac_ += pres_growth_fac_ * 0.1;
            break;
        case 1:
            pres_fac_ += pres_growth_fac_ * 0.05;
            break;
        case 2:
            pres_fac_ += pres_growth_fac_ * 0.01;
            break;
#if RLROUTE_IMPL == 1 || RLROUTE_IMPL == 7 || RLROUTE_IMPL == 9
        case 4:
            pres_fac_ -= pres_growth_fac_ * 0.05;
            break;
        case 5:
            pres_fac_ -= pres_growth_fac_ * 0.01;
            break;
        case 6:
            pres_fac_ -= pres_growth_fac_ * 0.1;
            break;
#elif RLROUTE_IMPL == 2
        case 4:
            pres_fac_ -= pres_growth_fac_ * 0.05;
            break;
        case 5:
            pres_fac_ -= pres_growth_fac_ * 0.01;
            break;
#elif RLROUTE_IMPL == 4
        case 4:
            pres_fac_ -= pres_growth_fac_ * 0.01;
            break;
#elif RLROUTE_IMPL == 3 || RLROUTE_IMPL == 5 || RLROUTE_IMPL == 6 || RLROUTE_IMPL == 8 || RLROUTE_IMPL == 10 || RLROUTE_IMPL == 11
        case 4:
            pres_fac_ -= pres_growth_fac_ * 0.05;
            break;
#endif
        case 3:
        default:
            VTR_ASSERT_SAFE(action_index == 3);
            break;
    }

    // Clip pres_fac to upper and lower bounds
    if (pres_fac_ >= max_pres_fac_ || pres_fac_ <= min_pres_fac_) {
        action_index = 3;
    }
    pres_fac_ = std::max(min_pres_fac_, pres_fac_);
    pres_fac_ = std::min(max_pres_fac_, pres_fac_);

    // VTR_LOG("%.5f.\n", pres_fac_);

    curr_action_index_ = action_index;
    return action_index;
}

void RLRouteAgent::update_probability_distribution() {
    // Calculate denominator
    float softmax_denom_sum = 0.0;
    for (const ActionData& action : action_data_) {
        softmax_denom_sum += exp(action.preference);
    }

    // Get all probabilities using softmax formula (equation 2.11)
    std::vector<float> probabilities;
    for (ActionData& action : action_data_) {
        action.probability = exp(action.preference) / softmax_denom_sum;
        probabilities.emplace_back(action.probability);
    }

    // Update probability distribution
    probability_distribution_ = std::discrete_distribution(probabilities.begin(), probabilities.end());
}

void RLRouteAgent::update_after_sink_route(const RouteTree& tree, RRNodeId sink_node) {
    if (itry_ <= 1) {
        return;
    }

    // Determine reward
    int reward = calculate_reward(tree, sink_node);
    if (reward) {
        // VTR_LOG("Reward: %d\n", reward);
    }

    // Update action reward prediction (equation 2.5)
    ActionData& action = action_data_[curr_action_index_];
    action.prediction += (k_step_size_ * ((float)reward - action.prediction));

    // Update numerical preference for each action (equation 2.12)
    for (size_t i = 0; i < action_data_.size(); ++i) {
        ActionData& curr_action = action_data_[i];

        if (i == curr_action_index_)
            curr_action.preference += k_step_size_ * (reward - curr_action.prediction) * (1 - curr_action.probability);
        else
            curr_action.preference -= k_step_size_ * (reward - curr_action.prediction) * curr_action.probability;
    }

    update_probability_distribution();

    if (reward) {
        // VTR_LOG("Updating probability distribution... ");
        for (double prob : probability_distribution_.probabilities()) {
            // VTR_LOG("%.2f ", prob);
        }
        // VTR_LOG("\n");
    }
}

float RLRouteAgent::pres_fac() {
    return pres_fac_;
}

int RLRouteAgent::calculate_reward(const RouteTree& tree, RRNodeId sink_node) {
    const auto& rr_node_route_inf = g_vpr_ctx.routing().rr_node_route_inf;
    const auto& rr_graph = g_vpr_ctx.device().rr_graph;

    int curr_cong = 0;
    int curr_length = 0;

    // Traverse backward from sink and calculate congestion and wirelength
    VTR_ASSERT(tree.find_by_rr_id(sink_node).has_value());
    VTR_ASSERT(rr_graph.node_type(tree.find_by_rr_id(sink_node)->inode) == SINK);

    tl::optional<const RouteTreeNode&> tree_node = tree.find_by_rr_id(sink_node)->parent();
    while (tree_node->parent().has_value()) {
        RRNodeId node_id = tree_node->inode;

        int overuse = rr_node_route_inf[node_id].occ() - rr_graph.node_capacity(node_id);
        curr_cong += std::max(0, overuse);

        if (is_chan(rr_graph.node_type(node_id)))
            curr_length += rr_graph.node_length(node_id);

        tree_node = tree_node->parent();
    }

    VTR_ASSERT(rr_graph.node_type(tree_node->inode) == SOURCE);

    int cong_diff = curr_cong - prev_cong_[sink_node];
    int length_diff = curr_length - prev_length_[sink_node];

#if (RLROUTE_IMPL >= 1 && RLROUTE_IMPL <= 4)
    // FORMULA #1
    int reward = (cong_diff <= 0) ? -cong_diff : 0;

#elif RLROUTE_IMPL == 5
    // FORMULA #2
    int reward = (length_diff <= 0) ? -length_diff : 0;

#elif RLROUTE_IMPL == 6
    // FORMULA #3
    int reward = -length_diff;

#elif RLROUTE_IMPL == 7 || RLROUTE_IMPL == 8
    // FORMULA #4
    int reward;
    if (length_diff > 0 && cong_diff > 0) {
        reward = (int)-log(length_diff);
        float LENGTH_FAC_GROWTH_RATE = 0.9;
        length_fac_ *= LENGTH_FAC_GROWTH_RATE;
    } else if (length_diff < 0 && cong_diff < 0) {
        reward = cong_diff * length_diff;
    } else {
        reward = 0;
    }

#elif RLROUTE_IMPL == 9 || RLROUTE_IMPL == 10
    // FORMULA #5
    int reward = rand();

#elif RLROUTE_IMPL == 11
    // FORMULA #6
    int reward;
    if (length_diff > 0 && cong_diff > 0) {
        reward = (int)(-length_diff * length_fac_);
        float LENGTH_FAC_GROWTH_RATE = 0.9;
        length_fac_ *= LENGTH_FAC_GROWTH_RATE;
    } else if (length_diff < 0 && cong_diff < 0) {
        reward = cong_diff * length_diff;
    } else {
        reward = 0;
    }
#endif

    prev_cong_[sink_node] = curr_cong;
    prev_length_[sink_node] = curr_length;

    return reward;
}