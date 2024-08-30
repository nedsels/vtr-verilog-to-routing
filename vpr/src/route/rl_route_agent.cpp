//
// Created by shrevena on 15/07/24.
//

#include "globals.h"
#include "rl_route_agent.h"

template<size_t NumActions>
void parameter_data<NumActions>::update_probability_distribution() {
#if defined(SOFTMAX)
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
#elif defined(EPSILON_DELTA)
    std::vector<float> probabilities;
    probabilities.reserve(action_data_.size());
    for (ActionData& action : action_data_) {
        probabilities.emplace_back(action.prediction);
    }
#endif

    // Update probability distribution
    probability_distribution_ = std::discrete_distribution(probabilities.begin(), probabilities.end());
}

template<size_t NumActions>
void parameter_data<NumActions>::update_reward_predictions(float k_step_size, int reward) {
    // Update action reward prediction (equation 2.5)
    ActionData& action = action_data_[curr_action_index_];
    action.prediction += (k_step_size * ((float)reward - action.prediction));

#ifdef SOFTMAX
    // Update numerical preference for each action (equation 2.12)
    for (size_t i = 0; i < action_data_.size(); ++i) {
        ActionData& curr_action = action_data_[i];

        if (i == curr_action_index_)
            curr_action.preference += k_step_size * (reward - curr_action.prediction) * (1 - curr_action.probability);
        else
            curr_action.preference -= k_step_size * (reward - curr_action.prediction) * curr_action.probability;
    }
#endif

    update_probability_distribution();

    VTR_LOG("Updating probability distribution... ");
    for (double prob : probability_distribution_.probabilities()) {
        VTR_LOG("%.5f ", prob);
    }
    VTR_LOG("\n");
}

template<size_t NumActions>
void pres_fac_data<NumActions>::do_action(int itry) {
    if (itry > 1) {
        // Determine pres_fac action
        curr_action_index_ = probability_distribution_(generator_);
        VTR_LOG("Action %d selected. Changing pres_fac from %.5f to ", curr_action_index_, value_);

        switch (curr_action_index_) {
            case 0:
                value_ += pres_growth_fac_ * 0.1;
                break;
            case 1:
                value_ += pres_growth_fac_ * 0.05;
                break;
            case 2:
                value_ += pres_growth_fac_ * 0.01;
                break;
#if RLROUTE_IMPL == 1 || RLROUTE_IMPL == 7 || RLROUTE_IMPL == 9
            case 4:
                value_ -= pres_growth_fac_ * 0.05;
                break;
            case 5:
                value_ -= pres_growth_fac_ * 0.01;
                break;
            case 6:
                value_ -= pres_growth_fac_ * 0.1;
                break;
#elif RLROUTE_IMPL == 2
            case 4:
                value_ -= pres_growth_fac_ * 0.05;
                break;
            case 5:
                value_ -= pres_growth_fac_ * 0.01;
                break;
#elif RLROUTE_IMPL == 4
            case 4:
                value_ -= pres_growth_fac_ * 0.01;
                break;
#elif RLROUTE_IMPL == 3 || RLROUTE_IMPL == 5 || RLROUTE_IMPL == 6 || RLROUTE_IMPL == 8 || RLROUTE_IMPL == 10 || RLROUTE_IMPL == 11
            case 4:
                value_ -= pres_growth_fac_ * 0.05;
                break;
#endif
            case 3:
            default:
                VTR_ASSERT_SAFE(curr_action_index_ == 3);
                break;
        }

        // Clip pres_fac to upper and lower bounds
        if (value_ >= max_pres_fac_ || value_ <= min_pres_fac_) {
            curr_action_index_ = 3;
        }
        value_ = std::max(min_pres_fac_, value_);
        value_ = std::min(max_pres_fac_, value_);

        VTR_LOG("%.5f.\n", value_);
    }
}

template<size_t NumActions>
void astar_fac_data<NumActions>::do_action(int /*itry*/) {
    // Determine astar_fac action
    value_ = probability_distribution_(generator_);
    VTR_LOG("Action %d selected. Changing astar_fac from %.5f to ", curr_action_index_, value_);

    switch (curr_action_index_) {
        case 0:
            value_ = 0.8;
            break;
        case 1:
            value_ = 1.0;
            break;
        case 2:
            value_ = 1.2;
            break;
        case 3:
            value_ = 1.4;
            break;
        case 4:
            value_ = 1.6;
            break;
        case 5:
            value_ = 1.8;
            break;
        case 6:
            value_ = 2.0;
            break;
        case 7:
            value_ = 2.2;
            break;
        default:
            VTR_ASSERT(false);
            break;
    }

    VTR_LOG("%.5f.\n", value_);
}

template<size_t NumActions>
void criticality_exp_data<NumActions>::do_action(int /*itry*/) {
    // Determine criticality_exp action
    curr_action_index_ = probability_distribution_(generator_);
    VTR_LOG("Action %d selected. Changing criticality_exp from %.5f to ", curr_action_index_, value_);

    switch (curr_action_index_) {
        case 0:
            value_ += 0.1;
            break;
        case 1:
            value_ += 0.05;
            break;
        case 2:
            value_ += 0.01;
            break;
        case 4:
            value_ -= 0.01;
            break;
        case 5:
            value_ -= 0.05;
            break;
        case 6:
            value_ -= 0.1;
            break;
        case 3:
        default:
            VTR_ASSERT_SAFE(curr_action_index_ == 3);
            break;
    }

    if (value_ < 0) {
        value_ = 0;
        curr_action_index_ = 3;
    }

    VTR_LOG("%.5f.\n", value_);
}

template<size_t NumActions>
void pres_fac_data<NumActions>::update_after_sink_route(const RouteTree& tree,
                                                        std::vector<RRNodeId> sink_nodes,
                                                        vtr::vector<RRNodeId, int>& prev_cong,
                                                        vtr::vector<RRNodeId, int>& prev_length,
                                                        const RouterStats& stats,
                                                        float k_step_size,
                                                        int& prev_heap_ops,
                                                        const int itry,
                                                        float& length_fac) {
    if (itry <= 1)
        return;

    // Determine reward
    int reward = calculate_reward(tree, sink_nodes, prev_cong, prev_length, stats, prev_heap_ops, itry, length_fac);
    VTR_LOG("Reward: %d\n", reward);

    update_reward_predictions(k_step_size, reward);
}

template<size_t NumActions>
void astar_fac_data<NumActions>::update_after_sink_route(const RouteTree& tree,
                                                         std::vector<RRNodeId> sink_nodes,
                                                         vtr::vector<RRNodeId, int>& prev_cong,
                                                         vtr::vector<RRNodeId, int>& prev_length,
                                                         const RouterStats& stats,
                                                         float k_step_size,
                                                         int& prev_heap_ops,
                                                         const int itry,
                                                         float& length_fac) {
    // Determine reward
    int reward = calculate_reward(tree, sink_nodes, prev_cong, prev_length, stats, prev_heap_ops, itry, length_fac);
    VTR_LOG("Reward: %d\n", reward);

    update_reward_predictions(k_step_size, reward);
}

template<size_t NumActions>
void criticality_exp_data<NumActions>::update_after_sink_route(const RouteTree& tree,
                                                               std::vector<RRNodeId> sink_nodes,
                                                               vtr::vector<RRNodeId, int>& prev_cong,
                                                               vtr::vector<RRNodeId, int>& prev_length,
                                                               const RouterStats& stats,
                                                               float k_step_size,
                                                               int& prev_heap_ops,
                                                               const int itry,
                                                               float& length_fac) {
    // Determine reward
    int reward = calculate_reward(tree, sink_nodes, prev_cong, prev_length, stats, prev_heap_ops, itry, length_fac);
    reward = std::max(-25, reward);
    reward = std::min(25, reward);
    VTR_LOG("Reward: %d\n", reward);

    update_reward_predictions(k_step_size, reward);
}

template<size_t NumActions>
int pres_fac_data<NumActions>::calculate_reward(const RouteTree& tree,
                                                const std::vector<RRNodeId>& sink_nodes,
                                                vtr::vector<RRNodeId, int>& prev_cong,
                                                vtr::vector<RRNodeId, int>& prev_length,
                                                const RouterStats& stats,
                                                int& prev_heap_ops,
                                                int itry,
                                                float& length_fac) {
    const auto& rr_node_route_inf = g_vpr_ctx.routing().rr_node_route_inf;
    const auto& rr_graph = g_vpr_ctx.device().rr_graph;

    RRNodeId sink_node = sink_nodes[0];

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

    int cong_diff = curr_cong - prev_cong[sink_node];
    int length_diff = curr_length - prev_length[sink_node];

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
        reward = round(-log(length_diff));
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
        reward = round(-length_diff * length_fac);
        float LENGTH_FAC_GROWTH_RATE = 0.9;
        length_fac *= LENGTH_FAC_GROWTH_RATE;
    } else if (length_diff < 0 && cong_diff < 0) {
        reward = cong_diff * length_diff;
    } else {
        reward = 0;
    }
#else
    int reward = 0;
#endif

    prev_cong[sink_node] = curr_cong;
    prev_length[sink_node] = curr_length;

    return reward;
}

template<size_t NumActions>
int astar_fac_data<NumActions>::calculate_reward(const RouteTree& tree,
                                                 const std::vector<RRNodeId>& sink_nodes,
                                                 vtr::vector<RRNodeId, int>& prev_cong,
                                                 vtr::vector<RRNodeId, int>& prev_length,
                                                 const RouterStats& stats,
                                                 int& prev_heap_ops,
                                                 int itry,
                                                 float& length_fac) {
    const auto& rr_node_route_inf = g_vpr_ctx.routing().rr_node_route_inf;
    const auto& rr_graph = g_vpr_ctx.device().rr_graph;

    RRNodeId sink_node = sink_nodes[0];

    int net_size = 1;
    for (const RouteTreeNode& node : tree.all_nodes()) {
        (void)node;
        net_size += 1;
    }

    int total_heap_ops = stats.heap_pops + stats.heap_pushes;

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

    int cong_diff = curr_cong - prev_cong[sink_node];
    int length_diff = curr_length - prev_length[sink_node];

    VTR_LOG("num_ripup_nodes: %d, total_heap_ops: %d, new_heap_ops: %d, length_diff: %d\n", net_size, total_heap_ops, total_heap_ops - prev_heap_ops, length_diff);

#if RLROUTE_IMPL == 12
    int reward = -round((total_heap_ops - prev_heap_ops) / net_size / 10.0);
#elif RLROUTE_IMPL == 13
    int reward = rand();
#elif RLROUTE_IMPL == 14
    int reward = -round((total_heap_ops - prev_heap_ops) / net_size / 10.0);
    if (itry > 1) {
        reward *= round(length_diff / 10.0);
    }
#elif RLROUTE_IMPL == 15
    int reward = -round((total_heap_ops - prev_heap_ops) / net_size / 10.0);
    if (itry > 1) {
        if (length_diff > 0) {
            reward = round(-length_diff * length_fac);
            float LENGTH_FAC_GROWTH_RATE = 0.9;
            length_fac *= LENGTH_FAC_GROWTH_RATE;
        } else {
            reward *= length_diff / 10.0;
        }
    }
#elif RLROUTE_IMPL == 16
    int reward;
    if (itry > 1) {
        reward = -length_diff;
    } else {
        reward = 0;
    }
#else
    int reward = 0;
#endif

    prev_heap_ops = total_heap_ops;

#if (RLROUTE_IMPL >= 12 && RLROUTE_IMPL <= 16)
    prev_cong[sink_node] = curr_cong;
    prev_length[sink_node] = curr_length;
#endif

    return reward;
}

template<size_t NumActions>
int criticality_exp_data<NumActions>::calculate_reward(const RouteTree& tree,
                                                       const std::vector<RRNodeId>& sink_nodes,
                                                       vtr::vector<RRNodeId, int>& prev_cong,
                                                       vtr::vector<RRNodeId, int>& prev_length,
                                                       const RouterStats& stats,
                                                       int& prev_heap_ops,
                                                       int itry,
                                                       float& length_fac) {
    const auto& rr_node_route_inf = g_vpr_ctx.routing().rr_node_route_inf;
    const auto& rr_graph = g_vpr_ctx.device().rr_graph;

    int total_heap_ops = stats.heap_pops + stats.heap_pushes;

    int net_size = 1;
    for (const RouteTreeNode& node : tree.all_nodes()) {
        (void)node;
        net_size += 1;
    }

    int cong_diff = 0;
    int length_diff = 0;

    for (RRNodeId sink_node : sink_nodes) {
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

        cong_diff += curr_cong - prev_cong[sink_node];
        length_diff += curr_length - prev_length[sink_node];

        prev_cong[sink_node] = curr_cong;
        prev_length[sink_node] = curr_length;
    }

    VTR_LOG("net_size: %d, total_heap_ops: %d, new_heap_ops: %d, length_diff: %d\n", net_size, total_heap_ops, total_heap_ops - prev_heap_ops, length_diff);

#if RLROUTE_IMPL == 17
    int reward = -round((total_heap_ops - prev_heap_ops) / sink_nodes.size() / net_size);
#elif RLROUTE_IMPL == 18
    int reward;
    if (itry > 1) {
        reward = -log((total_heap_ops - prev_heap_ops) / sink_nodes.size() / net_size);
    } else {
        reward = 0;
    }
#elif RLROUTE_IMPL == 19
    int reward;
    if (itry > 1) {
        reward = -length_diff;
    } else {
        reward = 0;
    }
#else
    int reward = 0;
#endif

    prev_heap_ops = total_heap_ops;

    return reward;
}

RLRouteAgent::RLRouteAgent(const t_router_opts& router_opts)
    : itry_(0)
    , k_step_size_(0.1f)
    , length_fac_(2.0f)
    , prev_heap_ops_(0) {
#if defined(SOFTMAX) && defined(EPSILON_DELTA)
    throw vtr::VtrError("Define either SOFTMAX or EPSILON_DELTA, not both.", "rl_route_agent.cpp", 522);
#endif

#if !(defined(SOFTMAX) || defined(EPSILON_DELTA))
    throw vtr::VtrError("Define either SOFTMAX or EPSILON_DELTA.", "rl_route_agent.cpp", 526);
#endif

#if RLROUTE_IMPL <= 11
    pres_fac_data_.value_ = router_opts.first_iter_pres_fac;
    pres_fac_data_.initial_pres_fac_ = router_opts.initial_pres_fac;
    pres_fac_data_.pres_fac_mult_ = router_opts.pres_fac_mult;
    pres_fac_data_.min_pres_fac_ = 0.5f;
    pres_fac_data_.max_pres_fac_ = std::numeric_limits<float>::max();
    pres_fac_data_.pres_growth_fac_ = std::numeric_limits<float>::max();
#elif (RLROUTE_IMPL >= 12 && RLROUTE_IMPL <= 16)
    astar_fac_data_.value_ = router_opts.astar_fac;
#elif (RLROUTE_IMPL >= 17 && RLROUTE_IMPL <= 19)
    criticality_exp_data_.value_ = router_opts.criticality_exp;
#endif

    prev_cong_.resize(g_vpr_ctx.device().rr_graph.num_nodes(), 0.0);
    prev_length_.resize(g_vpr_ctx.device().rr_graph.num_nodes(), 0.0);

    update_probability_distribution();
}

void RLRouteAgent::update_per_iteration(float pres_fac, int itry) {
    itry_ = itry;

#if RLROUTE_IMPL <= 11
    if (itry_ <= 1) {
        return;
    }

    // Update pres_fac
    pres_fac_data_.value_ = pres_fac;
    VTR_LOG("initial pres_fac for this iteration: %.3f\n", pres_fac);

    // How much pres_fac would grow normally per iteration
    pres_fac_data_.pres_growth_fac_ = (pres_fac * pres_fac_data_.pres_fac_mult_) - pres_fac;

    // Set upper and lower bounds for pres_fac for this iteration
    const size_t k_expansion_factor = 5; // arbitrary
    const float pres_fac_range = pres_fac_data_.pres_growth_fac_ * k_expansion_factor;

    pres_fac_data_.max_pres_fac_ = pres_fac + pres_fac_range;
    pres_fac_data_.min_pres_fac_ = std::max(pres_fac - pres_fac_range, pres_fac_data_.initial_pres_fac_);

#elif (RLROUTE_IMPL >= 12 && RLROUTE_IMPL <= 16)
    if (itry >= 2) {
        astar_fac_data_.action_data_ = std::array<ActionData, 8>();
        astar_fac_data_.update_probability_distribution();
    }

#endif
    // Router stats are reset every iteration
    prev_heap_ops_ = 0;
}

void RLRouteAgent::do_action() {
#if RLROUTE_IMPL <= 11
    pres_fac_data_.do_action(itry_);
#elif (RLROUTE_IMPL >= 12 && RLROUTE_IMPL <= 16)
    astar_fac_data_.do_action(itry_);
#elif (RLROUTE_IMPL >= 17 && RLROUTE_IMPL <= 19)
    criticality_exp_data_.do_action(itry_);
#endif
}

void RLRouteAgent::update_probability_distribution() {
#if RLROUTE_IMPL <= 11
    pres_fac_data_.update_probability_distribution();
#elif (RLROUTE_IMPL >= 12 && RLROUTE_IMPL <= 16)
    astar_fac_data_.update_probability_distribution();
#elif (RLROUTE_IMPL >= 17 && RLROUTE_IMPL <= 19)
    criticality_exp_data_.update_probability_distribution();
#endif
}

void RLRouteAgent::update_after_sink_route(const RouteTree& tree, std::vector<RRNodeId> sink_nodes, const RouterStats& stats) {
#if RLROUTE_IMPL <= 11
    pres_fac_data_.update_after_sink_route(tree, sink_nodes, prev_cong_, prev_length_, stats, k_step_size_, prev_heap_ops_, itry_, length_fac_);
#elif (RLROUTE_IMPL >= 12 && RLROUTE_IMPL <= 16)
    astar_fac_data_.update_after_sink_route(tree, sink_nodes, prev_cong_, prev_length_, stats, k_step_size_, prev_heap_ops_, itry_, length_fac_);
#elif (RLROUTE_IMPL >= 17 && RLROUTE_IMPL <= 19)
    criticality_exp_data_.update_after_sink_route(tree, sink_nodes, prev_cong_, prev_length_, stats, k_step_size_, prev_heap_ops_, itry_, length_fac_);
#endif
}

float RLRouteAgent::pres_fac() {
    return pres_fac_data_.value_;
}

float RLRouteAgent::astar_fac() {
    return astar_fac_data_.value_;
}

float RLRouteAgent::criticality_exp() {
    return criticality_exp_data_.value_;
}
