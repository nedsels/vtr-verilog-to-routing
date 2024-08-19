//
// Created by shrevena on 15/07/24.
//

#ifndef VTR_RL_ROUTE_AGENT_H
#define VTR_RL_ROUTE_AGENT_H

#define RLROUTE_IMPL 18

#include <array>
#include <random>

#include "route_tree.h"
#include "router_stats.h"
#include "vpr_types.h"

struct ActionData {
    // The reward prediction for the action (ie. Q(a))
    float prediction = 0.;
    // The numerical preference for the action (ie. H(a), section 2.8)
    float preference = 0.;
    // The probability of selecting this action (calculate using eq. 2.11)
    float probability = 0.;
};

struct pres_fac_data {
    float pres_fac_;
    float initial_pres_fac_;
    float pres_fac_mult_;
    float min_pres_fac_;
    float max_pres_fac_;
    float pres_growth_fac_;

#if (RLROUTE_IMPL >= 3 && RLROUTE_IMPL <= 6) || RLROUTE_IMPL == 8 || RLROUTE_IMPL == 10 || RLROUTE_IMPL == 11
    std::array<ActionData, 5> action_data_;
#elif RLROUTE_IMPL == 2
    std::array<ActionData, 6> action_data_;
#else
    std::array<ActionData, 7> action_data_;
#endif

    size_t curr_action_index_;
    std::default_random_engine generator_;
    std::discrete_distribution<> probability_distribution_;

    int calculate_reward(const RouteTree& tree,
                         RRNodeId sink_node,
                         vtr::vector<RRNodeId, int>& prev_cong,
                         vtr::vector<RRNodeId, int>& prev_length,
                         float& length_fac);
    void update_probability_distribution();
    void update_after_sink_route(const RouteTree& tree,
                                 RRNodeId sink_node,
                                 vtr::vector<RRNodeId, int>& prev_cong,
                                 vtr::vector<RRNodeId, int>& prev_length,
                                 float k_step_size,
                                 float& length_fac);
};

struct astar_fac_data {
    float astar_fac_;

    std::array<ActionData, 8> action_data_;

    size_t curr_action_index_;
    std::default_random_engine generator_;
    std::discrete_distribution<> probability_distribution_;

    int calculate_reward(const RouteTree& tree,
                         RRNodeId sink_node,
                         vtr::vector<RRNodeId, int>& prev_cong,
                         vtr::vector<RRNodeId, int>& prev_length,
                         const RouterStats& stats,
                         int& prev_heap_ops,
                         int itry,
                         float& length_fac);
    void update_probability_distribution();
    void update_after_sink_route(const RouteTree& tree,
                                 RRNodeId sink_node,
                                 vtr::vector<RRNodeId, int>& prev_cong,
                                 vtr::vector<RRNodeId, int>& prev_length,
                                 const RouterStats& stats,
                                 float k_step_size,
                                 int& prev_heap_ops,
                                 const int itry,
                                 float& length_fac);
};

struct criticality_exp_data {
    float criticality_exp_;

    std::array<ActionData, 7> action_data_;

    size_t curr_action_index_;
    std::default_random_engine generator_;
    std::discrete_distribution<> probability_distribution_;

    int calculate_reward(const RouteTree& tree,
                         std::vector<RRNodeId> sink_nodes,
                         vtr::vector<RRNodeId, int>& prev_cong,
                         vtr::vector<RRNodeId, int>& prev_length,
                         const RouterStats& stats,
                         int& prev_heap_ops,
                         int itry,
                         float& length_fac);
    void update_probability_distribution();
    void update_after_sink_route(const RouteTree& tree,
                                 std::vector<RRNodeId> sink_nodes,
                                 vtr::vector<RRNodeId, int>& prev_cong,
                                 vtr::vector<RRNodeId, int>& prev_length,
                                 const RouterStats& stats,
                                 float k_step_size,
                                 int& prev_heap_ops,
                                 const int itry,
                                 float& length_fac);
};

class RLRouteAgent {
  public:
    RLRouteAgent(const t_router_opts& router_opts);

    RLRouteAgent(const RLRouteAgent&) = delete;
    void operator=(const RLRouteAgent&) = delete;

    void update_per_iteration(float pres_fac, int itry);
    void do_action();
    void update_after_sink_route(const RouteTree& tree, std::vector<RRNodeId> sink_nodes, const RouterStats& stats);

    float pres_fac();
    float astar_fac();
    float criticality_exp();

  private:
    void update_probability_distribution();

    int itry_;

    const float k_step_size_;
    float length_fac_;

    vtr::vector<RRNodeId, int> prev_cong_;
    vtr::vector<RRNodeId, int> prev_length_;
    int prev_heap_ops_;

    pres_fac_data pres_fac_data_;
    astar_fac_data astar_fac_data_;
    criticality_exp_data criticality_exp_data_;
};

#endif //VTR_RL_ROUTE_AGENT_H
