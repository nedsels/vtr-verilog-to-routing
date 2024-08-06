//
// Created by shrevena on 15/07/24.
//

#ifndef VTR_RL_ROUTE_AGENT_H
#define VTR_RL_ROUTE_AGENT_H

#define RLROUTE_IMPL 11

#include <array>
#include <random>

#include "route_tree.h"
#include "vpr_types.h"

class RLRouteAgent {
  public:
    RLRouteAgent(const t_router_opts& router_opts);

    RLRouteAgent(const RLRouteAgent&) = delete;
    void operator=(const RLRouteAgent&) = delete;

    void update_per_iteration(float pres_fac, int itry);
    size_t do_action();
    void update_after_sink_route(const RouteTree& tree, RRNodeId sink_node);

    float pres_fac();

  private:
    struct ActionData {
        // The reward prediction for the action (ie. Q(a))
        float prediction = 0.;
        // The numerical preference for the action (ie. H(a), section 2.8)
        float preference = 0.;
        // The probability of selecting this action (calculate using eq. 2.11)
        float probability = 0.;
    };

    void update_probability_distribution();
    int calculate_reward(const RouteTree& tree, RRNodeId sink_node);

    int itry_;
    float pres_fac_;
    float initial_pres_fac_;
    const float pres_fac_mult_;
    float min_pres_fac_;
    float max_pres_fac_;
    float pres_growth_fac_;
    const float k_step_size_;
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
    vtr::vector<RRNodeId, int> prev_cong_;
    vtr::vector<RRNodeId, int> prev_length_;
    float length_fac_;
};

#endif //VTR_RL_ROUTE_AGENT_H
