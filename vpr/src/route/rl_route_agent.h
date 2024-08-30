//
// Created by shrevena on 15/07/24.
//

#ifndef VTR_RL_ROUTE_AGENT_H
#define VTR_RL_ROUTE_AGENT_H

#define SOFTMAX
//#define EPSILON_DELTA

#define RLROUTE_IMPL 19

#include <array>
#include <random>

#include "route_tree.h"
#include "router_stats.h"
#include "vpr_types.h"

struct ActionData {
#if defined(SOFTMAX)
    // The reward prediction for the action (ie. Q(a))
    float prediction = 0.;
    // The numerical preference for the action (ie. H(a), section 2.8)
    float preference = 0.;
    // The probability of selecting this action (calculate using eq. 2.11)
    float probability = 0.;
#elif defined(EPSILON_DELTA)
    // The reward prediction for the action (ie. Q(a)), with optimistic initial value
    float prediction = 5.;
#endif
};

template<size_t NumActions>
class parameter_data {
  public:
    parameter_data()
        : value_(0.)
        , curr_action_index_(0) {}

    float value_;

    std::array<ActionData, NumActions> action_data_;
    size_t curr_action_index_;
    std::default_random_engine generator_;
    std::discrete_distribution<> probability_distribution_;

    void update_probability_distribution();
    void update_reward_predictions(float k_step_size, int reward);

    virtual void do_action(int itry) = 0;
    virtual int calculate_reward(const RouteTree& tree,
                                 const std::vector<RRNodeId>& sink_nodes,
                                 vtr::vector<RRNodeId, int>& prev_cong,
                                 vtr::vector<RRNodeId, int>& prev_length,
                                 const RouterStats& stats,
                                 int& prev_heap_ops,
                                 int itry,
                                 float& length_fac)
        = 0;
    virtual void update_after_sink_route(const RouteTree& tree,
                                         std::vector<RRNodeId> sink_nodes,
                                         vtr::vector<RRNodeId, int>& prev_cong,
                                         vtr::vector<RRNodeId, int>& prev_length,
                                         const RouterStats& stats,
                                         float k_step_size,
                                         int& prev_heap_ops,
                                         const int itry,
                                         float& length_fac)
        = 0;

    virtual ~parameter_data() = default;
};

template<size_t NumActions>
class pres_fac_data : public parameter_data<NumActions> {
  public:
    using parameter_data<NumActions>::value_;
    using parameter_data<NumActions>::action_data_;
    using parameter_data<NumActions>::curr_action_index_;
    using parameter_data<NumActions>::generator_;
    using parameter_data<NumActions>::probability_distribution_;
    using parameter_data<NumActions>::update_probability_distribution;
    using parameter_data<NumActions>::update_reward_predictions;

    pres_fac_data()
        : parameter_data<NumActions>()
        , initial_pres_fac_(0.)
        , pres_fac_mult_(0.)
        , min_pres_fac_(0.)
        , max_pres_fac_(0.)
        , pres_growth_fac_(0.) {}

    float initial_pres_fac_;
    float pres_fac_mult_;
    float min_pres_fac_;
    float max_pres_fac_;
    float pres_growth_fac_;

    void do_action(int itry) override;
    int calculate_reward(const RouteTree& tree,
                         const std::vector<RRNodeId>& sink_nodes,
                         vtr::vector<RRNodeId, int>& prev_cong,
                         vtr::vector<RRNodeId, int>& prev_length,
                         const RouterStats& stats,
                         int& prev_heap_ops,
                         int itry,
                         float& length_fac) override;
    void update_after_sink_route(const RouteTree& tree,
                                 std::vector<RRNodeId> sink_nodes,
                                 vtr::vector<RRNodeId, int>& prev_cong,
                                 vtr::vector<RRNodeId, int>& prev_length,
                                 const RouterStats& stats,
                                 float k_step_size,
                                 int& prev_heap_ops,
                                 const int itry,
                                 float& length_fac) override;
};

template<size_t NumActions>
class astar_fac_data : public parameter_data<NumActions> {
  public:
    using parameter_data<NumActions>::value_;
    using parameter_data<NumActions>::action_data_;
    using parameter_data<NumActions>::curr_action_index_;
    using parameter_data<NumActions>::generator_;
    using parameter_data<NumActions>::probability_distribution_;
    using parameter_data<NumActions>::update_probability_distribution;
    using parameter_data<NumActions>::update_reward_predictions;

    astar_fac_data()
        : parameter_data<NumActions>() {}

    void do_action(int itry) override;
    int calculate_reward(const RouteTree& tree,
                         const std::vector<RRNodeId>& sink_nodes,
                         vtr::vector<RRNodeId, int>& prev_cong,
                         vtr::vector<RRNodeId, int>& prev_length,
                         const RouterStats& stats,
                         int& prev_heap_ops,
                         int itry,
                         float& length_fac) override;
    void update_after_sink_route(const RouteTree& tree,
                                 std::vector<RRNodeId> sink_nodes,
                                 vtr::vector<RRNodeId, int>& prev_cong,
                                 vtr::vector<RRNodeId, int>& prev_length,
                                 const RouterStats& stats,
                                 float k_step_size,
                                 int& prev_heap_ops,
                                 const int itry,
                                 float& length_fac) override;
};

template<size_t NumActions>
class criticality_exp_data : public parameter_data<NumActions> {
  public:
    using parameter_data<NumActions>::value_;
    using parameter_data<NumActions>::action_data_;
    using parameter_data<NumActions>::curr_action_index_;
    using parameter_data<NumActions>::generator_;
    using parameter_data<NumActions>::probability_distribution_;
    using parameter_data<NumActions>::update_probability_distribution;
    using parameter_data<NumActions>::update_reward_predictions;

    criticality_exp_data()
        : parameter_data<NumActions>() {}

    void do_action(int itry) override;
    int calculate_reward(const RouteTree& tree,
                         const std::vector<RRNodeId>& sink_nodes,
                         vtr::vector<RRNodeId, int>& prev_cong,
                         vtr::vector<RRNodeId, int>& prev_length,
                         const RouterStats& stats,
                         int& prev_heap_ops,
                         int itry,
                         float& length_fac) override;
    void update_after_sink_route(const RouteTree& tree,
                                 std::vector<RRNodeId> sink_nodes,
                                 vtr::vector<RRNodeId, int>& prev_cong,
                                 vtr::vector<RRNodeId, int>& prev_length,
                                 const RouterStats& stats,
                                 float k_step_size,
                                 int& prev_heap_ops,
                                 const int itry,
                                 float& length_fac) override;
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

#if (RLROUTE_IMPL >= 3 && RLROUTE_IMPL <= 6) || RLROUTE_IMPL == 8 || RLROUTE_IMPL == 10 || RLROUTE_IMPL == 11
    pres_fac_data<5> pres_fac_data_;
#elif RLROUTE_IMPL == 2
    pres_fac_data<6> pres_fac_data_;
#else
    pres_fac_data<7> pres_fac_data_;
#endif
    astar_fac_data<8> astar_fac_data_;
    criticality_exp_data<7> criticality_exp_data_;
};

#endif //VTR_RL_ROUTE_AGENT_H
