
#include <unordered_map>
#include <vector>
#include <random>
#include <algorithm>
#include "IEnvironment.hpp"



template <typename ACTION, typename STATE, typename REWARD>
class Agent{
    
    private:
        uint16_t num_episodes = 5000;
        double alpha = 0.1;
        double gamma = 0.9;
        double epsilon = 0.9;
        double initial_epsilon = 0.9;

        std::vector<REWARD> rewards_all_episodes;

        std::random_device rd;
        std::mt19937 mt;
        std::uniform_real_distribution<double> real_dist;
        std::uniform_int_distribution<int> int_dist;

        double** Q;
        STATE* states;
        ACTION* actions;
        size_t actions_count = 0, states_count = 0;

        STATE current_state;

        ACTION pick_action(const STATE& state){
            if(real_dist(mt) < epsilon){
                // chooses in random an action
                size_t idx = int_dist(mt);
                return actions[idx];
            }else{
                size_t state_idx = state_index(state);
                double* p = Q[state_idx];
                size_t action_idx = find_max_probability_index(p);
                return actions[action_idx];
            }
        }

        size_t action_index(const ACTION& act){
            size_t idx = 0;
            for(size_t i = 0; i < actions_count; i++){
                if(act == actions[i]) return idx;
                idx++;
            }
            return idx;
        }
        
        size_t state_index(const STATE& state){
            size_t idx = 0;
            for(size_t i = 0; i < states_count; i++){
                if(state == states[i]) return idx;
                idx++;
            }
            return idx;
        }

        size_t find_max_probability_index(const double* arr){
            size_t idx = 0;
            double max_prob = 0.0;

            for(size_t i = 0; i < actions_count; i++){
                if(arr[i] >= max_prob){
                    max_prob = arr[i];
                    idx = i;
                }
            }
            return idx;
        }

        double find_max_probability(const double* arr){
            double max_prob = 0.0;

            for(size_t i = 0; i < actions_count; i++){
                if(arr[i] >= max_prob){
                    max_prob = arr[i];
                }
            }
            return max_prob;
        }

    public:

        Agent(ACTION* _actions, size_t _actions_count, STATE* _states, size_t _states_count)
        : mt(rd()), real_dist(0.0, 1.0), int_dist(0, (int)_actions_count),
        states(_states), actions(_actions), actions_count(_actions_count), states_count(_states_count)
        {
            Q = (double**)malloc(states_count * sizeof(double*));
            for(size_t i = 0; i < states_count; i++){
                Q[i] = (double*)malloc(actions_count * sizeof(double));
                for(size_t j = 0; j < actions_count; j++){
                    Q[i][j] = 0.0;
                }
            }


        }

        ~Agent(){
            for(size_t i = 0; i < states_count; i++){
                free(Q[i]);
            }
            free(Q);            
        }

        void prepare_for_training(IEnvironment<ACTION, STATE>* env){
            current_state = env->get_initial_state();

            rewards_all_episodes.clear();

            if(rewards_all_episodes.capacity() < num_episodes){
                rewards_all_episodes.reserve(num_episodes);
            }

            epsilon = initial_epsilon;
        }

        void act_once(bool& done, REWARD& total_rewards, IEnvironment<ACTION, STATE>* env){

            const ACTION& act = pick_action(current_state);
            size_t curr_state_idx = state_index(current_state);

            STATE next_state;
            REWARD reward;

            if(!env->get_next_state(act, next_state, reward)){
                done = true;
                total_rewards += reward;
                return;
            }

            size_t next_state_idx = state_index(next_state);
            size_t act_idx = action_index(act);
            double old_value = Q[curr_state_idx][act_idx];
            double next_max = (!done) ? find_max_probability(Q[next_state_idx]) : 0.0;

            Q[curr_state_idx][action_index(act)] = 
                    old_value + alpha * (reward + gamma * next_max - old_value);

            current_state = next_state;
            
            total_rewards += reward;
        }

        void prepare_for_next_episode(bool& done, REWARD& total_rewards, IEnvironment<ACTION, STATE>* env){
            done = false;
            total_rewards = 0;
            current_state = env->get_initial_state();
            env->set_current_state(current_state);
        }

        void finish_episode(const size_t& i, const REWARD& total_rewards){
            epsilon = std::max(0.01, epsilon * 0.995);
            rewards_all_episodes[i] = total_rewards;
        }

        void train(IEnvironment<ACTION, STATE>* env){
            
            prepare_for_training(env);

            bool done = false;
            REWARD total_rewards = 0;

            for(size_t i = 0; i < num_episodes; i++){

                prepare_for_next_episode(done, total_rewards, env);

                while(!done){
                    act_once(done, total_rewards, env);
                }

                finish_episode(i, total_rewards);

            }

        }

        size_t get_num_episodes() { return num_episodes; }

        STATE get_current_state(){ return current_state; }

};