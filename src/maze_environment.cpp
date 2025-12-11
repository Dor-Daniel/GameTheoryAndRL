#include "maze_environment.hpp"

namespace maze{

    point ACT(action a) { 
        return point{
                -1 + 2 * (a - 2) ? a > 1 : 0,
                -1 + 2 * a ? a < 2 : 0
            }; 
    }

    bool MazeEnv::is_state_valid(const STATE& s){
        if(!(s.x >= 0 && s.x < M && s.y >= 0 && s.y < N))
            return false;
        return maze[s.y * M + s.x] != 1; 
    }

    REWARD MazeEnv::get_reward(const STATE& s){
        if(s == goal) return goal_reward;
        if(!is_state_valid(s)) return hit_reward;
        return step_reward;
    }

    MazeEnv::MazeEnv() : N(10), M(10) {
        current_state = init;
        initial_state = init;
        states_count = N * M;
        actions_count = 4;
        states = (STATE*)malloc(states_count * sizeof(STATE));
        actions = (ACTION*)malloc(actions_count * sizeof(ACTION)); 
        for(int i = 0; i < N; i++){
            for(int j = 0; j < M; j++){
                states[i * M + j] = STATE{i, j};
            }
        }
        for(size_t i = 0; i < actions_count; i++){
            actions[i] = (ACTION)i;
        }

    }

    MazeEnv::~MazeEnv(){
        free(states);
        free(actions);

    }

    // returns true if not done
    bool MazeEnv::get_next_state(const ACTION& act, STATE& next_state, REWARD& reward){
        next_state = (current_state + ACT(act));
        current_state = next_state;
        reward = get_reward(next_state);
        return reward == step_reward;
    }

    bool MazeEnv::is_goal_reached(){
        return current_state == goal;
    }
    
}