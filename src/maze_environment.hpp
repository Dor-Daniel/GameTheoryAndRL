#pragma once 

#include "IEnvironment.hpp"
#include <vector>
#include <stdbool.h>
#include <stdlib.h>
#include <memory>


namespace maze{

    struct point{
        int x, y;

        void operator=(const point& other){ x = other.x; y = other.y; }
        const bool operator==(const point& other) const { return x == other.x && y == other.y; }
        const point operator+(const point& other) { 
            return point{x + other.x, y + other.y};
        }
    };

    enum action{
        LEFT = 0, RIGHT = 1, UP = 2, DOWN = 3
    };

    point ACT(action a);

    typedef point STATE; 
    typedef action ACTION;


    const static STATE init = STATE{0, 0};
    const STATE goal = STATE{9, 9};

    const REWARD goal_reward = 50, hit_reward = -10, step_reward = -1;


    static const std::vector<int> maze{
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
        1, 1, 1, 0, 1, 0, 1, 1, 0, 1,
        1, 0, 0, 0, 0, 0, 1, 0, 0, 1,
        1, 0, 1, 1, 1, 1, 1, 0, 1, 1,
        1, 0, 1, 0, 0, 0, 0, 0, 1, 1,
        1, 0, 1, 0, 1, 1, 1, 0, 1, 1,
        1, 0, 1, 0, 1, 0, 0, 0, 1, 1,
        1, 0, 1, 0, 1, 0, 1, 0, 0, 1,
        1, 1, 1, 0, 1, 1, 1, 1, 0, 0
    };


    class MazeEnv : public IEnvironment<ACTION, STATE> {
        private:
            // N rows, M columns 
            const int N, M;

            bool is_state_valid(const STATE& s);

            REWARD get_reward(const STATE& s);

            
            

        public:
        
            MazeEnv();

            ~MazeEnv();

            // returns true if not done
            bool get_next_state(const ACTION& act, STATE& next_state, REWARD& reward);

            void get_dimention(int& n, int& m){ n = N; m = M; }

            bool is_goal_reached();

    };

}