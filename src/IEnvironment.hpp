#pragma once
#include <stdlib.h>

typedef double REWARD;

template <typename ACTION, typename STATE>
class IEnvironment{
    protected:
    
        STATE current_state;
        STATE initial_state;

    public:
        IEnvironment() = default; 
        ~IEnvironment() = default;
        
        virtual bool get_next_state(const ACTION& act, STATE& next_state, REWARD& reward) = 0;
        STATE get_initial_state() const { return initial_state; }
        void set_current_state(const STATE& state){ current_state = state; }

        STATE* states;
        size_t states_count;

        ACTION* actions;
        size_t actions_count;
};