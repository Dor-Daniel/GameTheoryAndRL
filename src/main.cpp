
#include <iostream>
#include "IEnvironment.hpp"
#include "maze_environment.hpp"
#include "agent.cpp"
#include <raylib.h>


#define WIDTH 1000
#define HEIGHT 1000
#define BACKGROUND_COLOR Color{ .r = 20, .g = 20, .b = 20, .a = 255 }

void DrawMaze(maze::MazeEnv& env, Agent<maze::ACTION, maze::STATE, REWARD>& oo7){
    int x = 0, y = 0;
    const auto& maze_data = maze::maze;
    int n = 0, m = 0;
    size_t size_of_block = 100;
    env.get_dimention(n, m);

    Color wall = {.r = 20 , .g = 20, .b = 20, .a = 255}, empty_cell = {.r = 233, .g = 168, .b = 255, .a = 255};
    
    for(int i = 0; i < n; i++){ // rows
        for(int j = 0; j < m; j++){ // columns

            if(maze_data[ i * n + j ] != 0)
                DrawRectangle(x + j * size_of_block, y + i * size_of_block, size_of_block, size_of_block,wall);
            else{
                DrawRectangle(x + j * size_of_block, y + i * size_of_block, size_of_block, size_of_block,empty_cell);
            }
        }
    } 
    float a = (float)oo7.get_current_state().x;
    float b = (float)oo7.get_current_state().y;
    a = x + (a + 0.5f) * size_of_block;
    b = y + (b + 0.5f) * size_of_block;
    DrawCircle(a, b, size_of_block / 2.0f, (Color){.r = 255, .g = 0, .b = 225, . a = 255});
}

int main(void){
    
    maze::MazeEnv env{};

    Agent<maze::ACTION, maze::STATE, REWARD> oo7(
        env.actions,
        env.actions_count,
        env.states,
        env.states_count
    );  

    float speed = 1.0f;
    float t = 0.0f;
    float treshold = 1.0f;
    
    InitWindow(WIDTH, HEIGHT, "RL maze");
    SetTargetFPS(60);

    oo7.prepare_for_training(&env);
    
    size_t current_episode = 0;
    bool done = false;
    REWARD total_reward = 0;
    bool finished_training = false;
    oo7.prepare_for_next_episode(done, total_reward, &env);

    while (!WindowShouldClose())
    {
        if(!finished_training){
            
            t += GetFrameTime() * speed;
            
            if(t > treshold){
                t = 0.0f;
                if(!done){
                    // episode in process
                    oo7.act_once(done, total_reward, &env);
                }else{
                    current_episode++;
                    if(current_episode == oo7.get_num_episodes()){
                        finished_training = true;
                    }else{
                        oo7.finish_episode(current_episode, total_reward);
                        oo7.prepare_for_next_episode(done, total_reward, &env);
                        done = false;
                    }
                }
            }
        }

        BeginDrawing();
            ClearBackground(BACKGROUND_COLOR);
            DrawMaze(env, oo7);
            DrawText(std::to_string(current_episode).c_str(), 10, 10, 20, BLACK);
        EndDrawing();

    }
    
    CloseWindow();


    return 0;
}