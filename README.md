# Towards Realistic Crowd Simulation with Collision Avoidance Behavior Modeling

> This is a minimal realization for our work *Towards Realistic Crowd Simulation with Collision Avoidance Behavior Modeling*.
> For more details, contact me or see (the verbose realization)[https://github.com/yuzhTHU/RL4Pedsim]

<!-- ## TTC-MD Domain Transform Algorithm for Collision Avoidance Process Detection -->

1. To train an TEC-RL model, use `python train.py`. See `get_args()` in ./utils/utils.py for settable arguments.
2. To evaluate an trained model, use `python evaluate.py --LOAD_MODEL <MODEL>`.
    - We provide a demonstration model in `./checkpoint/demonstration/model_final.bin`, so you can use `python evaluation.py --LOAD_MODEL ./checkpoint/demonstration/model_final.bin` to evaluate its performance.
3. To visualize a model, use `python visualize.py --LOAD <MODEL>`. If no model is specified, the script can run normally but the agents just walk randomly.