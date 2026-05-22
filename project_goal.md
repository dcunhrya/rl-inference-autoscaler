# Project Outline: Dynamic Inference Autoscaler

**Objective:** Build and deploy a predictive, reinforcement learning-based autoscaler for ML inference workloads using the Ray ecosystem.

## 1. System Architecture
*   **Simulator:** A custom Gymnasium environment simulating the cluster.
*   **Distributed Training:** Ray RLlib parallelizing data collection.
*   **Online Serving:** A Ray Serve deployment wrapped in FastAPI to trigger scaling.

## 2. Mathematical Formulations (MDP)

**State Space ($S_t$)**
The continuous state vector at timestep t:
$S_t = [\lambda_t, \mu_t, N_t, \delta_lambda_t]$
Where:
* $\lambda_t$: Request arrival rate (RPS)
* $\mu_t$: Average hardware utilization
* $N_t$: Current active replica count
* $\delta_{\lambda_t}$: First derivative of the arrival rate

**Action Space ($A_t$)**
Discrete set of scaling actions to prevent thrashing:
$A_t$ in {-1, 0, +1}
* -1: Terminate one replica
*  0: Maintain capacity
* +1: Provision one new replica

**Reward Function ($R_t$)**
Optimizing the Pareto frontier between financial cost and system latency:
$R_t = - (\alpha * N_t) - \beta * max(0, \lambda_t - (\mu * N_t))$
Where:
* $\alpha$: Cost coefficient per active replica
* $\beta$: Latency penalty coefficient
* $\mu$: Maximum safe throughput of a single replica

## 3. Algorithmic Details (PPO)

Algorithm: Proximal Policy Optimization (PPO)

**Objective Function**
PPO uses a clipped surrogate objective to ensure stable policy updates.
$L_{CLIP}(\theta) = E_t [ min( r_t(\theta) * A_hat_t, clip(r_t(\theta), 1 - eps, 1 + eps) * A_hat_t ) ]$
Where:
* $r_t(\theta)$: Probability ratio of new vs old policy
* $A_hat_t$: Advantage estimate
* $eps$: Clipping parameter (e.g., 0.2)

**Network Architecture**
* Actor Network: 2-layer MLP (256 units each), outputting softmax over 3 actions.
* Critic Network: 2-layer MLP (256 units each), outputting a scalar value estimate $V(S_t)$.

## 4. The 3 Phases of the Project

### Phase 1: Environment Engineering
1. Implement the simulation as a custom gymnasium.Env.
2. Define observation_space as spaces.Box and action_space as `spaces.Discrete(3)`.
3. Implement the `step()` function to simulate K8s cold-start delays and queue build-ups.

### Phase 2: Distributed Training
1. Initialize the Ray cluster via `ray.init()`.
2. Configure the RL algorithm using `PPOConfig()`.
3. Set num_env_runners to parallelize data collection across available CPU cores.
4. Track episode_return_mean and policy entropy via MLflow or W&B.

### Phase 3: Deployment
1. Extract the trained PyTorch module using Algorithm.`get_module()`.
2. Wrap the inference logic in a `@serve.deployment` class.
3. Expose the deployment via a FastAPI endpoint to ingest live traffic metrics and return scaling actions.