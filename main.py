def main():
    import sys
    from pathlib import Path

    _src = Path(__file__).resolve().parent / "src"
    if _src.is_dir() and str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

    from rl_inference_autoscaler.baselines import evaluate_baseline

    print("Learning to Scale GPU Workloads with Reinforcement Learning — Phase 1 smoke run")
    metrics = evaluate_baseline(episodes=1, seed=0)
    print(f"Target-utilization baseline return: {metrics['episode_return_mean']:.2f}")


if __name__ == "__main__":
    main()
