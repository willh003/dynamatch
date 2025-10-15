#!/usr/bin/env python3
"""
Example script showing how to use eval_trajectory_policies.py

This script demonstrates how to:
1. Collect a trajectory using one policy
2. Test multiple policies on those states
3. Generate comparison plots

Usage examples:
    # Basic usage with one base policy and one translator policy
    python eval_trajectory_policies.py \
        --trajectory_policy_config configs/source_policy/ant_ppo.yaml \
        --policy_configs configs/source_policy/ant_ppo.yaml configs/translator_policy/ant_translator.yaml \
        --policy_names "Base Policy" "Translator Policy" \
        --env_id Ant-v5 \
        --max_steps 500

    # With custom checkpoints
    python eval_trajectory_policies.py \
        --trajectory_policy_config configs/source_policy/ant_ppo.yaml \
        --trajectory_policy_checkpoint /path/to/trajectory/checkpoint.zip \
        --policy_configs configs/source_policy/ant_ppo.yaml configs/translator_policy/ant_translator.yaml \
        --policy_checkpoints /path/to/base/checkpoint.zip /path/to/translator/checkpoint.zip \
        --policy_names "Base Policy" "Translator Policy" \
        --env_id Ant-v5 \
        --max_steps 500 \
        --output_dir my_trajectory_eval

    # With multiple policies for comparison
    python eval_trajectory_policies.py \
        --trajectory_policy_config configs/source_policy/ant_ppo.yaml \
        --policy_configs \
            configs/source_policy/ant_ppo.yaml \
            configs/translator_policy/ant_translator_v1.yaml \
            configs/translator_policy/ant_translator_v2.yaml \
        --policy_names "Base" "Translator V1" "Translator V2" \
        --env_id Ant-v5 \
        --max_steps 150 \
        --deterministic
"""

import subprocess
import sys
import os

def run_example():
    """Run an example trajectory evaluation."""
    
    cmd = [
        "python", "eval_trajectory_policies.py",
        "--trajectory_policy_config", "configs/source_policy/ant_modified_physics_10M.yaml",
        "--policy_configs", 
            "configs/source_policy/ant_ppo_5M.yaml",
            #"configs/translator_policy/ant_ppo_og_modphys.yaml",
            #"configs/translator_policy/ant_ppo_og_modphys_bc.yaml",
            #"configs/translator_policy/ant_ppo_og_modphys_actonly.yaml",
        "--policy_names", "Base Policy", #"Translator Policy", #"Base Policy", #"Translator Policy BC", #, "Translator Policy ActOnly",
        "--env_id", "AntModifiedPhysics-v1",
        "--max_steps", "250",
        "--deterministic",
        "--output_dir", "example_trajectory_eval"
    ]

    # cmd = [
    #     "python", "eval_trajectory_policies.py",
    #     "--trajectory_policy_config", "configs/source_policy/ant_ppo_5M.yaml",
    #     "--policy_configs", 
    #         "configs/source_policy/ant_modified_physics_10M.yaml",
    #         #"configs/source_policy/ant_ppo_5M.yaml",
    #         #"configs/translator_policy/ant_ppo_og_modphys.yaml",
    #         #"configs/translator_policy/ant_ppo_og_modphys_bc.yaml",
    #         #"configs/translator_policy/ant_ppo_og_modphys_actonly.yaml",
    #     "--policy_names", "Mod Friction Policy", #"Translator Policy", #"Base Policy", #"Translator Policy BC", #, "Translator Policy ActOnly",
    #     "--env_id", "Ant-v5",
    #     "--max_steps", "250",
    #     "--deterministic",
    #     "--output_dir", "example_trajectory_eval"
    # ]
    
    print("Running example trajectory evaluation...")
    print("Command:", " ".join(cmd))
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False
    except FileNotFoundError:
        print("Error: eval_trajectory_policies.py not found. Make sure you're in the correct directory.")
        return False
    
    return True

if __name__ == "__main__":
    success = run_example()
    sys.exit(0 if success else 1)
