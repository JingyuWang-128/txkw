#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Training workflow for Robot Vacuum.
清扫大作战训练工作流。
"""

import os
import shutil
import tempfile
import time
import zipfile

import numpy as np

from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import SampleData, sample_process
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    last_save_model_time = time.time()
    env = envs[0]
    agent = agents[0]

    # Read and validate user configuration
    # 读取和校验用户配置
    usr_conf = read_usr_conf("agent_ppo/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error("usr_conf is None, please check agent_ppo/conf/train_env_conf.toml")
        return

    # Resume from zip checkpoint
    # 从 zip 检查点恢复训练
    resume_zip_path = usr_conf.get("resume_zip_path", "")
    resume_dir = None
    if resume_zip_path and os.path.isfile(resume_zip_path):
        logger.info(f"Resume training from zip: {resume_zip_path}")
        try:
            resume_dir = tempfile.mkdtemp(prefix="ppo_resume_")
            with zipfile.ZipFile(resume_zip_path, 'r') as zip_ref:
                zip_ref.extractall(resume_dir)
            logger.info(f"Extracted zip to: {resume_dir}")
            # Load model from extracted directory
            # 优先加载 latest，其次加载 checkpoint-1
            latest_path = os.path.join(resume_dir, "model.ckpt-latest.pkl")
            ckpt_path = os.path.join(resume_dir, "model.ckpt-1.pkl")
            if os.path.exists(latest_path):
                agent.load_model(path=resume_dir, id="latest")
            elif os.path.exists(ckpt_path):
                agent.load_model(path=resume_dir, id="1")
            else:
                # 列出解压后的所有 .pkl 文件
                pkl_files = [f for f in os.listdir(resume_dir) if f.endswith('.pkl')]
                logger.warning(f"No model.ckpt-latest.pkl or model.ckpt-1.pkl found. Available: {pkl_files}")
        except Exception as e:
            logger.error(f"Failed to resume from zip: {e}")
            if resume_dir and os.path.exists(resume_dir):
                shutil.rmtree(resume_dir)
            resume_dir = None
    else:
        if resume_zip_path:
            logger.warning(f"resume_zip_path set but file not found: {resume_zip_path}")
        logger.info("Starting training from scratch (no resume zip)")

    episode_runner = EpisodeRunner(
        env=env,
        agent=agent,
        usr_conf=usr_conf,
        logger=logger,
        monitor=monitor,
    )

    while True:
        for g_data in episode_runner.run_episodes():
            agent.send_sample_data(g_data)
            g_data.clear()

            now = time.time()
            if now - last_save_model_time >= 1800:
                agent.save_model()
                last_save_model_time = now


class EpisodeRunner:
    def __init__(self, env, agent, usr_conf, logger, monitor):
        self.env = env
        self.agent = agent
        self.usr_conf = usr_conf
        self.logger = logger
        self.monitor = monitor
        self.episode_cnt = 0
        self.last_report_monitor_time = 0
        self.last_get_training_metrics_time = 0

    def run_episodes(self):
        """Run a single episode and yield collected samples.

        单局流程（generator），完成一局后 yield 整局样本。
        """
        while True:
            # Periodically get training metrics
            # 定期打印训练指标
            now = time.time()
            if now - self.last_get_training_metrics_time >= 60:
                training_metrics = get_training_metrics()
                self.last_get_training_metrics_time = now
                if training_metrics is not None:
                    self.logger.info(f"training_metrics: {training_metrics}")

            # Reset environment
            # 重置环境
            env_obs = self.env.reset(self.usr_conf)
            if handle_disaster_recovery(env_obs, self.logger):
                continue

            # Reset agent and load latest model
            # 重置 Agent，加载最新模型
            self.agent.reset(env_obs)
            self.agent.load_model(id="latest")

            # Initial observation processing
            # 初始观测
            obs_data, remain_info = self.agent.observation_process(env_obs)

            collector = []
            self.episode_cnt += 1
            done = False
            step = 0
            total_reward = 0.0
            ep_reward_comps = {}
            ep_min_npc_dist = 1e9
            ep_min_charger_dist = 1e9
            ep_max_coverage = 0.0
            ep_idle_steps = 0

            self.logger.info(f"Episode {self.episode_cnt} start")

            while not done:
                # Agent inference / 推理动作
                act_data_list = self.agent.predict([obs_data])
                act_data = act_data_list[0]
                act = self.agent.action_process(act_data)

                # Environment step / 与环境交互
                env_reward, env_obs = self.env.step(act)
                if handle_disaster_recovery(env_obs, self.logger):
                    break

                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]
                frame_no = env_obs["frame_no"]
                step += 1
                done = terminated or truncated

                # Process next observation
                # 特征处理
                _obs_data, _ = self.agent.observation_process(env_obs)
                _obs_data.frame_no = frame_no

                reward_scalar = float(self.agent.last_reward)
                total_reward += reward_scalar

                # Episode-level aggregation for reward components & key metrics
                fm = self.agent.preprocessor
                comps = getattr(fm, "last_reward_components", None) or {}
                for k, v in comps.items():
                    if k == "total":
                        continue
                    ep_reward_comps[k] = ep_reward_comps.get(k, 0.0) + float(v)

                ep_min_npc_dist = min(ep_min_npc_dist, float(getattr(fm, "d_npc_min", 1e9)))
                ep_min_charger_dist = min(ep_min_charger_dist, float(getattr(fm, "d_charger", 1e9)))
                ep_max_coverage = max(ep_max_coverage, float(getattr(fm, "coverage_ratio", 0.0)))
                if int(getattr(fm, "cleaned_this_step", 0)) == 0:
                    ep_idle_steps += 1

                # Terminal reward calculation
                # 终局奖励
                final_reward = 0.0
                if done:
                    total_score = env_obs["observation"]["env_info"]["total_score"]

                    if truncated:
                        # Survived to max steps: higher cleaning ratio → more reward
                        # 存活到最大步数：清扫比例越高奖励越多
                        cleaning_ratio = fm.dirt_cleaned / max(fm.total_dirt, 1)
                        final_reward = 5.0 + 5.0 * cleaning_ratio
                        result_str = "WIN"
                    else:
                        # Early termination (battery depleted or collision): small penalty
                        # 提前结束（电量耗尽或碰撞）：小惩罚
                        final_reward = -2.0
                        result_str = "FAIL"

                    comp_str = " ".join(
                        [f"{k}:{ep_reward_comps.get(k, 0.0):.3f}" for k in sorted(ep_reward_comps.keys())]
                    )
                    idle_rate = float(ep_idle_steps / max(step, 1))
                    charge_count = int(getattr(fm, "charge_count", 0))
                    self.logger.info(
                        f"[GAMEOVER] ep:{self.episode_cnt} steps:{step} "
                        f"result:{result_str} final_bonus:{final_reward:.2f} "
                        f"total_reward:{total_reward:.3f} "
                        f"dirt_cleaned:{fm.dirt_cleaned}/{fm.total_dirt} "
                        f"min_npc_dist:{ep_min_npc_dist:.2f} "
                        f"min_charger_dist:{ep_min_charger_dist:.2f} "
                        f"coverage_max:{ep_max_coverage:.3f} "
                        f"idle_rate:{idle_rate:.3f} "
                        f"charge_count:{charge_count} "
                        f"reward_comps[{comp_str}]"
                    )

                # Build sample frame
                # 构造样本帧
                reward_arr = np.array([reward_scalar], dtype=np.float32)
                value_arr = act_data.value.flatten()[: Config.VALUE_NUM]

                frame = SampleData(
                    obs=np.array(obs_data.feature, dtype=np.float32),
                    legal_action=np.array(obs_data.legal_action, dtype=np.float32),
                    act=np.array(act_data.action),
                    reward=reward_arr,
                    done=np.array([float(done)]),
                    reward_sum=np.zeros(Config.VALUE_NUM, dtype=np.float32),
                    value=value_arr,
                    next_value=np.zeros(Config.VALUE_NUM, dtype=np.float32),
                    advantage=np.zeros(Config.VALUE_NUM, dtype=np.float32),
                    prob=np.array(act_data.prob, dtype=np.float32),
                )
                collector.append(frame)

                if done:
                    # Add terminal reward to last frame
                    # 终局奖励叠加到最后一步
                    collector[-1].reward = collector[-1].reward + np.array([final_reward], dtype=np.float32)

                    # Monitor reporting / 监控上报
                    now = time.time()
                    if now - self.last_report_monitor_time >= 60 and self.monitor:
                        payload = {
                            "reward": total_reward + final_reward,
                            "episode_cnt": self.episode_cnt,
                            "min_npc_dist": float(ep_min_npc_dist),
                            "min_charger_dist": float(ep_min_charger_dist),
                            "coverage_max": float(ep_max_coverage),
                            "idle_rate": float(ep_idle_steps / max(step, 1)),
                            "charge_count": float(getattr(fm, "charge_count", 0)),
                        }
                        for k, v in ep_reward_comps.items():
                            payload[f"r_{k}"] = float(v)
                        self.monitor.put_data(
                            {
                                os.getpid(): {
                                    **payload,
                                }
                            }
                        )
                        self.last_report_monitor_time = now

                    # Compute GAE and yield samples
                    # GAE 计算并 yield 样本
                    if collector:
                        collector = sample_process(collector)
                        yield collector
                    break

                # Advance state / 状态推进
                obs_data = _obs_data
