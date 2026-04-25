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
import time
import random
import copy

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

    # Domain randomization ranges (evaluation conditions)
    # 域随机化范围（评估条件）
    DR_BATTERY_MAX = [150, 200, 300]
    DR_CHARGER_COUNT = [2, 3, 4]
    DR_ROBOT_COUNT = [2, 3, 4]

    def _randomize_conf(self):
        """Apply domain randomization to usr_conf before each episode.

        每局开始前随机化环境配置，确保智能体在多种条件下训练。
        """
        conf = copy.deepcopy(self.usr_conf)
        env_conf = conf.get("env_conf", conf)
        env_conf["battery_max"] = random.choice(self.DR_BATTERY_MAX)
        env_conf["charger_count"] = random.choice(self.DR_CHARGER_COUNT)
        env_conf["robot_count"] = random.choice(self.DR_ROBOT_COUNT)
        return conf

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

            # Domain randomization: vary battery/charger/NPC each episode
            # 域随机化：每局随机电量/充电桩/NPC数量
            ep_conf = self._randomize_conf()
            ep_env_conf = ep_conf.get("env_conf", ep_conf)

            # Reset environment
            # 重置环境
            env_obs = self.env.reset(ep_conf)
            if handle_disaster_recovery(env_obs, self.logger):
                continue

            # Reset agent and load latest model
            # 重置 Agent，加载最新模型
            self.agent.reset(env_obs)
            self.agent.load_model(id="latest")

            # Sync battery_max to preprocessor so features are consistent
            # 同步 battery_max 到预处理器，确保特征一致
            self.agent.preprocessor.battery_max = int(ep_env_conf.get("battery_max", 200))

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

            self.logger.info(
                f"Episode {self.episode_cnt} start "
                f"[DR battery={ep_env_conf.get('battery_max')}, "
                f"charger={ep_env_conf.get('charger_count')}, "
                f"npc={ep_env_conf.get('robot_count')}]"
            )

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
                        # Survived to max steps: moderate reward for survival + cleaning
                        # 存活到最大步数：适度奖励存活 + 清扫比例加成
                        cleaning_ratio = fm.dirt_cleaned / max(fm.total_dirt, 1)
                        final_reward = 0.5 + 2.0 * cleaning_ratio
                        result_str = "WIN"
                    else:
                        # Early death = bad, regardless of cause
                        # 提前死亡 = 适度惩罚（过大会导致GAE方差爆炸）
                        if ep_min_npc_dist < 0.5:
                            final_reward = -1.0
                            result_str = "NPC_COLLISION"
                        else:
                            final_reward = -1.0
                            result_str = "BATTERY_DEAD"

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
