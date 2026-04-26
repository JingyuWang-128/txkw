#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Feature preprocessor for Robot Vacuum.
清扫大作战特征预处理器。
"""

import numpy as np


def _norm(v, v_max, v_min=0.0):
    """Normalize value to [0, 1].

    将值线性归一化到 [0, 1]。
    """
    v = float(np.clip(v, v_min, v_max))
    if v_max == v_min:
        return 0.0
    return (v - v_min) / (v_max - v_min)


class Preprocessor:
    """Feature preprocessor for Robot Vacuum.

    清扫大作战特征预处理器。
    """

    GRID_SIZE = 128
    VIEW_HALF = 10  # Full local view radius (21×21) / 完整局部视野半径
    LOCAL_HALF = 3  # Cropped view radius (7×7) / 裁剪后的视野半径
    COARSE_GRID = 16  # coarse visited grid size / 粗粒度访问网格

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all internal state at episode start.

        对局开始时重置所有状态。
        """
        self.step_no = 0
        self.battery = 0
        self.battery_max = 1

        self.cur_pos = (0, 0)

        self.dirt_cleaned = 0
        self.last_dirt_cleaned = 0
        self.total_dirt = 1

        # Global passable map (0=obstacle, 1=passable), used for ray computation
        # 维护全局通行地图（0=障碍, 1=可通行），用于射线计算
        self.passable_map = np.ones((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)

        # Nearest dirt distance
        # 最近污渍距离
        self.nearest_dirt_dist = 200.0
        self.last_nearest_dirt_dist = 200.0

        self._view_map = np.zeros((21, 21), dtype=np.float32)
        self._legal_act = [1] * 8

        # -------- Phase1: entity parsing & history state --------
        self.chargers = []  # list[(x,z)] current frame visible
        self.known_chargers = set()  # persistent charger memory across frames
        self.npcs = []  # list[(x,z)]
        self.step_cleaned_cells = []  # list[(x,z)]

        self.last_pos = (0, 0)
        self.last_battery = self.battery

        self.visited = set()  # visited fine cells
        self.visited_coarse = set()  # visited coarse bins
        self.recent_positions = []  # sliding window of recent positions

        self.action_hist_k = 4
        self.action_hist = [-1] * self.action_hist_k

        self.clean_streak = 0
        self.idle_steps = 0
        self.charge_count = 0

        self.last_d_charger_norm = 1.0
        self.last_reward_components = {}

        # cached step metrics
        self.d_charger = 200.0
        self.d_npc_min = 200.0
        self.coverage_ratio = 0.0
        self.recent_coverage = 0.0
        self.cleaned_this_step = 0
        self.is_new_cell = False
        self.loop_flag = 0.0
        self.bfs_d_charger = 256.0
        self.bfs_d_dirt = 256.0

    def pb2struct(self, env_obs, last_action):
        """Parse and cache essential fields from observation dict.

        从 env_obs 字典中提取并缓存所有需要的状态量。
        """
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        hero = frame_state["heroes"]

        self.step_no = int(observation["step_no"])
        self.last_pos = self.cur_pos
        self.cur_pos = (int(hero["pos"]["x"]), int(hero["pos"]["z"]))

        # Battery / 电量
        self.last_battery = self.battery
        self.battery = int(hero["battery"])
        self.battery_max = max(int(hero["battery_max"]), 1)

        # Cleaning progress / 清扫进度
        self.last_dirt_cleaned = self.dirt_cleaned
        self.dirt_cleaned = int(hero["dirt_cleaned"])
        self.total_dirt = max(int(env_info["total_dirt"]), 1)

        # step_cleaned_cells / 本步清扫格子
        self.step_cleaned_cells = []
        raw_cleaned = env_info.get("step_cleaned_cells") if isinstance(env_info, dict) else None
        if raw_cleaned:
            for c in raw_cleaned:
                if isinstance(c, dict) and "x" in c and "z" in c:
                    self.step_cleaned_cells.append((int(c["x"]), int(c["z"])))
                elif isinstance(c, (list, tuple)) and len(c) >= 2:
                    self.step_cleaned_cells.append((int(c[0]), int(c[1])))

        # Legal actions / 合法动作
        self._legal_act = [int(x) for x in (observation.get("legal_action") or [1] * 8)]

        # Local view map (21×21) / 局部视野地图
        map_info = observation.get("map_info")
        if map_info is not None:
            self._view_map = np.array(map_info, dtype=np.float32)
            hx, hz = self.cur_pos
            self._update_passable(hx, hz)

        # Entities: chargers(organs sub_type=1) & NPCs
        self.chargers = []
        organs = frame_state.get("organs") if isinstance(frame_state, dict) else None
        if organs:
            for o in organs:
                try:
                    if int(o.get("sub_type", -1)) != 1:
                        continue
                    p = o.get("pos", {})
                    # +1 center correction: charger is 3x3, pos is top-left
                    # +1 中心修正：充电桩占 3×3，pos 是左上角
                    cx = int(p.get("x", 0)) + 1
                    cz = int(p.get("z", 0)) + 1
                    self.chargers.append((cx, cz))
                    self.known_chargers.add((cx, cz))
                except Exception:
                    continue
        # Use persistent memory so chargers are never "forgotten"
        # 使用持久化记忆，走远后不会“忘记”充电桩位置
        if self.known_chargers:
            self.chargers = list(self.known_chargers)

        self.npcs = []
        npcs = frame_state.get("npcs") if isinstance(frame_state, dict) else None
        if npcs:
            for n in npcs:
                try:
                    p = n.get("pos", {})
                    self.npcs.append((int(p.get("x", 0)), int(p.get("z", 0))))
                except Exception:
                    continue

        # History: action & visited / 历史：动作与访问记忆
        if last_action is None:
            last_action = -1
        self.action_hist = (self.action_hist[1:] + [int(last_action)])[-self.action_hist_k :]

        hx, hz = self.cur_pos
        self.is_new_cell = (hx, hz) not in self.visited
        self.visited.add((hx, hz))

        bin_size = max(1, self.GRID_SIZE // self.COARSE_GRID)
        cx, cz = hx // bin_size, hz // bin_size
        self.visited_coarse.add((int(cx), int(cz)))
        self.coverage_ratio = float(len(self.visited_coarse) / (self.COARSE_GRID * self.COARSE_GRID))

        # recent coverage
        self.recent_positions.append((hx, hz))
        max_recent = 50
        if len(self.recent_positions) > max_recent:
            self.recent_positions = self.recent_positions[-max_recent:]
        uniq_recent = len(set(self.recent_positions))
        self.recent_coverage = float(uniq_recent / max(1, len(self.recent_positions)))

        # loop flag (returned to a recent position)
        recent_window = 20
        recent = self.recent_positions[-recent_window:]
        self.loop_flag = 1.0 if len(recent) >= 2 and recent.count((hx, hz)) >= 2 else 0.0

        # charge count heuristic: battery increased (e.g., charging)
        if self.battery > self.last_battery + 5:
            self.charge_count += 1

        # cache step-level distances
        self.d_charger = self._calc_nearest_entity_dist(self.chargers)
        self.d_npc_min = self._calc_nearest_entity_dist(self.npcs)

        # BFS distances (Phase2, computed using passable_map + local targets)
        self.bfs_d_charger = self._calc_bfs_to_charger()
        self.bfs_d_dirt = self._calc_bfs_to_local_dirt()

    def _calc_nearest_entity_dist(self, entities):
        """Nearest Euclidean distance to a set of (x,z) entities."""
        if not entities:
            return 200.0
        hx, hz = self.cur_pos
        dx = np.array([e[0] - hx for e in entities], dtype=np.float32)
        dz = np.array([e[1] - hz for e in entities], dtype=np.float32)
        d = np.sqrt(dx * dx + dz * dz)
        return float(np.min(d)) if d.size > 0 else 200.0

    def _bfs_distance(self, start, targets, max_expand=4096):
        """BFS shortest path distance on passable_map (4-neighborhood).

        Returns large value if unreachable or targets empty.
        """
        if not targets:
            return 256.0
        sx, sz = int(start[0]), int(start[1])
        if (sx, sz) in targets:
            return 0.0
        if not (0 <= sx < self.GRID_SIZE and 0 <= sz < self.GRID_SIZE):
            return 256.0
        if self.passable_map[sx, sz] == 0:
            return 256.0

        from collections import deque

        q = deque()
        q.append((sx, sz, 0))
        seen = set([(sx, sz)])
        expands = 0
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        while q:
            x, z, d = q.popleft()
            expands += 1
            if expands > max_expand:
                break
            nd = d + 1
            for dx, dz in dirs:
                nx, nz = x + dx, z + dz
                if not (0 <= nx < self.GRID_SIZE and 0 <= nz < self.GRID_SIZE):
                    continue
                if (nx, nz) in seen:
                    continue
                if self.passable_map[nx, nz] == 0:
                    continue
                if (nx, nz) in targets:
                    return float(nd)
                seen.add((nx, nz))
                q.append((nx, nz, nd))
        return 256.0

    def _calc_bfs_to_charger(self):
        if not self.chargers:
            return 256.0
        # organs size w=h=3, treat as 3x3 area around center / 充电桩占 3x3
        targets = set()
        for (x, z) in self.chargers:
            for dx in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    tx, tz = int(x + dx), int(z + dz)
                    if 0 <= tx < self.GRID_SIZE and 0 <= tz < self.GRID_SIZE:
                        targets.add((tx, tz))
        return self._bfs_distance(self.cur_pos, targets)

    def _calc_bfs_to_local_dirt(self):
        view = self._view_map
        if view is None:
            return 256.0
        dirt = np.argwhere(view == 2)
        if len(dirt) == 0:
            return 256.0
        hx, hz = self.cur_pos
        half = self.VIEW_HALF
        targets = set()
        for ri, ci in dirt:
            gx = int(hx - half + int(ri))
            gz = int(hz - half + int(ci))
            if 0 <= gx < self.GRID_SIZE and 0 <= gz < self.GRID_SIZE:
                targets.add((gx, gz))
        return self._bfs_distance(self.cur_pos, targets)

    def _update_passable(self, hx, hz):
        """Write local view into global passable map.

        将局部视野写入全局通行地图。
        """
        view = self._view_map
        vsize = view.shape[0]
        half = vsize // 2

        for ri in range(vsize):
            for ci in range(vsize):
                gx = hx - half + ri
                gz = hz - half + ci
                if 0 <= gx < self.GRID_SIZE and 0 <= gz < self.GRID_SIZE:
                    # 0 = obstacle, 1/2 = passable
                    # 0 = 障碍, 1/2 = 可通行
                    self.passable_map[gx, gz] = 1 if view[ri, ci] != 0 else 0

    def _get_local_view_feature(self):
        """Local view feature (49D): 3×3 stride-3 max pool on 21×21 view.

        局部视野特征（49D）：21×21视野经3×3最大池化压缩为7×7。
        """
        view = self._view_map / 2.0
        # 3×3 stride=3 max pool: 21×21 → 7×7 (纯numpy实现)
        h, w = view.shape
        ph, pw = h // 3, w // 3
        pooled = view[:ph*3, :pw*3].reshape(ph, 3, pw, 3).max(axis=(1, 3))
        return pooled.flatten()

    def _get_global_state_feature(self):
        """Global state feature (12D).

        全局状态特征（12D）。

        Dimensions / 维度说明：
          [0]  step_norm         step progress / 步数归一化 [0,1]
          [1]  battery_ratio     battery level / 电量比 [0,1]
          [2]  cleaning_progress cleaned ratio / 已清扫比例 [0,1]
          [3]  remaining_dirt    remaining dirt ratio / 剩余污渍比例 [0,1]
          [4]  pos_x_norm        x position / x 坐标归一化 [0,1]
          [5]  pos_z_norm        z position / z 坐标归一化 [0,1]
          [6]  ray_N_dirt        north ray distance / 向上（z-）方向最近污渍距离
          [7]  ray_E_dirt        east ray distance / 向右（x+）方向
          [8]  ray_S_dirt        south ray distance / 向下（z+）方向
          [9]  ray_W_dirt        west ray distance / 向左（x-）方向
          [10] nearest_dirt_norm nearest dirt Euclidean distance / 最近污渍欧氏距离归一化
          [11] dirt_delta        approaching dirt indicator / 是否在接近污渍（1=是, 0=否）
        """
        step_norm = _norm(self.step_no, 2000)
        battery_ratio = _norm(self.battery, self.battery_max)
        cleaning_progress = _norm(self.dirt_cleaned, self.total_dirt)
        remaining_dirt = 1.0 - cleaning_progress

        hx, hz = self.cur_pos
        pos_x_norm = _norm(hx, self.GRID_SIZE)
        pos_z_norm = _norm(hz, self.GRID_SIZE)

        # 4-directional ray to find nearest dirt
        # 四方向射线找最近污渍距离
        ray_dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N E S W
        ray_dirt = []
        max_ray = 30
        for dx, dz in ray_dirs:
            x, z = hx, hz
            found = max_ray
            for step in range(1, max_ray + 1):
                x += dx
                z += dz
                if not (0 <= x < self.GRID_SIZE and 0 <= z < self.GRID_SIZE):
                    break
                if self._view_map is not None:
                    cell = (
                        int(
                            self._view_map[
                                np.clip(x - (hx - self.VIEW_HALF), 0, 20), np.clip(z - (hz - self.VIEW_HALF), 0, 20)
                            ]
                        )
                        if (0 <= x - hx + self.VIEW_HALF < 21 and 0 <= z - hz + self.VIEW_HALF < 21)
                        else 0
                    )
                    if cell == 2:
                        found = step
                        break
            ray_dirt.append(_norm(found, max_ray))

        # Nearest dirt Euclidean distance (estimated from 7×7 crop)
        # 最近污渍欧氏距离（视野内 7×7 粗估）
        self.last_nearest_dirt_dist = self.nearest_dirt_dist
        self.nearest_dirt_dist = self._calc_nearest_dirt_dist()
        nearest_dirt_norm = _norm(self.nearest_dirt_dist, 180)

        dirt_delta = 1.0 if self.nearest_dirt_dist < self.last_nearest_dirt_dist else 0.0

        return np.array(
            [
                step_norm,
                battery_ratio,
                cleaning_progress,
                remaining_dirt,
                pos_x_norm,
                pos_z_norm,
                ray_dirt[0],
                ray_dirt[1],
                ray_dirt[2],
                ray_dirt[3],
                nearest_dirt_norm,
                dirt_delta,
            ],
            dtype=np.float32,
        )

    def _calc_nearest_dirt_dist(self):
        """Find nearest dirt Euclidean distance from local view.

        从局部视野中找最近污渍的欧氏距离。
        """
        view = self._view_map
        if view is None:
            return 200.0
        dirt_coords = np.argwhere(view == 2)
        if len(dirt_coords) == 0:
            return 200.0
        center = self.VIEW_HALF
        dists = np.sqrt((dirt_coords[:, 0] - center) ** 2 + (dirt_coords[:, 1] - center) ** 2)
        return float(np.min(dists))

    def get_legal_action(self):
        """Return legal action mask (8D list).

        返回合法动作掩码（8D list）。
        """
        return list(self._legal_act)

    def _action_hist_onehot(self):
        """One-hot encode last K actions (K*8). Invalid (-1) -> all zeros."""
        k = self.action_hist_k
        out = np.zeros((k, 8), dtype=np.float32)
        for i, a in enumerate(self.action_hist[-k:]):
            if 0 <= int(a) < 8:
                out[i, int(a)] = 1.0
        return out.flatten()

    def _charger_feats(self):
        # nearest charger distance (euclidean) and direction, plus low battery gate
        max_d = 180.0
        hx, hz = self.cur_pos
        if not self.chargers:
            d = max_d
            dx = 0.0
            dz = 0.0
        else:
            # find nearest
            best = None
            best_d2 = 1e18
            for (x, z) in self.chargers:
                ddx, ddz = float(x - hx), float(z - hz)
                d2 = ddx * ddx + ddz * ddz
                if d2 < best_d2:
                    best_d2 = d2
                    best = (ddx, ddz)
            dx, dz = best if best is not None else (0.0, 0.0)
            d = float(np.sqrt(best_d2)) if best is not None else max_d

        d_norm = _norm(d, max_d)
        dx_norm = float(np.clip(dx / max_d, -1.0, 1.0))
        dz_norm = float(np.clip(dz / max_d, -1.0, 1.0))

        battery_ratio = _norm(self.battery, self.battery_max)
        thr = 0.35
        low_battery_gate = float(np.clip((thr - battery_ratio) / max(thr, 1e-6), 0.0, 1.0))
        return np.array([d_norm, dx_norm, dz_norm, low_battery_gate], dtype=np.float32)

    def _npc_feats(self):
        max_d = 50.0
        hx, hz = self.cur_pos
        safe_radius = 2.5
        if not self.npcs:
            d = max_d
            dx = 0.0
            dz = 0.0
        else:
            best = None
            best_d2 = 1e18
            for (x, z) in self.npcs:
                ddx, ddz = float(x - hx), float(z - hz)
                d2 = ddx * ddx + ddz * ddz
                if d2 < best_d2:
                    best_d2 = d2
                    best = (ddx, ddz)
            dx, dz = best if best is not None else (0.0, 0.0)
            d = float(np.sqrt(best_d2)) if best is not None else max_d

        d_norm = _norm(d, max_d)
        dx_norm = float(np.clip(dx / max_d, -1.0, 1.0))
        dz_norm = float(np.clip(dz / max_d, -1.0, 1.0))
        too_close = 1.0 if d < safe_radius else 0.0
        return np.array([d_norm, dx_norm, dz_norm, too_close], dtype=np.float32)

    def _traj_feats(self):
        # K*8 action hist onehot + last move dx,dz + loop flag
        hx, hz = self.cur_pos
        lx, lz = self.last_pos
        dx = float(hx - lx)
        dz = float(hz - lz)
        max_step = 3.0  # per-step move roughly within 1-3 cells
        dx_norm = float(np.clip(dx / max_step, -1.0, 1.0))
        dz_norm = float(np.clip(dz / max_step, -1.0, 1.0))
        return np.concatenate(
            [
                self._action_hist_onehot(),
                np.array([dx_norm, dz_norm, float(self.loop_flag)], dtype=np.float32),
            ]
        )

    def _memory_feats(self):
        """Memory feature (5D): coverage, recent_coverage, cleaned_norm, est_steps_remaining, idle_norm.

        记忆特征（5D）：覆盖率、最近覆盖率、清扫格数、预计剩余步数、连续idle归一化。
        """
        cleaned_cnt = float(len(self.step_cleaned_cells))
        cleaned_norm = float(np.clip(cleaned_cnt / 10.0, 0.0, 1.0))
        # 预计剩余步数：基于当前电量估算（假设满电可走1000步）
        est_steps_remaining = float(np.clip(self.battery / max(self.battery_max / 1000.0, 1e-6), 0.0, 1000.0)) / 1000.0
        # 连续idle步数（归一化）
        idle_norm = float(np.clip(self.idle_steps / 50.0, 0.0, 1.0))
        return np.array(
            [
                float(np.clip(self.coverage_ratio, 0.0, 1.0)),
                float(np.clip(self.recent_coverage, 0.0, 1.0)),
                cleaned_norm,
                est_steps_remaining,
                idle_norm,
            ],
            dtype=np.float32,
        )

    def _bfs_feats(self):
        # BFS distances (path length) normalized
        max_bfs = 256.0
        d_ch = float(np.clip(self.bfs_d_charger, 0.0, max_bfs))
        d_di = float(np.clip(self.bfs_d_dirt, 0.0, max_bfs))
        return np.array([d_ch / max_bfs, d_di / max_bfs], dtype=np.float32)

    def feature_process(self, env_obs, last_action):
        """Generate 119D feature vector, legal action mask, and scalar reward.

        生成 119D 特征向量（49D视野 + 12D全局 + 8D合法动作 + 4D充电桩 + 4D NPC + 35D轨迹 + 5D记忆 + 2D BFS）、合法动作掩码和标量奖励。
        """
        self.pb2struct(env_obs, last_action)

        local_view = self._get_local_view_feature()  # 49D
        global_state = self._get_global_state_feature()  # 12D
        legal_action = self.get_legal_action()  # 8D
        legal_arr = np.array(legal_action, dtype=np.float32)

        # Phase1/2 extra features / 新增特征
        charger_feats = self._charger_feats()  # 4D
        npc_feats = self._npc_feats()  # 4D
        traj_feats = self._traj_feats()  # 35D (K=4 => 32 + 3)
        memory_feats = self._memory_feats()  # 5D
        bfs_feats = self._bfs_feats()  # 2D

        feature = np.concatenate(
            [local_view, global_state, legal_arr, charger_feats, npc_feats, traj_feats, memory_feats, bfs_feats]
        )

        reward = self.reward_process()

        return feature, legal_action, reward

    def reward_process(self):
        # Weights / 权重（清扫为主，回充为辅）
        w_clean = 0.10       # 清扫奖励：主目标，权重最高
        w_charge_nav = 0.03  # 回充导航：低电量时引导靠近充电桩
        w_charge_act = 0.08  # 成功充电奖励：踏上充电桩恢复电量
        w_npc =30        # NPC躲避
        w_new = 0.001        # 探索新格子
        w_repeat = 0.0003    # 回环惩罚
        w_streak = 0.001     # 连清奖励
        w_idle = 0.001       # 空跑惩罚

        # Cleaning reward / 清扫奖励
        self.cleaned_this_step = max(0, self.dirt_cleaned - self.last_dirt_cleaned)
        r_clean = w_clean * float(self.cleaned_this_step)

        # Efficiency: streak + idle penalty / 效率：连清 + 空跑惩罚
        if self.cleaned_this_step > 0:
            self.clean_streak += 1
        else:
            self.clean_streak = 0
            self.idle_steps += 1

        streak_cap = 20
        r_streak = w_streak * float(min(self.clean_streak, streak_cap))
        r_idle = -w_idle if self.cleaned_this_step == 0 else 0.0

        # Explore: new cell reward + repeat penalty / 探索：新格子 + 回环惩罚
        r_new = w_new if self.is_new_cell else 0.0
        r_repeat = -w_repeat * float(self.loop_flag)

        # Charger navigation: potential-based shaping (low battery gated)
        # 回充导航：势函数差分（仅低电量时激活，避免与清扫目标冲突）
        battery_ratio = _norm(self.battery, self.battery_max)
        charge_thr = 0.4  # 电量 < 40% 时开始引导回充
        gate = float(np.clip((charge_thr - battery_ratio) / max(charge_thr, 1e-6), 0.0, 1.0))
        # Prefer BFS path distance if available (Phase2) / 若可用优先用 BFS 路径距离
        d_bfs_norm = float(np.clip(self.bfs_d_charger / 256.0, 0.0, 1.0))
        d_euc_norm = float(np.clip(self._charger_feats()[0], 0.0, 1.0))
        d_charger_norm = d_bfs_norm if d_bfs_norm < 1.0 else d_euc_norm
        phi = -d_charger_norm
        last_phi = -float(np.clip(self.last_d_charger_norm, 0.0, 1.0))
        r_charge_nav = w_charge_nav * gate * (phi - last_phi)
        self.last_d_charger_norm = d_charger_norm

        # Actual charging reward: battery increased (skip step 1 to avoid init spike)
        # 成功充电奖励：电量回升时给予正奖励（跳过首步避免初始化尖刺）
        battery_gained = max(0, self.battery - self.last_battery) if self.step_no > 1 else 0
        r_charge_act = w_charge_act * float(battery_gained > 0) if battery_ratio < 0.95 else 0.0

        # Step penalty: fixed small negative to encourage efficiency
        # 步数惩罚：固定小负值，鼓励高效清扫
        step_penalty = -0.0005

        # NPC avoid: soft penalty inside safe radius / NPC 躲避：安全半径内软惩罚
        safe_radius = 2.5
        d_npc = float(self.d_npc_min)
        s = max(0.0, (safe_radius - d_npc) / safe_radius)
        r_npc = -w_npc * (s * s)

        total = float(r_clean + r_charge_nav + r_charge_act + r_npc + r_new + r_repeat + r_streak + r_idle + step_penalty)

        self.last_reward_components = {
            "cleaning": float(r_clean),
            "charge_nav": float(r_charge_nav),
            "charge_act": float(r_charge_act),
            "npc_avoid": float(r_npc),
            "explore_new": float(r_new),
            "explore_repeat": float(r_repeat),
            "eff_streak": float(r_streak),
            "eff_idle": float(r_idle),
            "step_penalty": float(step_penalty),
            "d_charger_norm": float(d_charger_norm),
            "battery_gate": float(gate),
            "total": total,
        }

        return total
