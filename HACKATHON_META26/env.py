import gymnasium as gym
import numpy as np
from gymnasium import spaces

class DisasterEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, n_cities=8, n_teams=4, max_steps=72):
        super().__init__()
        self.n_cities = n_cities
        self.n_teams = n_teams
        self.max_steps = max_steps

        self.city_names = [
            "Chennai", "Mumbai", "Delhi",
            "Bengaluru", "Mysuru", "Hyderabad",
            "Kochi", "Coorg"
        ]
        self.city_types = np.array([1, 1, 3, 3, 2, 0, 1, 2])

        self.city_population = np.array([
            87, 205, 320, 143, 9, 98, 21, 2
        ], dtype=np.float32)
        self.pop_normalized = self.city_population / self.city_population.max()

        self.city_elevation = np.array([
            6, 14, 216, 920, 763, 542, 9, 1525
        ], dtype=np.float32)
        self.elev_normalized = self.city_elevation / self.city_elevation.max()

        self.neighbors = {
            0: [1, 2], 1: [0, 6], 2: [0, 3, 5],
            3: [2, 4, 7], 4: [3, 7], 5: [2, 3],
            6: [1, 4], 7: [3, 4]
        }

        self.power_grid = {
            0: [1, 2], 1: [0, 6], 2: [0, 3, 5],
            3: [2, 4, 7], 4: [3, 7], 5: [2, 3],
            6: [1, 4], 7: [3, 4]
        }

        self.disaster_names = [
            "none", "fire", "flood", "earthquake",
            "lightning", "tsunami", "compound", "epidemic", "aftershock"
        ]

        self.disaster_growth = {
            0: 0.000,
            1: 0.080,
            2: 0.050,
            3: 0.030,
            4: 0.120,
            5: 0.100,
            6: 0.150,
            7: 0.060,
            8: 0.040,
        }

        self.disaster_resistance = {
            0: 0.00,
            1: 0.30,
            2: 0.25,
            3: 0.20,
            4: 0.15,
            5: 0.20,
            6: 0.10,
            7: 0.20,
            8: 0.25,
        }

        obs_per_city = 10
        global_obs = 5
        obs_size = n_cities * obs_per_city + global_obs

        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(obs_size,),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(n_cities * 2 + 2)
        self.cascade_log = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.severity           = self.np_random.uniform(0.05, 0.35, self.n_cities).astype(np.float32)
        self.disaster_type      = self.np_random.integers(0, 5, self.n_cities)
        self.compound_flag      = np.zeros(self.n_cities, dtype=np.float32)
        self.infrastructure_dmg = np.zeros(self.n_cities, dtype=np.float32)
        self.power_out          = np.zeros(self.n_cities, dtype=np.float32)
        self.evacuation_status  = np.zeros(self.n_cities, dtype=np.float32)
        self.disease_risk       = np.zeros(self.n_cities, dtype=np.float32)
        self.team_fatigue       = np.zeros(self.n_cities, dtype=np.float32)

        self.teams_available    = self.n_teams
        self.teams_deployed     = np.zeros(self.n_cities, dtype=int)
        self.supply_level       = 1.0
        self.step_count         = 0
        self.total_saved        = 0
        self.total_casualties   = 0

        self.weather_state      = int(self.np_random.integers(0, 4))
        self.weather_timer      = int(self.np_random.integers(6, 18))

        self.cascade_log        = []
        self.aftershock_timer   = {}

        return self._get_obs(), {}

    def _get_obs(self):
        city_obs = np.stack([
            self.severity,
            self.disaster_type / 8.0,
            self.compound_flag,
            self.infrastructure_dmg,
            self.power_out,
            self.evacuation_status,
            self.disease_risk,
            self.team_fatigue,
            self.pop_normalized,
            self.elev_normalized,
        ], axis=1).flatten()

        time_of_day = (self.step_count % 24) / 24.0
        global_obs = np.array([
            self.teams_available / self.n_teams,
            self.supply_level,
            time_of_day,
            self.step_count / self.max_steps,
            self.weather_state / 3.0,
        ], dtype=np.float32)

        return np.concatenate([city_obs, global_obs]).astype(np.float32)

    def _update_weather(self):
        self.weather_timer -= 1
        if self.weather_timer <= 0:
            self.weather_state = int(self.np_random.integers(0, 4))
            self.weather_timer = int(self.np_random.integers(6, 18))

        for i in range(self.n_cities):
            dtype = int(self.disaster_type[i])
            if self.weather_state == 1:
                if dtype == 1:
                    self.severity[i] = max(0.0, self.severity[i] - 0.03)
                if dtype == 2:
                    self.severity[i] = min(1.0, self.severity[i] + 0.02)
            if self.weather_state == 2:
                self.severity[i] = min(1.0, self.severity[i] + 0.02)
                if dtype == 4:
                    self.severity[i] = min(1.0, self.severity[i] + 0.04)
            if self.weather_state == 3:
                if dtype == 1:
                    self.severity[i] = min(1.0, self.severity[i] + 0.05)

    def _get_time_multiplier(self):
        hour = self.step_count % 24
        if hour >= 20 or hour <= 6:
            return 1.4
        return 1.0

    def _lightning_strikes(self):
        for i in range(self.n_cities):
            if self.disaster_type[i] == 4:
                strike_chance = 0.3
                if self.weather_state == 2:
                    strike_chance = 0.6
                if self.np_random.random() < strike_chance:
                    spike = self.np_random.uniform(0.1, 0.25)
                    self.severity[i] = min(1.0, self.severity[i] + spike)
                    self.power_out[i] = 1.0
                    for neighbor in self.power_grid[i]:
                        if self.np_random.random() < 0.4:
                            self.power_out[neighbor] = min(1.0, self.power_out[neighbor] + 0.5)

    def _trigger_aftershocks(self, city):
        n_aftershocks = self.np_random.integers(1, 4)
        for _ in range(n_aftershocks):
            delay = int(self.np_random.integers(1, 8))
            target_step = self.step_count + delay
            if target_step not in self.aftershock_timer:
                self.aftershock_timer[target_step] = []
            if self.np_random.random() < 0.6:
                target = city
            else:
                target = self.np_random.choice(self.neighbors[city])
            self.aftershock_timer[target_step].append(target)

    def _process_aftershocks(self):
        if self.step_count in self.aftershock_timer:
            for city in self.aftershock_timer[self.step_count]:
                mag = self.np_random.uniform(0.1, 0.3)
                self.severity[city] = min(1.0, self.severity[city] + mag)
                self.disaster_type[city] = 8
                self.cascade_log.append(
                    f"Aftershock hit {self.city_names[city]}!"
                )

    def _update_epidemic(self):
        for i in range(self.n_cities):
            if self.disaster_type[i] == 2 and self.pop_normalized[i] > 0.3:
                self.disease_risk[i] = min(1.0,
                    self.disease_risk[i] + 0.03 * self.severity[i])
            if self.disease_risk[i] > 0.7 and self.disaster_type[i] != 7:
                self.disaster_type[i] = 7
                self.compound_flag[i] = 1.0
                self.cascade_log.append(
                    f"Epidemic outbreak in {self.city_names[i]}!"
                )
            if self.disaster_type[i] == 7:
                for neighbor in self.neighbors[i]:
                    self.disease_risk[neighbor] = min(1.0,
                        self.disease_risk[neighbor] + 0.01)

    def _update_supplies(self, teams_dispatched):
        self.supply_level = max(0.0, self.supply_level - teams_dispatched * 0.02)
        resupply = 0.01
        if np.mean(self.power_out) > 0.5:
            resupply = 0.005
        self.supply_level = min(1.0, self.supply_level + resupply)

    def _trigger_cascades(self):
        new_events = []
        for i in range(self.n_cities):
            sev   = self.severity[i]
            dtype = int(self.disaster_type[i])

            if dtype == 3 and sev > 0.7:
                self._trigger_aftershocks(i)
                for nb in self.neighbors[i]:
                    if self.city_types[nb] == 1:
                        tsunami_mag = sev * 0.6 * (1 - self.elev_normalized[nb])
                        self.severity[nb] = min(1.0, self.severity[nb] + tsunami_mag)
                        self.disaster_type[nb] = 5
                        new_events.append(
                            f"Earthquake->Tsunami: {self.city_names[i]}"
                            f"->{self.city_names[nb]} (mag {tsunami_mag:.2f})"
                        )

            if dtype == 1 and sev > 0.8:
                for nb in self.neighbors[i]:
                    if self.city_types[nb] == 2:
                        spread = 0.3 * (1 + (self.weather_state == 3) * 0.3)
                        self.severity[nb] = min(1.0, self.severity[nb] + spread)
                        self.disaster_type[nb] = 1
                        new_events.append(
                            f"Fire spread: {self.city_names[i]}"
                            f"->{self.city_names[nb]}"
                        )

            if dtype == 4 and sev > 0.6:
                if self.np_random.random() < 0.4:
                    self.compound_flag[i] = 1.0
                    self.disaster_type[i] = 6
                    new_events.append(
                        f"Lightning->Compound: {self.city_names[i]}"
                    )

            if dtype in [2, 5] and sev > 0.75:
                self.infrastructure_dmg[i] = min(1.0,
                    self.infrastructure_dmg[i] + 0.15)
                if self.elev_normalized[i] < 0.1:
                    self.infrastructure_dmg[i] = min(1.0,
                        self.infrastructure_dmg[i] + 0.1)
                new_events.append(
                    f"Infrastructure collapse: {self.city_names[i]}"
                )

            if dtype == 6 and sev > 0.85:
                for nb in self.neighbors[i]:
                    self.compound_flag[nb] = 1.0
                    self.severity[nb] = min(1.0, self.severity[nb] + 0.15)
                    new_events.append(
                        f"Compound cascade: {self.city_names[i]}"
                        f"->{self.city_names[nb]}"
                    )

            if self.power_out[i] > 0.7:
                for nb in self.neighbors[i]:
                    self.infrastructure_dmg[nb] = min(1.0,
                        self.infrastructure_dmg[nb] + 0.05)

        self.cascade_log.extend(new_events)
        return new_events

    def _update_fatigue(self):
        for i in range(self.n_cities):
            if self.teams_deployed[i] > 0:
                self.team_fatigue[i] = min(1.0, self.team_fatigue[i] + 0.05)
            else:
                self.team_fatigue[i] = max(0.0, self.team_fatigue[i] - 0.03)

    def state(self):
        """Returns current full state as a dict — required by OpenEnv spec"""
        return {
            "severity":           self.severity.tolist(),
            "disaster_type":      [int(x) for x in self.disaster_type],
            "disaster_names":     [self.disaster_names[int(t)] for t in self.disaster_type],
            "city_names":         self.city_names,
            "compound_flag":      self.compound_flag.tolist(),
            "infrastructure_dmg": self.infrastructure_dmg.tolist(),
            "power_out":          self.power_out.tolist(),
            "evacuation_status":  self.evacuation_status.tolist(),
            "disease_risk":       self.disease_risk.tolist(),
            "team_fatigue":       self.team_fatigue.tolist(),
            "teams_available":    int(self.teams_available),
            "supply_level":       float(self.supply_level),
            "total_saved":        int(self.total_saved),
            "total_casualties":   int(self.total_casualties),
            "weather":            ["clear","rain","storm","heatwave"][int(self.weather_state)],
            "time_of_day":        f"{self.step_count % 24:02d}:00",
            "step":               int(self.step_count),
        }

    def step(self, action):
        reward = 0.0
        teams_dispatched = 0
        time_mult = self._get_time_multiplier()

        if action == 0:
            pass

        elif action == self.action_space.n - 1:
            self.team_fatigue = np.maximum(0, self.team_fatigue - 0.15)
            reward += 5.0

        elif action > self.n_cities:
            city = action - self.n_cities - 1
            if self.teams_available > 0:
                self.evacuation_status[city] = min(1.0,
                    self.evacuation_status[city] + 0.3)
                self.teams_available -= 1
                teams_dispatched += 1
                saved = int(self.pop_normalized[city] * 50 * self.severity[city])
                self.total_saved += saved
                reward += float(saved) * 0.15
                self.teams_available = min(self.n_teams, self.teams_available + 1)

        elif action > 0:
            city = action - 1
            if self.teams_available > 0 and self.supply_level > 0.05:
                dtype     = int(self.disaster_type[city])
                fatigue_pen = self.team_fatigue[city]
                infra_pen   = self.infrastructure_dmg[city]
                night_pen   = 0.2 if time_mult > 1.0 else 0.0
                power_pen   = 0.15 * self.power_out[city]

                effective = self.disaster_resistance[dtype] * (
                    1 - fatigue_pen * 0.4
                      - infra_pen   * 0.3
                      - night_pen
                      - power_pen
                )
                effective = max(0.05, effective)

                self.teams_deployed[city] += 1
                self.teams_available -= 1
                teams_dispatched += 1

                saved = int(self.severity[city] * self.pop_normalized[city] * 100 * effective)
                self.severity[city] = max(0.0, self.severity[city] - effective)
                self.total_saved += saved
                reward += float(saved) * 0.1

                if self.disaster_type[city] == 7:
                    self.disease_risk[city] = max(0.0, self.disease_risk[city] - 0.1)

                self.infrastructure_dmg[city] = max(0.0, self.infrastructure_dmg[city] - 0.05)
                self.power_out[city]          = max(0.0, self.power_out[city] - 0.1)

                if self.severity[city] < 0.05:
                    self.compound_flag[city]  = 0.0
                    self.disaster_type[city]  = 0

                self.teams_deployed[city]  = max(0, self.teams_deployed[city] - 1)
                self.teams_available       = min(self.n_teams, self.teams_available + 1)

        self._lightning_strikes()
        self._process_aftershocks()
        self._update_weather()
        self._update_epidemic()
        self._update_fatigue()
        self._update_supplies(teams_dispatched)
        cascade_events = self._trigger_cascades()

        for i in range(self.n_cities):
            if self.teams_deployed[i] == 0 and self.severity[i] > 0:
                dtype  = int(self.disaster_type[i])
                growth = float(self.disaster_growth[dtype]) * float(time_mult)
                if self.compound_flag[i]:
                    growth *= 1.5
                if self.power_out[i] > 0.5:
                    growth *= 1.2
                self.severity[i] = min(1.0, self.severity[i] + growth)
                casualties = int(self.severity[i] * self.pop_normalized[i] * 5)
                self.total_casualties += casualties
                reward -= float(self.severity[i]) * 20.0 * float(self.pop_normalized[i])

        reward -= 15.0 * float(len(cascade_events))
        if self.supply_level < 0.2:
            reward -= 10.0
        reward -= float(np.sum(self.power_out)) * 5.0

        self.step_count += 1
        terminated = False
        truncated  = bool(self.step_count >= self.max_steps)

        if truncated:
            reward += (1.0 - float(np.mean(self.severity)))           * 100.0
            reward += (1.0 - float(np.mean(self.infrastructure_dmg))) * 30.0
            reward += (1.0 - float(np.mean(self.disease_risk)))       * 30.0
            reward += float(self.supply_level)                         * 20.0
            reward -= float(self.total_casualties)                     * 0.1

        reward = float(reward)

        info = {
            "severity":           self.severity.copy(),
            "disaster_type":      self.disaster_type.copy(),
            "disaster_names":     [self.disaster_names[int(t)] for t in self.disaster_type],
            "city_names":         list(self.city_names),
            "compound_flag":      self.compound_flag.copy(),
            "infrastructure_dmg": self.infrastructure_dmg.copy(),
            "power_out":          self.power_out.copy(),
            "evacuation_status":  self.evacuation_status.copy(),
            "disease_risk":       self.disease_risk.copy(),
            "team_fatigue":       self.team_fatigue.copy(),
            "teams_available":    int(self.teams_available),
            "supply_level":       float(self.supply_level),
            "total_saved":        int(self.total_saved),
            "total_casualties":   int(self.total_casualties),
            "weather":            ["clear", "rain", "storm", "heatwave"][int(self.weather_state)],
            "cascade_events":     list(cascade_events),
            "time_of_day":        f"{self.step_count % 24:02d}:00",
        }

        return self._get_obs(), reward, terminated, truncated, info
    def grade_task(self, task_id: str, info: dict) -> float:
        """Programmatic grader returning score in [0.0, 1.0]"""
        saved      = info.get("total_saved", 0)
        casualties = info.get("total_casualties", 1)
        severity   = info.get("severity", np.ones(self.n_cities))
        avg_sev    = float(np.mean(severity))
        supply     = info.get("supply_level", 0.0)

        if task_id == "easy":
            score = max(0.0, 1.0 - avg_sev)
        elif task_id == "medium":
            ratio = saved / max(saved + casualties, 1)
            score = ratio * 0.7 + (supply / 1.0) * 0.3
        else:  # hard
            ratio = saved / max(saved + casualties, 1)
            sev_score = max(0.0, 1.0 - avg_sev)
            score = ratio * 0.5 + sev_score * 0.3 + (supply / 1.0) * 0.2

        return float(max(0.0, min(1.0, score)))

