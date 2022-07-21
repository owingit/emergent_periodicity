import collections
import math
import pickle
import random

import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FixedLocator, FixedFormatter
from scipy.signal import find_peaks, peak_prominences

import Firefly
import obstacle
import simulation_helpers

IS_TEST = False


class Simulation:
    def __init__(self, num_agents, side_length, step_count, thetastar, coupling_strength, Tb,
                 beta, phrase_duration, epsilon_delta, r_or_u="uniform", use_linear=False, one_flash=False,
                 timestepsize=0.1, no_refrac=False, cutoff='min', use_obstacles=False, use_kuramato=True):
        self.firefly_array = []
        self.timestepsize = timestepsize
        self.use_kuramato = use_kuramato
        self.use_integrate_and_fire = not self.use_kuramato
        self.beta = beta
        self.phrase_duration = phrase_duration
        self.use_linear = use_linear
        self.one_flash = one_flash
        self.no_refrac = no_refrac

        # constants set by run.py
        self.total_agents = num_agents
        self.n = side_length
        self.coupling_strength = coupling_strength
        self.alpha = 2
        self.Tb = Tb
        self.steps = step_count
        self.r_or_u = r_or_u
        self.tstar_seed = thetastar
        self.epsilon_delta = epsilon_delta
        thetastars = [np.linspace(-thetastar, thetastar, simulation_helpers.TSTAR_RANGE)]
        self.thetastar = list(thetastars[random.randint(0, len(thetastars) - 1)])
        self.use_obstacles = use_obstacles

        self.has_run = False
        self.obstacles = None
        if self.use_obstacles is True:
            self.init_obstacles()

        # initialize all Firefly agents
        for i in range(0, self.total_agents):
            self.firefly_array.append(Firefly.Firefly(
                i, total=self.total_agents, tstar=self.thetastar,
                tstar_range=simulation_helpers.TSTAR_RANGE,
                n=self.n, steps=self.steps, r_or_u=self.r_or_u,
                beta=beta,
                phrase_duration=phrase_duration,
                epsilon_delta=epsilon_delta,
                use_periodic_boundary_conditions=False,
                use_linear=use_linear,
                one_flash=one_flash,
                no_refrac=no_refrac,
                timestepsize=0.1,
                cutoff=cutoff,
                tb=self.Tb, obstacles=self.obstacles)
            )
        if self.use_kuramato:
            self.boilerplate = '({}density, {}rad natural frequency)'.format(self.total_agents / (self.n * self.n),
                                                                             self.Tb)
        else:
            self.boilerplate = '{}density, {}beta, {}Tb'.format(self.total_agents / (self.n * self.n),
                                                                beta, phrase_duration)

        # network stuff
        self.use_networks = False

        if self.use_networks:
            self.delta_t = 10 * [ff.charging_time + ff.discharging_time for ff in [self.firefly_array[0]]][0]
            self.delta_x = {}
            self.connection_probability = None
            self.cascade_networks = {}
            self.indices_in_cascade_ = {}
            self.networks_in_cascade_ = {}
            self.connected_temporal_networks = {}

        if self.use_obstacles:
            self.boilerplate = self.boilerplate + '_obstacles'

        self.init_stats()

    def init_obstacles(self):
        """Initialize an array of obstacles randomly placed throughout the arena."""
        num_obstacles = random.randint(10, 20)
        obstacle_generator = obstacle.ObstacleGenerator(num_obstacles, self.n)
        self.obstacles = obstacle_generator.get_obstacles()

    def init_stats(self):
        """Initialize per-timestep dictionaries tracking firefly phase and TODO: more things."""
        if self.use_kuramato:
            for i in range(self.steps):
                self.num_fireflies_with_phase_x[i] = {key: 0 for key in range(0, 360)}
                self.wave_statistics[i] = {}
        initial_flashers = [(ff.positionx[0], ff.positiony[0]) for ff in self.firefly_array if ff.flashed_at_this_step[0]]

        if self.use_networks:
            centroid = None
            if initial_flashers:
                centroid = simulation_helpers.centroid(initial_flashers)
            for i in range(int(self.delta_t)):
                if centroid:
                    k_mean_differences = [np.sqrt((x - centroid[0]) ** 2 + (y - centroid[1]) ** 2) for (x, y, _) in
                                          initial_flashers]
                    self.delta_x[i] = np.mean(k_mean_differences)
                else:
                    self.delta_x[i] = math.sqrt(self.delta_t)
            self.connection_probability = 1 / max(list(self.delta_x.keys()))

        if self.use_kuramato:
            phase_zero_fireflies = []
            for firefly in self.firefly_array:
                phase_in_degrees = int(math.degrees(firefly.phase[0]))
                if phase_in_degrees < 0:
                    phase_in_degrees += 360
                if 0 <= phase_in_degrees < 1 or 359 < phase_in_degrees <= 360:
                    phase_zero_fireflies.append(firefly)
                self.num_fireflies_with_phase_x[0][phase_in_degrees] += 1

            if phase_zero_fireflies:
                phase_zero_fit = np.polyfit([ff.positionx[0] for ff in phase_zero_fireflies],
                                            [ff.positiony[0] for ff in phase_zero_fireflies],
                                            1)
            else:
                phase_zero_fit = [0, 0]
            self.wave_statistics[0]['count'] = len(phase_zero_fireflies)
            self.wave_statistics[0]['regression'] = phase_zero_fit

    def run(self):
        """
        Run the simulation. At each timestep, a firefly moves in relation to obstacles present and
        experiences phase interactions, either by slightly modified Kuramato model interactions or
        """
        logging = False
        for step in range(1, self.steps):
            if logging:
                print(step)
            if step % 5000 == 0:
                print(step)
            for firefly in self.firefly_array:
                firefly.move(step, self.obstacles)
            if self.use_kuramato:
                self.kuramato_phase_interactions(step)

            if self.use_integrate_and_fire:
                self.lco_interactions(step)

        self.has_run = True

    def look(self, step):
        """Find neighbors in line of sight. Set limited to true to explore varying the FoV.

        Options:
        1. All-to-all adjacency matrix.
        2. Attention-limited adjacency matrix (draw a hemisphere 90 degrees to either side of direction)
        3. Obstacle-limited adjacency matrix (obstacles impede attention)
        """

        def get_ordered_list(points, x, y):
            points.sort(key=lambda p: (p.x - x) ** 2 + (p.y - y) ** 2)
            return points
        limited = False
        distance = False
        neighbors = {}
        for i in range(0, len(self.firefly_array)):
            ff_i = self.firefly_array[i]
            neighbors[ff_i.number] = []
            for j in range(0, self.total_agents):
                if i == j:
                    # same agent
                    continue
                else:
                    ff_j = self.firefly_array[j]
                    if self.obstacles:
                        skip_dist = self.evaluate_obstacles(ff_i, ff_j, step)
                        if skip_dist:
                            continue
                        else:
                            if ff_j.flashed_at_this_step[step - 1]:
                                neighbors[ff_i.number].append(ff_j)
                    elif distance:
                        d = (ff_i.positionx[step] - ff_j.positionx[step]) ** 2 + (ff_i.positiony[step]- ff_j.positiony[step]) ** 2
                        if ff_j.flashed_at_this_step[step - 1]:
                            neighbors[ff_i.number].append((ff_j, d))

                    elif limited:
                        i_direction = ff_i.direction[step]
                        i_right_range = i_direction + (math.pi / 2)
                        i_left_range = i_direction - (math.pi / 2)
                        a = [ff_i.positionx[step], ff_i.positiony[step]]
                        b = [ff_j.positionx[step], ff_j.positiony[step]]

                        if i_left_range < math.atan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b)) < i_right_range:
                            if ff_j.flashed_at_this_step[step - 1]:
                                neighbors[ff_i.number].append(ff_j)
                    else:
                        if ff_j.flashed_at_this_step[step - 1]:
                            neighbors[ff_i.number].append(ff_j)

        return neighbors

    def listen(self, step, neighbors):
        """Update voltages based on adjacency matrix"""
        for i in range(0, self.total_agents):
            ff_i = self.firefly_array[i]
            dvt = ff_i.set_dvt(step, ff_i.in_burst)
            neighbors_of_i = neighbors[ff_i.number]
            beta_addition = 0
            if neighbors_of_i:
                for ff_j in neighbors_of_i:
                    # beta_addition += 1 / (d ** ff_j.beta)
                    beta_addition += (ff_i.beta) * (1 - ff_j.is_charging)
            if ff_i.in_burst:
                beta_addition = 0
            elif (step - ff_i.last_flashed_at) < ((ff_i.charging_time + ff_i.discharging_time) * ff_i.flashes_per_burst):
                beta_addition = 0
            voltage_at_step = ff_i.voltage_instantaneous[step - 1] + (dvt + (ff_i.sign * beta_addition))
            ff_i.voltage_instantaneous[step] = min([ff_i.discharging_threshold, voltage_at_step])

    def lco_interactions(self, step):
        """
        At timestep step:
        A) The neighbor set is chosen in _look_ (here it is all-to-all)
        B) Agents receive pulsatile inputs from their neighbors in _listen_
        C) Agents update their voltages and epsilon parameters in _update_epsilon_and_readiness_.
        """
        neighbors = self.look(step)
        self.listen(step, neighbors)
        self.update_epsilon_and_readiness(step, neighbors)

    def update_epsilon_and_readiness(self, step, neighbors):
        """Set epsilon based on voltage for all fireflies, flash if possible."""
        for i in range(0, self.total_agents):
            ff_i = self.firefly_array[i]

            # update epsilon to discharging (V is high enough)
            if ff_i.voltage_instantaneous[step] >= ff_i.discharging_threshold:
                if ff_i.in_burst is False and ff_i.sign == 1:
                    if ff_i.flashes_per_burst == 1 and step - ff_i.last_flashed_at > (ff_i.discharging_time / ff_i.timestepsize):
                        # on its own
                        ff_i.set_ready(step)
                    elif ff_i.flashes_per_burst > 1:
                        ff_i.set_ready(step)
                # elif ff_i.in_burst is True and ff_i.sign == 1:
                #     # in a burst
                #     #ff_i.unset_ready()
                # else:
                #     #ff_i.unset_ready()
                if ff_i.sign == 1:
                    ff_i.is_charging = 0
                    ff_i.epsilon[step] = 0

            # update epsilon to charging if agent flashes
            elif ff_i.voltage_instantaneous[step] < ff_i.charging_threshold:
                #if ff_i.in_burst is False:
                ff_i.is_charging = 1
                ff_i.epsilon[step] = 1
                ff_i.unset_ready()
            self.flash_conditions(ff_i, step)

    @staticmethod
    def flash_conditions(ff, t):
        """When to flash for firefly ff at step t"""
        if ff.is_charging == 0 and ff.ready and not ff.in_burst:
            ff.flash(t)

        elif ff.in_burst is True and t - ff.last_flashed_at > (ff.discharging_time + ff.charging_time):
            ff.flash(t)

    def evaluate_obstacles(self, ff_i, ff_j, step):
        """Agents ff_i and ff_j see if obstacles are between them.

        :param ff_i, ff_j: firefly objects
        :param step: timestep
        :returns whether these agents should skip their interaction (True means obstacle was between them)"""
        skip = False
        line = simulation_helpers.generate_line_points(
            (ff_i.positionx[step], ff_i.positiony[step]),
            (ff_j.positionx[step], ff_j.positiony[step]),
            num_points=100
        )
        for obstacle in self.obstacles:
            if not skip:
                for xy in line:
                    if obstacle.contains(xy[0], xy[1]):
                        skip = True
                        break
        return skip

    def kuramato_phase_interactions(self, step):
        """
        OUTDATED.
        Each firefly's phase wave interacts with the phase wave of its detectable neighbors by the Kuramato model.
        """
        for i in range(0, self.total_agents):
            ff_i = self.firefly_array[i]
            kuramato = 0
            for j in range(0, self.total_agents):
                if i != j:
                    ff_j = self.firefly_array[j]
                    if self.obstacles:
                        skip_dist = self.evaluate_obstacles(ff_i, ff_j, step)
                        if skip_dist:
                            continue
                        else:
                            dist = ((ff_j.positionx[step] - ff_i.positionx[step]) ** 2 +
                                    (ff_j.positiony[step] - ff_i.positiony[step]) ** 2) ** 0.5

                    else:
                        dist = ((ff_j.positionx[step] - ff_i.positionx[step]) ** 2 +
                                (ff_j.positiony[step] - ff_i.positiony[step]) ** 2) ** 0.5
                    if dist != 0:
                        kuramato_term = math.sin(ff_j.phase[step - 1] - ff_i.phase[step - 1]) / dist
                        kuramato += kuramato_term

            coupling_term = (ff_i.phase[step - 1] + self.coupling_strength * kuramato)
            ff_i.phase[step] = (ff_i.nat_frequency + coupling_term) % math.radians(360)
            phase_key = int(math.degrees(ff_i.phase[step]))
            if phase_key < 0:
                phase_key += 360
            self.num_fireflies_with_phase_x[step][phase_key] += 1
        phase_zero_fireflies = [ff
                                for ff in self.firefly_array
                                if 0 <= ff.phase[step] < 1 or 359 < ff.phase[step] <= 360
                                # or ((ff.phase[step-1] + ff.phase[step] - 2 * (360 - ff.phase[step-1])) % 360 <= math.degrees(self.Tb))
                                ]
        if phase_zero_fireflies:
            phase_zero_fit = np.polyfit(
                [ff.positionx[step] for ff in phase_zero_fireflies],
                [ff.positiony[step] for ff in phase_zero_fireflies],
                1)
        else:
            phase_zero_fit = self.wave_statistics[step - 1]['regression']
        self.wave_statistics[step]['count'] = len(phase_zero_fireflies)
        self.wave_statistics[step]['regression'] = phase_zero_fit
        ff_phases = [ff.phase[step] for ff in self.firefly_array]
        mean_resultant_vector_length = self.circ_r(np.array(ff_phases))
        self.mean_resultant_vector_length[step] = float(mean_resultant_vector_length)

    def animate_phase_bins(self, now, write_gif=False, show_gif=False):
        """Animate the # of ff's in each phase (0 -> 2*pi) over time."""
        assert self.has_run, "Animation cannot render until the simulation has been run!"
        plt.style.use('seaborn-pastel')
        num_bins = 360
        fig = plt.figure()
        ax = plt.axes(xlim=(0, num_bins), ylim=(0, self.total_agents+1))
        ax.set_xlim([0.0, 360])
        ax.set_xlabel('Phase theta in degrees')
        x_formatter = FixedFormatter([
            "0°", "45°", "90°", "135°", "180°", "225°", "270°", "315°", "360°"])
        x_locator = FixedLocator([0, 45, 90, 135, 180, 225, 270, 315, 360])
        ax.xaxis.set_major_formatter(x_formatter)
        ax.xaxis.set_major_locator(x_locator)

        ax.set_ylim([0.0, self.total_agents+1])
        ax.set_ylabel('Num agents')
        rects = ax.bar(range(num_bins), self.num_fireflies_with_phase_x[0].values(), align='center', color='blue')

        def animate(i, data):
            for rect, n in zip(rects, data[i].keys()):
                rect.set_height(data[i][n])
            ax.set_title('Num agents with particular phase at step {}'.format(i) + self.boilerplate)

        anim = FuncAnimation(fig, animate, frames=self.steps, fargs=[self.num_fireflies_with_phase_x],
                             interval=50, blit=False, repeat=False)

        save_string = self.set_save_string('numphaseovertime', now)
        if write_gif:
            writervideo = animation.FFMpegWriter(fps=10)
            anim.save(save_string, writer=writervideo)
        if show_gif:
            plt.show()
        plt.clf()

    def animate_walk(self, now, write_gif=False, show_gif=False):
        """Animate the 2d correlated random walks of all fireflies, colored by phase."""
        assert self.has_run, "Animation cannot render until the simulation has been run!"
        plt.style.use('seaborn-pastel')
        plt.clf()
        fig = plt.figure()
        ax = plt.axes(xlim=(0, self.n), ylim=(0, self.n))
        if self.use_kuramato:
            color_dict = self.setup_color_legend(ax, self.use_kuramato, self.use_integrate_and_fire)

        xdatas = {n: [] for n in range(0, self.total_agents)}
        ydatas = {n: [] for n in range(0, self.total_agents)}

        firefly_paths = [ax.plot([], [], '*')[0] for _ in self.firefly_array]

        def animate(i, flies, lines):
            """Animate walks."""
            for line, fly in zip(lines, flies):
                if not fly.trace.get(i):
                    step_key = str(i)
                else:
                    step_key = i
                xdatas[fly.number].append(fly.trace.get(step_key)[0])
                ydatas[fly.number].append(fly.trace.get(step_key)[1])
                line.set_data(xdatas[fly.number][0], ydatas[fly.number][0])
                if self.use_kuramato:
                    deg = math.degrees(fly.phase[i])
                    if deg < 0:
                        deg += 360
                    line.set_color(color_dict[int(deg)])
                if self.use_integrate_and_fire:
                    if fly.flashed_at_this_step[i]:
                        line.set_color('red')
                    else:
                        line.set_color('blue')
                xdatas[fly.number].pop(0)
                ydatas[fly.number].pop(0)

            if self.use_kuramato:
                title_str = "Kuramato Model"
            else:
                title_str = "Integrate-and-Fire Model"
            ax.set_title('2D Walk {} Interactions (step {})'.format(title_str, i) + self.boilerplate)
            return lines

        ax.set_xlim([0.0, self.n])
        ax.set_xlabel('X')

        ax.set_ylim([0.0, self.n])
        ax.set_ylabel('Y')
        if self.obstacles:
            for obstacle in self.obstacles:
                ax.add_artist(plt.Circle((obstacle.centerx, obstacle.centery), obstacle.radius))

        interval = 50 if self.use_kuramato else 300
        anim = FuncAnimation(fig, animate, frames=self.steps,
                             fargs=(self.firefly_array, firefly_paths),
                                    # uncomment for velocity line
                                    # self.wave_statistics,
                                    # regression_line),
                             interval=interval, blit=False)

        save_string = self.set_save_string('phaseanim', now)
        if write_gif:
            writervideo = animation.FFMpegWriter(fps=10)
            anim.save(save_string, writer=writervideo)
            plt.close()
        if show_gif:
            plt.show()

    def set_save_string(self, plot_type, now, path):
        """Sets up default save string."""
        if 'phaseanim' in plot_type or 'numphaseovertime' in plot_type:
            end = '.mp4'
        elif 'burst_dict' in plot_type:
            end = '.pickle'
        else:
            end = '.png'
        if self.use_obstacles:
            if not path:
                save_string = 'data/{}_{}agents_{}x{}_beta={}_Tb={}_k={}_steps={}_{}distribution{}_obstacles{}'.format(
                    plot_type,
                    self.total_agents,
                    self.n, self.n,
                    self.beta,
                    self.phrase_duration,
                    self.coupling_strength,
                    self.steps,
                    self.r_or_u,
                    str(now).replace(' ', '_'),
                    end
                )
            else:
                save_string = '{}/{}_{}agents_{}x{}_beta={}_Tb={}_k={}_steps={}_{}distribution{}{}'.format(
                    path,
                    plot_type,
                    self.total_agents,
                    self.n, self.n,
                    self.beta,
                    self.phrase_duration,
                    self.coupling_strength,
                    self.steps,
                    self.r_or_u,
                    str(now).replace(' ', '_'),
                    end
                )
        else:
            if not path:
                save_string = 'data/{}_{}agents_{}x{}_beta={}_Tb={}_k={}_steps={}_{}distribution{}_obstacles{}'.format(
                    plot_type,
                    self.total_agents,
                    self.n, self.n,
                    self.beta,
                    self.phrase_duration,
                    self.coupling_strength,
                    self.steps,
                    self.r_or_u,
                    str(now).replace(' ', '_'),
                    end
                )
            else:
                save_string = '{}{}_{}agents_{}x{}_beta={}_Tb={}_k={}_steps={}_{}distribution{}{}'.format(
                    path,
                    plot_type,
                    self.total_agents,
                    self.n, self.n,
                    self.beta,
                    self.phrase_duration,
                    self.coupling_strength,
                    self.steps,
                    self.r_or_u,
                    str(now).replace(' ', '_'),
                    end
            )
        return save_string

    def save_time_series(self, now, instance, path=None):
        bursts_at_each_timestep = self.get_burst_data()
        path = '/Users/om/Desktop/school/Chaotic_Dynamics'
        save_string = self.set_save_string('burst_dict{}'.format(instance), now, path)
        with open(save_string, 'wb') as f:
            pickle.dump(bursts_at_each_timestep, f, pickle.HIGHEST_PROTOCOL)

    def save_voltage_trajectories(self, now, instance):
        for ff in self.firefly_array:
            filename = '/Users/om/Desktop/school/Chaotic_Dynamics/{}_beta_ff{}_in_sim_with_{}_voltages.txt'.format(
                self.beta,
                ff.number,
                len(self.firefly_array))
            to_save = {}
            for i, v in enumerate(ff.voltage_instantaneous):
                to_save[i*self.timestepsize] = v
            voltage_trajectory = np.array(list(to_save.items()))
            with open(filename, 'w') as f:
                np.savetxt(f, voltage_trajectory)

    def plot_bursts(self, now, instance, write_gif=False, show_gif=False, shared_ax=None, last_highest=0):
        """Plot the flash bursts over time"""
        assert self.has_run, "Plot cannot render until the simulation has been run!"
        plt.style.use('seaborn-pastel')
        retval = {}

        if self.use_obstacles:
            color = 'green'
            label = 'No adoption of fastest'
        else:
            color = 'blue'
            label = 'Adoption of fastest'
        if not shared_ax:
            ax = plt.axes(xlim=(0, self.steps), ylim=(0, self.total_agents))
            bursts_at_each_timestep = self.get_burst_data()
            # bursts_at_each_timestep = self.get_burst_data_epsilon()
            xs = [(t * self.timestepsize) for t in list(bursts_at_each_timestep.keys())]
            ax.plot(xs, list(bursts_at_each_timestep.values()),
                    label=label, color='steelblue', lw=1)
            retval = dict(zip(xs, list(bursts_at_each_timestep.values())))
            ax.set_xlim(0, 300)
            ax.set_xlabel('T[s]')
            ax.set_ylim([0.0, 20])
            ax.set_ylabel('Number of flashes')
            plt.title('Time series {}ff'.format(self.total_agents))
            save_string = self.set_save_string('flashplot_no_refrac_{}'.format(instance), now, path='time_series_plots/')

            if write_gif:
                plt.savefig(save_string)
                #plt.close()
        else:

            bursts_at_each_timestep = self.get_burst_data()
            shared_ax.plot(list(bursts_at_each_timestep.keys()), list(bursts_at_each_timestep.values()),
                           label=label, color=color)

        if not shared_ax:
            save_string = self.set_save_string('flashplot_{}'.format(instance), now, path='time_series_plots/')

            if write_gif:
                plt.savefig(save_string)
                plt.close()
            if show_gif:
                plt.show()
        return retval

    def raster(self):
        colormap = cm.get_cmap('plasma', len(self.firefly_array))
        fig, ax = plt.subplots()
        for i, ff in enumerate(self.firefly_array):
            xs = [x * self.timestepsize for x, y in enumerate(ff._get_flashed_at_this_step())]
            ax.plot(xs, [int(f) + i for f in ff._get_flashed_at_this_step()], color=colormap.colors[i])
        ax.set_title(self.beta)

    def combined_raster(self):
        fig, ax = plt.subplots()
        xs = [x * self.timestepsize for x, y in enumerate(self.firefly_array[0]._get_flashed_at_this_step())]
        ys = []
        firefly_flashes = {}
        for i,ff in enumerate(self.firefly_array):
            firefly_flashes[i] = ff._get_flashed_at_this_step()
        for step in range(self.steps):
            num = 0
            for i,_ in enumerate(self.firefly_array):
                num += firefly_flashes[i][step]
            if num > 1:
                ys.append(num)
            else:
                ys.append(0)
        ax.plot(xs, ys)
        ax.set_title(self.beta)
        plt.show()

    def voltage_raster(self):
        if len(self.firefly_array) == 2:
            colors = ['black', 'grey']
            fig, ax = plt.subplots()
            for i, ff in enumerate(self.firefly_array):
                xs = [x * self.timestepsize for x, y in enumerate(ff.voltage_instantaneous)]
                ax.plot(xs, [f for f in ff.voltage_instantaneous], color=colors[i])
        else:
            colormap = cm.get_cmap('plasma', len(self.firefly_array))
            fig, ax = plt.subplots()
            for i, ff in enumerate(self.firefly_array):
                xs = [x * self.timestepsize for x, y in enumerate(ff.voltage_instantaneous)]
                ax.plot(xs, [f + i for f in ff.voltage_instantaneous], color=colormap.colors[i])

    def voltage_flash_raster(self):
        colormap = cm.get_cmap('plasma', 20)#len(self.firefly_array))
        fig, ax = plt.subplots()
        for i, ff in enumerate(self.firefly_array):
            xs = [x * self.timestepsize for x, y in enumerate(ff.voltage_instantaneous)]
            x_s = [x * self.timestepsize for x, y in enumerate(ff._get_flashed_at_this_step())]
            ax.plot(xs, [f + i for f in ff.voltage_instantaneous], color=colormap.colors[i])
            ax.plot(x_s, [int(f) + i for f in ff._get_flashed_at_this_step()], color=colormap.colors[i], alpha=0.5)

    @staticmethod
    def setup_color_legend(axis, use_kuramato=True, use_integrate_and_fire=False):
        """Set the embedded color axis for the 2d correlated random walk that shows color-phase relations."""
        if use_kuramato:
            steps = 360

            color_dict = {}
            cmap_seed = matplotlib.cm.get_cmap('hsv', steps)
            norm = matplotlib.colors.Normalize(0, steps)
            display_axes = axis.inset_axes(bounds=[0.79, 0.01, 0.21, 0.01])

            cb = matplotlib.colorbar.ColorbarBase(display_axes,
                                                  cmap=matplotlib.cm.get_cmap('hsv', steps),
                                                  norm=norm,
                                                  orientation='horizontal')
            cb.outline.set_visible(False)
            x_formatter = FixedFormatter([
                "0°", "90°", "180°", "270°", "360°"])
            x_locator = FixedLocator([0, 90, 180, 270, 360])
            display_axes.xaxis.tick_top()
            display_axes.tick_params(axis="x", labelsize=6)
            display_axes.set_xlim([0.0, 360.0])
            display_axes.xaxis.set_major_formatter(x_formatter)
            display_axes.xaxis.set_major_locator(x_locator)

            cmap = matplotlib.colors.ListedColormap(cmap_seed(np.tile(np.linspace(0, 1, steps), 2)))
            for i, color in enumerate(cmap.colors):
                color_dict[i] = color
        else:
            steps = 100

            color_dict = {}
            cmap_seed = matplotlib.cm.get_cmap('YlGnBu', steps)
            norm = matplotlib.colors.Normalize(0, steps)
            display_axes = axis.inset_axes(bounds=[0.79, 0.01, 0.21, 0.01])

            cb = matplotlib.colorbar.ColorbarBase(display_axes,
                                                  cmap=matplotlib.cm.get_cmap('YlGnBu', steps),
                                                  norm=norm,
                                                  orientation='horizontal')
            cb.outline.set_visible(False)
            x_formatter = FixedFormatter([
                "0", ".2", ".4", ".6", ".8", "1.0"])
            x_locator = FixedLocator([0, 20, 40, 60, 80, 100])
            display_axes.xaxis.tick_top()
            display_axes.tick_params(axis="x", labelsize=6)
            display_axes.set_xlim([0.0, 100.0])
            display_axes.xaxis.set_major_formatter(x_formatter)
            display_axes.xaxis.set_major_locator(x_locator)

            cmap = matplotlib.colors.ListedColormap(cmap_seed(np.tile(np.linspace(0, 1, steps), 2)))
            for i, color in enumerate(cmap.colors):
                color_dict[i] = color

        return color_dict

    @staticmethod
    def circ_r(alpha, w=None, d=0, axis=0):
        """Computes mean resultant vector length for circular data.

        Args:
            alpha: array
                Sample of angles in radians

        Kwargs:
            w: array, optional, [def: None]
                Number of incidences in case of binned angle data

            d: radians, optional, [def: 0]
                Spacing of bin centers for binned data, if supplied
                correction factor is used to correct for bias in
                estimation of r

            axis: int, optional, [def: 0]
                Compute along this dimension

        Return:
            r: mean resultant length

        Code taken from the Circular Statistics Toolbox for Matlab
        By Philipp Berens, 2009
        Python adaptation by Etienne Combrisson
        """
        if w is None:
            w = np.ones(alpha.shape)
        elif alpha.size is not w.size:
            raise ValueError("Input dimensions do not match")

        # Compute weighted sum of cos and sin of angles:
        r = np.multiply(w, np.exp(1j * alpha)).sum(axis=axis)

        # Obtain length:
        r = np.abs(r) / w.sum(axis=axis)

        # For data with known spacing, apply correction factor to
        # correct for bias in the estimation of r
        if d != 0:
            c = d / 2 / np.sin(d / 2)
            r = c * r

        return np.array(r)

    def calc_interburst_distribution(self):
        """Calculate the distribution of interburst intervals for all individuals in a simulation.

        :returns: Flat list of interburst distributions
        """
        starts_of_bursts = {}
        for firefly in self.firefly_array:
            starts_of_bursts[firefly.number] = []
            flashes = firefly.flashes_per_burst
            for i, yes in enumerate(firefly.flashed_at_this_step):
                if yes and flashes == firefly.flashes_per_burst:
                    starts_of_bursts[firefly.number].append(i)
                    flashes -= 1
                else:
                    if yes:
                        flashes -= 1
                        if flashes == 0:
                            flashes = firefly.flashes_per_burst

        interburst_distribution = [[starts_of_bursts[a][i+1] - starts_of_bursts[a][i]
                                   for i in range(len(starts_of_bursts[a])-1)]
                                   for a in starts_of_bursts.keys()]
        flat_interburst_distribution = [item for sublist in interburst_distribution for item in sublist]

        return flat_interburst_distribution

    def peak_variances(self, thresh):
        """Finds peaks and the variances of the burst regions around them.

        :returns dict of peak locations
        :returns dict of peak heights
        :returns int of step of last tallest peak
        :returns dict of variances"""
        _prominences = {
            1: 1.0,
            5: 2.0,
            10: 3.0,
            15: 4.0,
            20: 5.0
        }
        _x = self.get_burst_data()
        prominences = peak_prominences(list(_x.values())[thresh:], find_peaks(list(_x.values())[thresh:])[0], wlen=30)

        prominence_threshold = _prominences[len(self.firefly_array)]
        try:
            peaks_of_x = find_peaks(list(_x.values())[thresh:],
                                    height=2.0, #prominence_threshold,
                                    distance=50
                                    )
            if len(peaks_of_x[0]) < 5:
                raise RuntimeError('Not enough peeeeeeks')
        except RuntimeError:
            try:
                peaks_of_x = find_peaks(list(_x.values())[thresh:],
                                        prominence=2.0,
                                        distance=50
                                        )
                if len(peaks_of_x[0]) < 5:
                    raise RuntimeError('Not enough peeeeeeks')
            except RuntimeError:
                peaks_of_x = find_peaks(list(_x.values())[thresh:],
                                        prominence=1.0,
                                        distance=50
                                        )

        peaks = [peak * self.timestepsize for peak in peaks_of_x[0]]
        try:
            peak_heights = [peak for peak in peaks_of_x[1]['peak_heights']]
        except KeyError:
            peak_heights = [peak for peak in peaks_of_x[1]['prominences']]
        mids = [(peaks[i + 1] + peaks[i]) / 2 for i in range(len(peaks) - 1)]
        all_flashes = []
        for ff in self.firefly_array:
            all_flashes.extend([i for i, val in enumerate(ff.flashed_at_this_step[thresh:]) if val])
        all_flashes = sorted(all_flashes)
        variances = {}
        for i in range(len(mids) - 1):
            variance_bucket = [f for f in all_flashes if mids[i] <= f * self.timestepsize < mids[i + 1]]
            variances[i] = math.sqrt(np.var(variance_bucket))
        last_high_step = 0
        for e in variances.keys():
            if variances[e] > 30:
                last_high_step = mids[e] * 10
        return peaks, peak_heights, last_high_step, variances

    def temporal_interburst_dist(self, thresh=None):
        """Returns dict of inter-burst intervals over time."""
        starts_of_bursts = {}
        for firefly in self.firefly_array:
            j = 0
            starts_of_bursts[firefly.number] = []
            flashes = firefly._flashes_per_burst[j]
            for i, yes in enumerate(firefly.flashed_at_this_step[0:]):
                if yes and flashes == firefly._flashes_per_burst[j]:
                    starts_of_bursts[firefly.number].append(i)
                    flashes -= 1
                else:
                    if yes:
                        flashes -= 1
                        if flashes == 0:
                            j += 1
                            flashes = firefly._flashes_per_burst[j]
        longest_list = max(list(starts_of_bursts.values()), key=lambda l: len(l))
        number_of_bursts = len(longest_list)

        # pad shorties
        for k, burst in starts_of_bursts.items():
            if len(burst) < number_of_bursts:
                starts_of_bursts[k].extend([float("inf")] * (number_of_bursts - len(burst)))

        collective_burst_starts = []
        for index in range(0, number_of_bursts):
            starting_points = np.array([burst[index] for burst in list(starts_of_bursts.values())])
            collective_burst_starts.append(np.mean(starting_points[starting_points < 1000000]))
        collective_interburst_distribution = np.array([collective_burst_starts[i + 1] - collective_burst_starts[i]
                                                       for i in range(len(collective_burst_starts) - 1)])
        temporal_interbursts = {}
        for i in range(len(collective_burst_starts) - 1):
            interburst = collective_burst_starts[i + 1] - collective_burst_starts[i]
            temporal_interbursts[i] = interburst
        return temporal_interbursts

    def swarm_interburst_dist(self,  is_one=True, is_null=False):
        """Calculate the distribution of interburst intervals for the collective bursting events.

        :returns: Flat list of interburst distributions
        """
        if is_one:
            all_flashes = []
            for firefly in self.firefly_array:
                all_flashes.extend([x for x in firefly.starts_of_bursts])
            all_flashes = sorted(all_flashes)
            isis = [max(abs(j - i), abs(k - j)) for i, j, k in zip(all_flashes, all_flashes[1:], all_flashes[2:])]
            return np.array(isis)
        else:
            starts_of_bursts = {}
            for firefly in self.firefly_array:
                starts_of_bursts[firefly.number] = firefly.starts_of_bursts

            longest_list = max(list(starts_of_bursts.values()), key=lambda l: len(l))
            number_of_bursts = len(longest_list)

            # TODO: Examine this more thoroughly
            # pad shorties
            for k, burst in starts_of_bursts.items():
                if len(burst) < number_of_bursts:
                    starts_of_bursts[k].extend([float("inf")] * (number_of_bursts - len(burst)))

            # TODO: Examine this more thoroughly
            collective_burst_starts = []
            for index in range(0, number_of_bursts):
                starting_points = np.array([burst[index] for burst in list(starts_of_bursts.values())])
                collective_burst_starts.append(np.mean(starting_points[starting_points < 1000000]))
            collective_interburst_distribution = np.array([collective_burst_starts[i+1] - collective_burst_starts[i]
                                                           for i in range(len(collective_burst_starts)-1)])
            temporal_interbursts = {}
            for i in range(len(collective_interburst_distribution) - 1):
                interburst = collective_interburst_distribution[i + 1] - collective_interburst_distribution[i]
                temporal_interbursts[i] = interburst

            peaks, _, last_high_step, _ = self.peak_variances(thresh=0)

            _collective_interburst_distribution = [peaks[i+1] - peaks[i]
                                                   for i in range(len(peaks)-1)
                                                   ]
            if not is_null:
                _collective_interburst_distribution = collective_interburst_distribution[
                    collective_interburst_distribution > 0
                ]

            cid = np.array(_collective_interburst_distribution)

            return cid

    def get_burst_data(self):
        """Male bursts.

        :returns dict of flash counts at timesteps
        """
        to_plot = {i: 0 for i in range(self.steps)}
        for step in range(self.steps):
            for firefly in self.firefly_array:
                if firefly.flashed_at_this_step[step] is True:
                    to_plot[step] += 1
        return to_plot

    def _get_burst_data(self):
        to_plot = {i: 0 for i in range(self.steps)}
        for step in range(self.steps):
            for firefly in self.firefly_array:
                x = firefly._get_flashed_at_this_step()
                if x[step] is True:
                    to_plot[step] += 1
        return to_plot

    def get_burst_data_epsilon(self):
        """Male bursts.

        :returns dict of flash counts at timesteps
        """
        to_plot = {i: 0 for i in range(self.steps)}
        for step in range(self.steps):
            for firefly in self.firefly_array:
                if firefly.epsilon[step] == 0:
                    to_plot[step] += 1
        return to_plot

    def get_burst_data_from_females(self):
        """Female bursts.

        :returns dict of flash counts at timesteps
        """
        to_plot = {i: 0 for i in range(self.steps)}
        for step in range(self.steps):
            for firefly in self.firefly_array:
                if firefly.flashed_at_this_step[step] is True and firefly.sign == -1:
                    to_plot[step] += 1
        return to_plot
