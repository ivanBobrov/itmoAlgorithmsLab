import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import least_squares

# Generating data according to the task
x = np.linspace(0, 3, 1001)
y = []
for i in range(1001):
    f_x = 1 / (x[i] ** 2 - 3 * x[i] + 2)
    if f_x < -100:
        y.append(-100 + random.gauss(0, 1))
        continue
    if f_x > 100:
        y.append(100 + random.gauss(0, 1))
        continue
    y.append(f_x + random.gauss(0, 1))


def test_function(p):
    return (p[0] * x + p[1]) / (x ** 2 + p[2] * x + p[3])


def minimization_function(a, b, c, d):
    s = 0
    for i in range(1001):
        F = (a * x[i] + b) / (x[i] ** 2 + c * x[i] + d)
        s += (F - y[i]) ** 2
    return s


nelder_mead = minimize(lambda x: minimization_function(x[0], x[1], x[2], x[3]),
                       np.array([0.5, 0.5, 0.5, 0.5]), method='Nelder-Mead',
                       options={'xatol': 0.01, 'maxiter': 1000})
nelder_mead_y = (nelder_mead.x[0] * x + nelder_mead.x[1]) / (x ** 2 + nelder_mead.x[2] * x + nelder_mead.x[3])
print("Nelder-Mead:", nelder_mead.x[0], nelder_mead.x[1], nelder_mead.x[2], nelder_mead.x[3])
print("Nelder-Mead minimization function:", minimization_function(nelder_mead.x[0], nelder_mead.x[0],
                                                                  nelder_mead.x[2], nelder_mead.x[3]))


lm = least_squares(lambda p: test_function(p) - y, (0.5, 0.5, 0.5, 0.5), method='lm', max_nfev=1000)
lm_y = (lm.x[0] * x + lm.x[1]) / (x ** 2 + lm.x[2] * x + lm.x[3])
print("Levenberg-Marquardt:", lm.x[0], lm.x[1], lm.x[2], lm.x[3])
print("Levenberg-Marquardt minimization function:", minimization_function(lm.x[0], lm.x[1],
                                                                          lm.x[2], lm.x[3]))


def particle_swarm_minimization(min_function, min_bounds, max_bounds, initial_point,
                                particles_number, iterations_number=1000):
    omega = 0.99
    phi_r = 0.5
    phi_g = 0.001

    dimensions = len(initial_point)
    best_global_position = initial_point
    best_global_position_f = min_function(best_global_position)

    particle_position = []
    particle_velocity = []
    particle_best_position = []
    particle_best_position_f = []
    print("Initializing...")
    for i in range(particles_number):
        random_initial_position_i = []
        random_initial_velocity_i = []
        random_best_position_i = []
        for d in range(dimensions):
            min_bound = min_bounds[d]
            max_bound = max_bounds[d]
            area = max_bound - min_bound

            pos_d = random.uniform(min_bound, max_bound)
            vel_d = random.uniform(-area, area)

            random_initial_position_i.append(pos_d)
            random_initial_velocity_i.append(vel_d)
            random_best_position_i.append(pos_d)

        particle_position.append(random_initial_position_i)
        particle_velocity.append(random_initial_velocity_i)
        particle_best_position.append(random_best_position_i)
        particle_best_position_f.append(min_function(random_best_position_i))

    def update_particle_best_position(particle_index):
        position = particle_position[particle_index]
        position_f = min_function(position)
        saved_best_position_f = particle_best_position_f[particle_index]
        if position_f <= saved_best_position_f:
            particle_best_position[particle_index] = position.copy()
            particle_best_position_f[particle_index] = position_f

    def update_global_minimum():
        nonlocal best_global_position, best_global_position_f
        min_value = min(particle_best_position_f)
        if min_value <= best_global_position_f:
            min_index = particle_best_position_f.index(min_value)
            best_global_position = particle_best_position[min_index].copy()
            best_global_position_f = min_value

    def iteration_particle(i):
        position: list = particle_position[i]
        best_position: list = particle_best_position[i]
        velocity: list = particle_velocity[i]
        for d in range(dimensions):
            r_p = random.random()
            r_g = random.random()

            velocity[d] = omega * velocity[d] + phi_r * r_p * (best_position[d] - position[d]) \
                          + phi_g * r_g * (best_global_position[d] - position[d])
            position[d] = position[d] + velocity[d]
            if position[d] > max_bounds[d]:
                position[d] = 2 * max_bounds[d] - position[d]
                velocity[d] = -velocity[d]
            if position[d] < min_bounds[d]:
                position[d] = 2 * min_bounds[d] - position[d]
                velocity[d] = -velocity[d]
        update_particle_best_position(i)

    print("First update globals")
    update_global_minimum()

    for iteration in range(iterations_number):
        print("Starting iteration: ", iteration)
        for i in range(particles_number): iteration_particle(i)

        for i in range(5):
            print(particle_position[i], " : ", particle_velocity[i], " : ",
                  min_function(particle_best_position[i]))

        print("Best:", best_global_position, best_global_position_f, " <---- ")
        update_global_minimum()
        print("Best:", best_global_position, best_global_position_f, " <---- ")

    return best_global_position


psw = particle_swarm_minimization(lambda p: minimization_function(p[0], p[1], p[2], p[3]),
                                  (-2, 0, -3, 0), (0, 2, -1, 2), (-1.4, 1.2, -2.3, 1.5), 1000)
psw_y = (0.5 * x + 0.5) / (x ** 2 + 0.5 * x + 0.5)
print("PSW:", psw[0], psw[1], psw[2], psw[3])
print("PSW minimization function:", minimization_function(psw[0], psw[1],
                                                          psw[2], psw[3]))

plt.figure(dpi=300)
plt.plot(x, y, marker='o', linestyle='none', markersize='1', label='data')
plt.plot(x, nelder_mead_y, label='Nelder-Mead')
plt.plot(x, lm_y, linestyle="--", label='Levenberg-Marquardt')
plt.plot(x, psw_y, linestyle="dotted", label='Particle swarm optimization')
plt.legend(frameon=False)
plt.show()