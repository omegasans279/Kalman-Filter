import numpy as np
from matplotlib import pyplot as plt


class KalmanFilter(object):
    def __init__(self, dt, u, sigma_a, sigma_z):
        self.dt = dt # time between two iterations
        self.u = u # control signal (acceleration in this case)
        self.sigma_a = sigma_a # std deviation of acceleration
        self.sigma_z = sigma_z # std deviation of measurement

        self.A = np.array([[1,self.dt],[0,1]])
        self.B = np.array([[(self.dt**2)/2],[self.dt]])
        self.R = np.array([[(self.dt**4)/4,(self.dt**3)/2],[(self.dt**3)/2,self.dt**2]]) * self.sigma_a**2
        self.Q = sigma_z**2
        self.P = np.eye(self.A.shape[1])
        self.C = np.array([[1, 0]])
        self.x = np.array([[0], [0]]) # state of the system
        self.I = np.eye(self.C.shape[1])

    def predict(self):
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)
        self.P = np.dot(np.dot(self.A, self.P), (self.A).T) + self.R
        return self.x

    def update(self, z):
        S = np.dot(self.C, np.dot(self.P, self.C.T)) + self.Q
        K = np.dot(self.P, np.dot(self.C.T, np.linalg.inv(S)))
        self.x = self.x + K * (z - np.dot(self.C, self.x))
        self.P = np.dot((self.I - (K * self.C)), self.P)


dt = 0.1

t = np.arange(0, 100, dt) # all the time intervals our model estimates the outcome

# actual state of the system (ground truth)
real_x = 0.1 * ((t ** 2) - t)

#defining the model parameters
u = 1
sigma_a = 0.25
sigma_z = 2.0
KF = KalmanFilter(dt, u, sigma_a, sigma_z)

predictions = []
measurements = []

for x in real_x:
    # we have used a Gaussian distribution for the noise (obviously)
    z = KF.C.item(0) * x + np.random.normal(0, 50) # noisy measurement

    measurements.append(z)
    x_kf = KF.predict().item(0) # predict step
    KF.update(z) # update step

    predictions.append(x_kf)

fig = plt.figure()
# let the plotting begin

# values measured by the noisy sensor (blue)
plt.plot(t, measurements, label='Measurements', color='b', linewidth=0.5)

# "actual" position values of the car (the ground truth) (yellow)
plt.plot(t, np.array(real_x), label='Real Track', color='y', linewidth=1.5) # parabolic

# values estimated by our KF model (red)
plt.plot(t, predictions, label='Kalman Filter Prediction', color='r', linewidth=1.5)
plt.show()
