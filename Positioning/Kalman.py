import numpy as np

class CTRVKalmanFilter:
    def __init__(self, dt):
        """
        Initialize the Kalman Filter.
        :param dt: Time step
        """
        self.dt = dt
        
        # State vector [x, y, v, psi, psi_dot]
        self.x = np.zeros(5)
        
        # State covariance matrix
        self.P = np.eye(5) 
        
        # Process noise covariance matrix
        self.Q =  np.diag([2, 2, 2, 1, 1e-1])
        
        # Measurement noise covariance matrix
        self.R = np.diag([1, 1])
        
        # Measurement matrix
        self.H = np.array([[1, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0]])
        
    def predict(self):
        """Predict the next state and state covariance."""
        psi = self.x[3]
        psi_dot = self.x[4]
        
        if abs(psi_dot) > 1e-5:
            self.x[0] += (self.x[2] / psi_dot) * (np.sin(psi + psi_dot * self.dt) - np.sin(psi))
            self.x[1] += (self.x[2] / psi_dot) * (np.cos(psi) - np.cos(psi + psi_dot * self.dt))
        else:
            self.x[0] += self.x[2] * np.cos(psi) * self.dt
            self.x[1] += self.x[2] * np.sin(psi) * self.dt
        
        self.x[3] += psi_dot * self.dt

        # Update the state transition Jacobian
        F = np.eye(5)
        if abs(psi_dot) > 1e-5:
            F[0, 3] = (self.x[2] / psi_dot) * (np.cos(psi + psi_dot * self.dt) - np.cos(psi))
            F[0, 4] = (self.x[2] / (psi_dot ** 2)) * (np.sin(psi) - np.sin(psi + psi_dot * self.dt)) + (self.x[2] / psi_dot) * self.dt * np.cos(psi + psi_dot * self.dt)
            F[1, 3] = (self.x[2] / psi_dot) * (np.sin(psi + psi_dot * self.dt) - np.sin(psi))
            F[1, 4] = (self.x[2] / (psi_dot ** 2)) * (np.cos(psi + psi_dot * self.dt) - np.cos(psi)) + (self.x[2] / psi_dot) * self.dt * np.sin(psi + psi_dot * self.dt)
        else:
            F[0, 3] = -self.x[2] * self.dt * np.sin(psi)
            F[1, 3] = self.x[2] * self.dt * np.cos(psi)
        
        # Update the state covariance
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        """
        Update the state estimate using the measurement.
        
        :param z: Measurement vector [z_x, z_y]
        """
        # Measurement residual
        y = z - self.H @ self.x
        
        # Measurement residual covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Updated state estimate
        self.x = self.x + K @ y
        
        # Updated state covariance
        I = np.eye(len(self.x))
        self.P = (I - K @ self.H) @ self.P



