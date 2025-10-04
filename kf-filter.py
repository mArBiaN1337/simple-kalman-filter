import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, initial_uncertainty, A, B, C, measurement_noise_covar, process_noise_covar, initial_state, u, measurements, iterations):
        self.initial_uncertainty = initial_uncertainty
        self.A = A
        self.B = B
        self.C = C
        self.measurement_noise_covar = measurement_noise_covar
        self.process_noise_covar = process_noise_covar
        self.initial_state = initial_state
        self.u = u
        self.measurements = measurements
        self.iterations = iterations
        self.x_pred_history = np.array([])
        self.x_pred_history = np.append(self.x_pred_history, initial_state[0,0])

    def graph_states(self, measured, predicted, samples=150):

        plt.plot(range(samples), measured[0:samples], label='Measured', color='red', linestyle='dotted')
        plt.plot(range(samples), predicted[0:samples], label='Predicted', color='blue')
        plt.xlabel('Time Step')
        plt.ylabel('Position')
        plt.title('Kalman Filter: Measured vs Applied Filter Positions')
        plt.legend()
        plt.grid()
        plt.show()
        
    def filter(self, output_file : str):
        x_old = self.initial_state.T
        P_old = self.initial_uncertainty

        # Clear the output file before writing
        with open(output_file, "w") as file:
            file.write("")  # Just to clear the file
            # write datastamp
            file.write(f"Data processed on: {np.datetime64('now', 's')}\n")

        with open(output_file, "a") as file:
            #only 3 decimal places
            file.write("iter, measured, state-pred, uncertainty\n")
            file.write(f"{0},{self.measurements[0]},{x_old[0,0]:.3f},{P_old[0,0]:.3f}\n")

        for k in range(1,self.iterations):
            # 1. Predict State Estimate (xhat_k|k-1)
            # xhat_pred = A * xhat_old + B * u_k 
            x_pred = self.A @ x_old + self.B * self.u[k]
            # 2. Predict Error Covariance (P_k|k-1)
            # P_pred = A * P_old * A.T + Q 
            P_pred = self.A @ P_old @ self.A.T + self.process_noise_covar

            # 3. Compute Kalman Gain (K_k)
            # K = P_pred * C.T * inv(C * P_pred * C.T +
            S = self.C @ P_pred @ self.C.T + self.measurement_noise_covar
            K_gain = P_pred @ self.C.T @ np.linalg.inv(S)

            # 4. Update State Estimate (xhat_k|k)
            # xhat_new = xhat_pred + K * (y_k - C * xhat_pred)
            y_k = np.array([[self.measurements[k]]]) # Current measurement
            y_diff = y_k - (self.C @ x_pred)
            xhat_new = x_pred + (K_gain @ y_diff)

            # 5. Update Error Covariance (P_k|k)
            # P_new = (I - K * C) * P_pred
            I = np.eye(P_pred.shape[0])
            P_new = (I - K_gain @ self.C) @ P_pred

            x_old = xhat_new
            P_old = P_new

            self.x_pred_history = np.append(self.x_pred_history, xhat_new[0,0])

            with open(output_file, "a") as file:
                file.write(f"{k},{self.measurements[k]},{xhat_new[0,0]:.3f},{P_new[0,0]:.3f}\n")

        self.graph_states(self.measurements, self.x_pred_history, 100)
    
# Example usage
if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=6)
    measurements = np.loadtxt('position-data.txt')

    dt = 0.1
    A = np.array([[1, dt], [0, 1]])
    B = np.array([[0.5 * dt**2], [dt]])
    C = np.array([[1, 0]])
    
    pos_noise_std = 10
    accel_value = 1
    accel_noise_std = 0.2

    position_noise_var = ((dt**2)/2)**2*accel_noise_std**2
    velocity_noise_var = dt**2*accel_noise_std**2
    pos_vel_covar = ((dt**2)/2)*accel_noise_std * dt*pos_noise_std

    # [[p^2 pv] ; [vp v^2]]
    process_noise_covar = np.array([[position_noise_var, pos_vel_covar], [pos_vel_covar, velocity_noise_var]])
    measurement_noise_var = pos_noise_std**2
    measurement_noise_covar = np.array([[measurement_noise_var]])
    initial_state = np.array([[0, 0]])
    initial_uncertainty = np.array([[0, 0], [0, 0]])

    iterations = len(measurements)

    u = 1 * np.ones_like(measurements) # constant acceleration 1m/s^2

    kf = KalmanFilter(initial_uncertainty, A, B, C, measurement_noise_covar, process_noise_covar, initial_state, u, measurements, iterations)

    kf.filter("estimated-positions.txt")
    
