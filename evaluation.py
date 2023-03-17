import pinocchio as pin
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


########################################
# Calibration of extrinsics from dataset
########################################
seed = 1
np.random.seed(seed)
pin.seed(seed)

class CalibrationPbe:

    def __init__(self, T_b_c_lst, T_c_o_lst, T_o_b_0) -> None:
        assert(len(T_b_c_lst) == len(T_c_o_lst))
        self.N = len(T_b_c_lst)
        self.T_b_c_lst =  T_b_c_lst
        self.T_c_o_lst =  T_c_o_lst
        self.T_o_b_0 = T_o_b_0

    def func(self, nu):
        """
        Computes the vector of residuals according to current guess and recorded dataset.
        -> find optimal T_o_b (or equivalently T_o_b).

        Problem:
        T_o_b_opti = argmin \sum_i_N ||res(T_c_o, T_c_b*T_b_o)_i||^2

        Optimization is done on SE(3) so we should use a minimal representation for decision variable nu
        as a local increment from 
 

        2 options:
        1) SE(3) exp6/log6, 
        nu \in se3 equivalent to R^6
        T_o_b = T_o_b_0 * exp6(nu)
        res(i) = log6((T_c_b*T_b_o)^-1  * T_c_o) = log6(T_o_b * T_b_c  * T_c_o)

        2) R^3 x S0(3) -> - and log3 errors stacked 
        nu = [trans, omega] \in R^3 x so3 equivalent to R^6
        t_o_b = t_o_b_0 + trans
        R_o_b = R_o_b_0 * exp3(omega)
        res(i) = log3(R_o_b * R_b_c  * R_c_o)

        """
        # 1 residual: error in se3 tangent space -> R^6
        # N datapoints -> residual vector is size 6*N
        # ! Cannot be an object attribute
        res = np.zeros(self.N*6)

        #################
        # Option 1: SE(3)
        T_o_b = self.T_o_b_0 * pin.exp6(nu)

        for i in range(self.N):
            T_b_c = self.T_b_c_lst[i]
            T_c_o = self.T_c_o_lst[i]
            res[6*i:6*(i+1)] = pin.log6(T_o_b * T_b_c * T_c_o).vector
        #################
        return res


#########################
# FAKE dataset simulation
#########################

# Nb simulated poses
N = 50
STD_T = 0.00
STD_O = np.deg2rad(0)
# STD_T = 0.01
# STD_O = np.deg2rad(5)

# Introduce cst bias on camera to object pose measurements
#    -> Orientation bias does not change final error but is reflected on T_b_o opti
#    -> Translation bias changes both
BIAS_CO_T = np.zeros(3)
BIAS_CO_O = np.zeros(3)
# BIAS_CO_T = np.array([0.1, 0.0, 0.0])
# BIAS_CO_O = np.deg2rad(np.array([40, 30.0, 0.0]))

# Ground truth robot base  
T_b_o = pin.SE3.Random()
T_o_b = T_b_o.inverse()

# Ground truth robot poses (simulated configurations)
# TODO: random posi and angles
T_b_c_lst = [pin.SE3.Random() for _ in range(N)]
T_c_b_lst = [T_b_c.inverse() for T_b_c in T_b_c_lst]
# Ground truth visual system poses
T_c_o_lst = [T_c_b * T_b_o for T_c_b in T_c_b_lst]

# Add small noise to vision
def noise_nu(std_t, std_o):
    return np.concatenate([
        np.random.normal(0, std_t*np.ones(3)),
        np.random.normal(0, std_o*np.ones(3))
    ])

T_c_o_lst = [T_c_o * pin.exp(noise_nu(STD_T, STD_O)) for T_c_o in T_c_o_lst]

# BIASED visual pose estimation
print('Introduce bias in visual pose estimation pseudo measurements')
T_c_o_lst_biased = []
for T_c_o in T_c_o_lst:
    T_c_o_biased = pin.SE3.Identity()
    T_c_o_biased.translation = T_c_o.translation + BIAS_CO_T
    T_c_o_biased.rotation = T_c_o.rotation @ pin.exp3(BIAS_CO_O)
    T_c_o_lst_biased.append(T_c_o_biased)
T_c_o_lst = T_c_o_lst_biased


###########################
# Calibration of extrinsics
###########################


T_o_b_0 = pin.SE3.Identity()  # no prior
pbe = CalibrationPbe(T_b_c_lst, T_c_o_lst, T_o_b_0)
x0 = np.zeros(6)
# result = least_squares(fun=pbe.func, x0=x0, jac='2-point', method='trf', verbose=2)
result = least_squares(fun=pbe.func, x0=x0, verbose=2)

nu_opt = result.x
# residuals_opt = result.fun
T_o_b_opt = T_o_b_0*pin.exp(nu_opt)
T_b_o_opt = T_o_b_opt.inverse()

print('T_b_o')
print(T_b_o)
print('T_b_o_opt')
print(T_b_o_opt)
print('Diff T_b_o_opt - T_b_o')
diff = pin.log6(T_b_o.inverse() * T_b_o_opt)
print('diff_t (m):   ', diff.linear)
print('diff_o (deg): ', np.rad2deg(diff.angular))

########################
# Metrics on the dataset
########################


# Plot errors between T_c_o and T_c_b_opt*T_b_o.

T_c_o_kin_lst = [T_c_b*T_b_o_opt for T_c_b in T_c_b_lst]

def err(T1, T2):
    return np.concatenate([
        T2.translation - T1.translation,
        pin.log3(T2.rotation.T @ T1.rotation)
    ])

errors = np.array([err(T_c_o, T_c_o_kin)
                   for T_c_o, T_c_o_kin in zip(T_c_o_lst, T_c_o_kin_lst)])

err_t = np.linalg.norm(errors[:,:3], axis=1)
err_o = np.rad2deg(np.linalg.norm(errors[:,3:], axis=1))

err_t_mean, err_t_sig = np.mean(err_t), np.sqrt(np.var(err_t))
err_o_mean, err_o_sig = np.mean(err_o), np.sqrt(np.var(err_o))

print()
print('T_c_o translation error mean, sigma (m): \n', err_t_mean, err_t_sig)
print('T_c_o orientation error mean, sigma (deg): \n', err_o_mean, err_o_sig)

fig, axes = plt.subplots(2)
t = np.arange(N)
for i in range(3):
    l = 'xyz'[i]
    c = 'rgb'[i]
    axes[0].plot(t, errors[:,i], f'.{c}', label=f'err_t{l}')
    axes[1].plot(t, np.rad2deg(errors[:,3+i]), f'.{c}', label=f'err_o{l}')
axes[0].legend()
axes[1].legend()
axes[0].grid()
axes[1].grid()
axes[0].set_title('Translation error (m)')
axes[1].set_title('Rotational error (deg)')

fig, axes = plt.subplots(1,2)
axes[0].boxplot(err_t, 0)
axes[0].set_title('Translation error (m)')
axes[1].boxplot(err_o, 0)
axes[1].set_title('Rotational error (deg)')
axes[0].grid()
axes[1].grid()

plt.show()
    