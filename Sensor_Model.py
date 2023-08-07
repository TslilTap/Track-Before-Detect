import torch

class SensorModel:
    def __init(self,SNR):
        self.fc = 10e9
        self.N = 64
        self.M = 1
        self.dx = 0.5
        self.tau = 2e-5
        self.K = 64
        self.BW = 1e7
        self.spatial_wind_type = 'hamming'
        self.slow_wind_tpye = 'hamming'
        self.fast_wind_tpye = 'hamming'

        self.look_dir = 0
        self.ego_velocity = 200
        self.height = 1000

        self.Nc = 1000
        self.CNR_dB = 55
        self.theta_c_min = -80
        self.theta_c_max = 80
        self.range_c_min = 0
        self.range_c_max = 2600
        self.vr_c_min = -5
        self.vr_c_max = 5

        self.c = 3e8
        self.rmax = self.c*self.tau/2
        self.Lambda = self.c/self.fc
        self.range_res = self.c/(2*self.BW)
        self.L_range = torch.floor(self.rmax / self.range_res)
        self.t_res = self.tau/self.L_range
        self.vrmax = self.Lambda/(2*self.tau)

        self.array = torch.arange((-self.N+1)/2, (self.N-1)/2 + 1).unsqueeze(1) * self.dx * 2
        self.tVec = torch.arange(0, (self.L_range - 1) * self.t_res + self.t_res, self.t_res).unsqueeze(1)
        self.chirp_Vec = torch.arange(0, self.K).unsqueeze(1)

        self.Nt = 1
        self.SNR_dBVec = torch.tensor(SNR).unsqueeze(1)
        self.theta_t = torch.tensor(-20).unsqueeze(1)

        self.v_sin_targets = torch.sin(2*torch.pi*self.theta_t/360)





def calc_gain(thetha):
    gain = (torch.cos(2*torch.pi*thetha))**4