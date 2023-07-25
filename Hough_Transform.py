# Hough Transform
import numpy as np


class HoughTransform:

    """
       The Hough transform combines radar measurements across multiple timeframes into a single multi-dimensional map
       from which target tracks and states are simultaneously inferred based on a majority voting method

       Attributes:
       -----------------------------
       theta_resolution: Spacing between angles.
           Theta is between -90 degrees and 90 degrees. Default step size is 1 degree
        range_resolution: Spacing between range bins

       Returns:
       -----------------------------
       theta: array of angles used in the computation [radians]
       rhos array of rho values. Max size is twice the diagonal distance of the input image
       accumulator- 2D array of the hough transform accumulator

       """

    def __init__(self, theta_resolution=1, rho_resolution=1):
        self.theta_resolution = theta_resolution
        self.rho_resolution = rho_resolution

    def transform(self, radar_data, primary_threshold):

        """
        :param radar_data:  range-doppler measurements
                numpy array type of size (num_frames, vr_bins, range_bins)
        :param primary_threshold: initial threshold [dBs] to select points of interest
        :return:  accumulator cell
                numpy array of size ()
        """
        self.num_frames, range_bins, vr_bins = radar_data.shape
        diagonal_length = np.ceil(np.sqrt(self.num_frames ** 2 + range_bins ** 2))
        self.theta_range = np.deg2rad(np.arange(0, 180, self.theta_resolution))
        self.rho_range = np.arange(-diagonal_length, diagonal_length, self.rho_resolution)
        num_thetas = len(self.theta_range)
        num_rhos = len(self.rho_range)
        accumulators = np.zeros((num_thetas, num_rhos, vr_bins), dtype=float)

        for vr_idx in range(vr_bins):

            frame_data = radar_data[:, :, vr_idx]  # range-time space data matrix

            edge_points = np.nonzero(frame_data > primary_threshold)
            t_points, r_points, = edge_points
            # if vr_idx >=32 and vr_idx <= 33:
            #          plt.figure(figsize=(10, 6))
            #          plt.imshow(frame_data, origin='lower', aspect='auto', ) # clutter appears as veritcal lines
            #          plt.show()
            #          print(f' Correct trajectory {vr_idx}')

            for i in range(len(r_points)):

                t = t_points[i]
                r = r_points[i]

                for theta_index in range(0,75):
                    rho = (r * np.cos(self.theta_range[theta_index]) + t * np.sin(self.theta_range[theta_index]))  # scaled representation of the range
                    idx = np.argmin(np.abs(self.rho_range-rho)) # returns false index
                    accumulators[theta_index, idx, vr_idx] += 1
                for theta_index in range(100,180):
                    rho = (r * np.cos(self.theta_range[theta_index]) + t * np.sin(self.theta_range[theta_index]))  # scaled representation of the range
                    idx = np.argmin(np.abs(self.rho_range-rho)) # returns false index
                    accumulators[theta_index, idx, vr_idx] += 1

                pass



        return accumulators

    def inverse_transform(self, accumulator, rho_conversion_factor, theta_conversion_factor):
        """
        The hough transform returns a tuple (rho_o, theta_0) in which the reverse hough transform
        maps these values back to the true target trajectory
        :param r_scale (scalar): maps the range index bin to the actual radial distance from the source
        :param theta_0 (scale): maps the directionality of the target to the radial velocity
        :return:
        initial range r0: y-intercept rho_0/cos(theta_0)
        radial velocity ( slope ) vr_o = -tan(theta)
        """
        potential_tracks = []
        max_value = np.max(accumulator)
        secondary_threshold = int(max_value* (2/3))
        max_indices = np.unravel_index(np.argmax(accumulator), accumulator.shape)

        # TODO: Velocity information is already encoded into the accumulator cell
        temp = np.nonzero(accumulator > secondary_threshold)     #transform into sets to have unique elements
        candidate_tracks = [item for item in zip( temp[0], temp[1], temp[2]) if item[0] != 90]
        val = float('-inf')

        for _track in candidate_tracks:
            theta_idx = _track[0]
            rho_idx = _track[1]
            radial_vel = _track[2]
            val_acc = accumulator[theta_idx,rho_idx,radial_vel]
            if val_acc >= val:
                val = val_acc
                theta = self.theta_range[theta_idx]
                rho = self.rho_range[rho_idx]
                r0 = int(rho/np.cos(theta)) # theta correspond the index -> convert to actual value
                """range = ρ * ρ_conversion_factor
                    time = θ * θ_conversion_factor"""
                vr = radial_vel  # perhaps translation
                # print(np.abs(radial_vel - vr_scale * np.tan(theta)))
        return r0,vr
