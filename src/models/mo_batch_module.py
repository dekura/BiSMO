import math
import torch
from torch import nn

from src.models.litho.img_mask import Mask
from src.models.litho.source import Source

# import torch.nn.functional as F


# higher level.
class MO_Module(nn.Module):
    def __init__(
        self,
        source: Source,
        mask: Mask,
        mask_acti: str = "sigmoid",
        mask_sigmoid_steepness: float = 8,
        resist_sigmoid_steepness: float = 30,
        resist_intensity: float = 0.225,
        dose_list: list = [0.98, 1.00, 1.02],
        lens_n_liquid: float = 1.44,
        lens_reduction: float = 0.25,
        low_light_thres: float = 0.001,
        source_batch_size: int = 128,
        device: str = "cuda:0",
    ) -> None:
        super().__init__()

        # source and mask
        self.source = source
        self.source.update()

        self.mask = mask
        self.mask.open_layout()
        self.mask.maskfft()

        # source params
        self.lens_n_liquid = lens_n_liquid
        self.lens_reduction = lens_reduction
        self.low_light_thres = low_light_thres

        # forward params
        self.mask_acti = mask_acti
        self.mask_sigmoid_steepness = mask_sigmoid_steepness
        self.resist_sigmoid_steepness = resist_sigmoid_steepness
        self.resist_intensity = resist_intensity
        self.source_batch_size = source_batch_size
        # activation func
        self.sigmoid_mask = nn.Sigmoid()

        # loss params
        self.dose_list = dose_list
        # loss function
        self.criterion = nn.MSELoss()

        self.device = torch.device(device)

        self.init_freq_domain_on_device()
        # define mask_params
        self.init_mask_params()

    def init_freq_domain_on_device(self) -> None:
        device = self.device
        # hyper-parameters
        self.gnum = self.source.gnum
        self.fnum = self.source.fnum

        self.x_gridnum = self.mask.x_gridnum
        self.y_gridnum = self.mask.y_gridnum
        self.x1 = int(self.x_gridnum // 2 - self.fnum)
        self.x2 = int(self.x_gridnum // 2 + self.fnum + 1)
        self.y1 = int(self.y_gridnum // 2 - self.gnum)
        self.y2 = int(self.y_gridnum // 2 + self.gnum + 1)

        normalized_period_x = self.x_gridnum / (self.source.wavelength / self.source.na)
        normalized_period_y = self.y_gridnum / (self.source.wavelength / self.source.na)
        mask_fm = (torch.arange(self.x1, self.x2) - self.x_gridnum // 2) / normalized_period_x
        mask_fm = mask_fm.to(device)

        mask_gm = (torch.arange(self.y1, self.y2) - self.y_gridnum // 2) / normalized_period_y
        mask_gm = mask_gm.to(device)

        self.mask_fm, self.mask_gm = torch.meshgrid(mask_fm, mask_gm, indexing="xy")
        self.mask_fg2m = self.mask_fm.pow(2) + self.mask_gm.pow(2)

        # for intensity norm
        self.norm_spectrum_calc = normalized_period_x * normalized_period_y
        self.dfmdg = 1 / self.norm_spectrum_calc

        # source part
        self.s_fx = self.source.fx.to(device)
        self.s_fy = self.source.fy.to(device)
        self.source_fx1d = torch.reshape(self.s_fx, (-1, 1))
        self.source_fy1d = torch.reshape(self.s_fy, (-1, 1))
        self.source.data = self.source.data.to(device)

        # load mask data to device
        self.mask.data = self.mask.data.to(torch.float32).to(device)
        self.mask.fdata = self.mask.fdata.to(device)

        # load target data to device
        if hasattr(self.mask, "target_data"):
            self.mask.target_data = self.mask.target_data.to(torch.float32).to(device)
        else:
            self.mask.target_data = self.mask.data.detach().clone()

    def init_mask_params(self) -> None:
        # learnable, [-1, 1]
        self.mask_params = nn.Parameter(torch.zeros(self.mask.data.shape))
        if self.mask_acti == "sigmoid":
            self.mask_params.data[torch.where(self.mask.data > 0.5)] = 2 - 0.02
            self.mask_params.data.sub_(0.99)
        else:
            # default sigmoid
            self.mask_params.data[torch.where(self.mask.data > 0.5)] = 2 - 0.02
            self.mask_params.data.sub_(0.99)

    def sigmoid_resist(self, aerial) -> torch.Tensor:
        return torch.sigmoid(self.resist_sigmoid_steepness * (aerial - self.resist_intensity))

    def update_mask_value(self) -> None:
        if self.mask_acti == "sigmoid":
            # mask after activation func
            self.mask_value = self.sigmoid_mask(self.mask_sigmoid_steepness * self.mask_params)
        elif self.mask_acti == "multi":
            self.mask_value = self.sigmoid_mask(
                self.mask_sigmoid_steepness * (self.mask_params - 0.5)
            )
        else:
            self.mask_value = self.sigmoid_mask(self.mask_sigmoid_steepness * self.mask_params)
        # self.mask.maskfft()
        self.mask_fvalue_min = torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(self.mask_value * self.dose_list[0]))
        )
        self.mask_fvalue_norm = torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(self.mask_value * self.dose_list[1]))
        )
        self.mask_fvalue_max = torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(self.mask_value * self.dose_list[2]))
        )

    def cal_pupil(
        self,
        FX: torch.Tensor,
        FY: torch.Tensor,
    ) -> torch.Tensor:
        R = torch.sqrt(FX**2 + FY**2)  # rho
        fgSquare = torch.square(R)
        # source used
        NA = self.source.na
        n_liquid = self.lens_n_liquid
        M = self.lens_reduction
        obliquityFactor = torch.sqrt(
            torch.sqrt(
                (1 - (M**2 * NA**2) * fgSquare) / (1 - ((NA / n_liquid) ** 2) * fgSquare)
            )
        )
        # no aberrations
        return obliquityFactor * (1 + 0j)

    def get_valid_source(self, source_value):
        self.simple_source_value = torch.reshape(source_value, (-1, 1))
        high_light_mask = self.simple_source_value.ge(self.low_light_thres)

        self.simple_source_value = torch.masked_select(self.simple_source_value, high_light_mask)
        self.simple_source_fx1d = torch.masked_select(self.source_fx1d, high_light_mask)
        self.simple_source_fy1d = torch.masked_select(self.source_fy1d, high_light_mask)
        self.simple_source_fxy2 = self.simple_source_fx1d.pow(2) + self.simple_source_fy1d.pow(2)
        self.source_weight = torch.sum(self.simple_source_value)

    def get_norm_intensity(self):
        # get_norm_intensity
        norm_pupil_fdata = self.cal_pupil(self.simple_source_fx1d, self.simple_source_fy1d)
        norm_tempHAber = self.norm_spectrum_calc * norm_pupil_fdata
        norm_ExyzFrequency = norm_tempHAber.view(-1, 1).detach()
        norm_Exyz = torch.fft.fftshift(torch.fft.fft(norm_ExyzFrequency))
        norm_IntensityCon = torch.abs(norm_Exyz * torch.conj(norm_Exyz))
        norm_total_intensity = torch.matmul(
            self.simple_source_value.view(-1, 1).T, norm_IntensityCon
        )
        norm_IntensityTemp = self.lens_n_liquid * (self.dfmdg**2) * norm_total_intensity
        norm_Intensity = norm_IntensityTemp / self.source_weight
        self.norm_Intensity = norm_Intensity.detach()

    def forward(self, source_value: Source):
        self.update_mask_value()
        # self.source_data is un-learnable
        self.get_valid_source(source_value)
        self.get_norm_intensity()

        # obtain intensity
        self.intensity2D_list = []
        self.RI_list = []

        # 1. calculate pupil_fdata
        mask_fvalue = [self.mask_fvalue_min, self.mask_fvalue_norm, self.mask_fvalue_max]
        for fvalue in mask_fvalue:
            intensity2D = torch.zeros(
                self.mask.target_data.shape, dtype=torch.float32, device="cuda:3"
            )
            batch_size = self.source_batch_size
            n_batches = math.ceil(self.simple_source_value.shape[0] / batch_size)
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, self.simple_source_value.shape[0])
                batch = self.simple_source_value[start_idx:end_idx]
                device_id = batch_idx % 3
                batch_intensity2D = torch.zeros(
                    self.mask.target_data.shape, dtype=torch.float32, device=f"cuda:{device_id}"
                )
                for i in range(batch.shape[0]):
                    in_batch_idx = start_idx + i
                    rho2 = (
                        self.mask_fg2m
                        + 2
                        * (
                            self.simple_source_fx1d[in_batch_idx] * self.mask_fm
                            + self.simple_source_fy1d[in_batch_idx] * self.mask_gm
                        )
                        + self.simple_source_fxy2[in_batch_idx]
                    )

                    valid_source_mask = rho2.le(1)
                    f_calc = (
                        torch.masked_select(self.mask_fm, valid_source_mask)
                        + self.simple_source_fx1d[in_batch_idx]
                    )
                    g_calc = (
                        torch.masked_select(self.mask_gm, valid_source_mask)
                        + self.simple_source_fy1d[in_batch_idx]
                    )

                    pupil_fdata = self.cal_pupil(f_calc, g_calc)

                    # 2. calculate mask
                    valid_mask_fdata = torch.masked_select(
                        fvalue[self.y1 : self.y2, self.x1 : self.x2], valid_source_mask
                    )
                    tempHAber = valid_mask_fdata * pupil_fdata

                    # 3. calculate intensity
                    ExyzFrequency = torch.zeros(rho2.shape, dtype=torch.complex64, device=f"cuda:{device_id}")
                    ExyzFrequency[valid_source_mask] = tempHAber.to(f"cuda:{device_id}")

                    e_field = torch.zeros(fvalue.shape, dtype=torch.complex64, device=f"cuda:{device_id}")
                    e_field[self.y1 : self.y2, self.x1 : self.x2] = ExyzFrequency

                    AA = torch.fft.fftshift(torch.fft.ifft2(e_field))
                    AA = torch.abs(AA * torch.conj(AA))
                    AA = batch[i] * AA
                    batch_intensity2D += AA.to(batch_intensity2D)
                    # torch.cuda.empty_cache()
                intensity2D += batch_intensity2D.to(intensity2D)
                # torch.cuda.empty_cache()
            normed_intensity2D = intensity2D / self.source_weight.to("cuda:3") / self.norm_Intensity.to("cuda:3")
            self.intensity2D_list.append(normed_intensity2D)
            self.RI_list.append(self.sigmoid_resist(normed_intensity2D))

        return self.intensity2D_list, self.RI_list
