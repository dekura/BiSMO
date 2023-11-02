'''
add rdsmo.
'''
import torch
from torch import nn

from src.models.litho.source import Source
from src.models.litho.img_mask import Mask


class SO_Module(nn.Module):
    def __init__(self,
                source: Source,
                mask: Mask,
                source_acti: str = 'sigmoid',
                source_sigmoid_steepness: float = 8,
                resist_sigmoid_steepness: float = 30,
                resist_intensity: float = 0.225,
                dose_list: list = [0.98, 1.00, 1.02],
                lens_n_liquid: float = 1.44,
                lens_reduction: float = 0.25,
                low_light_thres: float = 0.001,
                device: str = "cuda:0",
                ) -> None:
        super().__init__()

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
        self.source_acti = source_acti
        self.source_sigmoid_steepness = source_sigmoid_steepness
        self.resist_sigmoid_steepness = resist_sigmoid_steepness
        self.resist_intensity = resist_intensity

        # activation func
        self.sigmoid_source = nn.Sigmoid()

        # loss params
        self.dose_list = dose_list

        self.device = torch.device(device)

        # self.source_value = self.source.data.float().to(self.device)
        self.init_freq_domain_on_device()
        # create a param 'self.source_params'
        self.init_source_params()

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

    def sigmoid_resist(self, aerial) -> torch.Tensor:
        return torch.sigmoid(
            self.resist_sigmoid_steepness * (aerial - self.resist_intensity)
        )

    def init_source_params(self) -> None:
        # [-1, 1]
        self.source_params = nn.Parameter(torch.zeros(self.source.data.shape))

        # for sigmoid
        if self.source_acti == "sigmoid":
            self.source_params.data[torch.where(self.source.data > 0.5)] = 2
            self.source_params.data.sub_(1)
        elif self.source_acti == "cosine":
            self.source_params.data[torch.where(self.source.data > 0.5)] = 0.1
            self.source_params.data[torch.where(self.source.data <= 0.5)] = torch.pi - 0.1
        else:
            # default cosine
            self.source_params.data[torch.where(self.source.data > 0.5)] = 0.1
            self.source_params.data[torch.where(self.source.data <= 0.5)] = torch.pi - 0.1


    def update_source_value(self) -> None:
        if self.source_acti == "cosine":
            self.source_value = (1 + torch.cos(self.source_params)) / 2
        elif self.source_acti == "sigmoid":
            self.source_value = self.sigmoid_source(
                self.source_sigmoid_steepness * self.source_params
            )
        else:
            self.source_value = (1 + torch.cos(self.source_params)) / 2


    def cal_pupil(self,
                FX: torch.Tensor,
                FY: torch.Tensor,) -> torch.Tensor:
        R = torch.sqrt(FX ** 2 + FY ** 2)
        fgSquare = torch.square(R)
        # source used
        NA = self.source.na
        n_liquid = self.lens_n_liquid
        M = self.lens_reduction
        obliquityFactor = torch.sqrt(
            torch.sqrt(
                (1 - (M ** 2 * NA ** 2) * fgSquare) / (1 - ((NA / n_liquid) ** 2) * fgSquare)
            )
        )
        # no aberrations
        return obliquityFactor * (1 + 0j)

    def get_valid_source(self):
        self.simple_source_value = torch.reshape(self.source_value, (-1, 1)).to(self.device)
        high_light_mask = self.simple_source_value.ge(self.low_light_thres).to(self.device)

        self.simple_source_value = torch.masked_select(self.simple_source_value, high_light_mask)
        self.simple_source_fx1d = torch.masked_select(self.source_fx1d, high_light_mask)
        self.simple_source_fy1d = torch.masked_select(self.source_fy1d, high_light_mask)
        self.simple_source_fxy2 = self.simple_source_fx1d.pow(2) + self.simple_source_fy1d.pow(2)
        self.source_weight = torch.sum(self.simple_source_value)

    def get_norm_intensity(self):
        norm_pupil_fdata = self.cal_pupil(self.simple_source_fx1d, self.simple_source_fy1d)
        # get_norm_intensity
        norm_tempHAber = self.norm_spectrum_calc * norm_pupil_fdata
        norm_ExyzFrequency = norm_tempHAber.view(-1, 1).detach()
        norm_Exyz = torch.fft.fftshift(torch.fft.fft(norm_ExyzFrequency))
        norm_IntensityCon = torch.abs(norm_Exyz * torch.conj(norm_Exyz))
        norm_total_intensity = torch.matmul(
            self.simple_source_value.view(-1, 1).T, norm_IntensityCon
        )
        norm_IntensityTemp = self.lens_n_liquid * (self.dfmdg ** 2) * norm_total_intensity
        norm_Intensity = norm_IntensityTemp / self.source_weight
        self.norm_Intensity = norm_Intensity.detach()


    def forward(self, mask_value: torch.Tensor) -> tuple[list, list]:
        self.update_source_value()
        self.get_valid_source()
        self.get_norm_intensity()

        # obtain intensity
        self.intensity2D_list = []
        self.RI_list = []

        # 1. calculate pupil_fdata
        self.mask_fvalue_min = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(mask_value * self.dose_list[0])))
        self.mask_fvalue_norm = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(mask_value * self.dose_list[1])))
        self.mask_fvalue_max = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(mask_value * self.dose_list[2])))
        mask_fvalue = [self.mask_fvalue_min, self.mask_fvalue_norm, self.mask_fvalue_max]
        for fvalue in mask_fvalue:
            intensity2D = torch.zeros(self.mask.target_data.shape, dtype=torch.float32, device=self.device)
            for i in range(self.simple_source_value.shape[0]):
                rho2 = (
                    self.mask_fg2m
                    + 2
                    * (
                        self.simple_source_fx1d[i] * self.mask_fm
                        + self.simple_source_fy1d[i] * self.mask_gm
                    )
                    + self.simple_source_fxy2[i]
                )

                valid_source_mask = rho2.le(1).to(self.device)
                f_calc = (
                    torch.masked_select(self.mask_fm, valid_source_mask) + self.simple_source_fx1d[i]
                )
                g_calc = (
                    torch.masked_select(self.mask_gm, valid_source_mask) + self.simple_source_fy1d[i]
                )

                pupil_fdata = self.cal_pupil(f_calc, g_calc)

                # 2. calculate mask
                valid_mask_fdata = torch.masked_select(
                    fvalue[self.y1 : self.y2, self.x1 : self.x2].to(self.device), valid_source_mask
                )

                tempHAber = valid_mask_fdata * pupil_fdata

                # 3. calculate intensity
                ExyzFrequency = torch.zeros(rho2.shape, dtype=torch.complex64, device=self.device)
                ExyzFrequency[valid_source_mask] = tempHAber

                e_field = torch.zeros(fvalue.shape, dtype=torch.complex64, device=self.device)
                e_field[self.y1 : self.y2, self.x1 : self.x2] = ExyzFrequency

                AA = torch.fft.fftshift(torch.fft.ifft2(e_field))
                AA = torch.abs(AA * torch.conj(AA))
                AA = self.simple_source_value[i] * AA
                intensity2D += AA
            normed_intensity2D = intensity2D / self.source_weight / self.norm_Intensity
            self.intensity2D_list.append(normed_intensity2D)
            self.RI_list.append(self.sigmoid_resist(normed_intensity2D))

        return self.intensity2D_list, self.RI_list