from scipy.optimize import differential_evolution
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
import torch.nn as nn
import torch
import pandas as pd
import ghhops_server as hs
from flask import Flask
import warnings
warnings.simplefilter("ignore", UserWarning)


app = Flask(__name__)
hops = hs.Hops(app)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(5, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 1024)
        self.fc5 = nn.Linear(1024, 512)
        self.fc6 = nn.Linear(512, 256)
        self.fc7 = nn.Linear(256, 128)
        self.fc8 = nn.Linear(128, 9)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = F.gelu(self.fc3(x))
        x = F.gelu(self.fc4(x))
        x = F.gelu(self.fc5(x))
        x = F.gelu(self.fc6(x))
        x = F.gelu(self.fc7(x))
        x = self.fc8(x)
        return x


# Load model
device = torch.device('cpu')
model = Net().to(device)
model.load_state_dict(torch.load(
    "model/model_bs8_lr1.6e-05_wd0.01_epochs2000_tl6.15.pt", map_location=torch.device('cpu')))
model.eval()

# Load data
df = pd.read_csv("data/purged.csv")

# Build scaler and fit data range
in_scaler = MinMaxScaler(feature_range=(0, 100))
in_scaler.fit(df.iloc[:, :5])
out_scaler = MinMaxScaler(feature_range=(0, 100))
out_scaler.fit(df.iloc[:, 5:])


def _predict(sc_height, sc_orient, room_orient, room_width, room_depth):
    input = torch.from_numpy(in_scaler.transform(pd.DataFrame(
        [[sc_height, sc_orient, room_orient, room_width, room_depth]])).astype('float32'))

    predicted_output = model(input)
    predicted_output = out_scaler.inverse_transform(
        predicted_output.detach().numpy())

    return predicted_output[0].tolist()


@hops.component(
    "/predict",
    name="Predict",
    description="Predict the performance with given solar chimney design parameters",
    icon="predict.jpeg",
    inputs=[
        hs.HopsNumber("SolarChimneyHeight", "sc_height",
                      "Solar chimney height", default=4.0),
        hs.HopsNumber("SolarChimneyOrient", "sc_orient",
                      "Solar chimney orientation", default=0.0),
        hs.HopsNumber("RoomOrient", "room_orient",
                      "Room orientation", default=0.0),
        hs.HopsNumber("RoomWidth", "room_width",
                      "Room width", default=10.0),
        hs.HopsNumber("RoomDepth", "room_depth",
                      "Room depth", default=10.0),
    ],
    outputs=[
        hs.HopsNumber("FlowRate", "flow_rate", "Air flow rate, m3/hr"),
        hs.HopsNumber("TotalDiscomfortHours", "total_dh",
                      "Total discomfort hours all year, hr/yr"),
        hs.HopsNumber("TooHotDiscomfortHours", "too_hot_dh",
                      "Too hot discomfort hours all year, hr/yr"),
        hs.HopsNumber("OT_TotalDiscomfortHours", "ot_total_dh",
                      "Total discomfort hours during office time, hr/yr"),
        hs.HopsNumber("OT_TooHotDiscomfortHours", "ot_too_hot_dh",
                      "Too hot discomfort hours during office time, hr/yr"),
        hs.HopsNumber("NS_OT_TotalDiscomfortHours", "ns_ot_total_dh",
                      "New schedule (with cooling) total discomfort hours during office time , hr/yr"),
        hs.HopsNumber("NS_OT_TooHotDiscomfortHours", "ns_ot_toohot_dh",
                      "New schedule (with cooling) too hot discomfort hours during office time , hr/yr"),
        hs.HopsNumber("TotalEnergyUse", "total_eui",
                      "Total energy use intensity, kwh/yrm2"),
        hs.HopsNumber("CoolingEnergyUSe", "cooling_eui",
                      "Cooling energy use intensity (new schedule), kwh/yrm2"),
    ],
)
def predict(sc_height, sc_orient, room_orient, room_width, room_depth):
    res = _predict(sc_height, sc_orient, room_orient, room_width, room_depth)
    return tuple(res)


def _optimize(room_orient: float, room_width: float, room_depth: float, alpha: float):
    BETA_AMPLIFICATION = 1000  # Empirically picked

    # Closures for internal usage
    def objective(x, x3, x4, x5, a):
        input = torch.tensor([[x[0], round(x[1]), x3, x4, x5]], dtype=torch.float32)
        output = model(input)
        res = a * output[0][6].item() + BETA_AMPLIFICATION * x[0] / (x4 * x5 + 1)
        return res

    def in_norm(sc_height, sc_orient, room_orient, room_width, room_depth):
        return tuple(in_scaler.transform(pd.DataFrame(
            [[sc_height, sc_orient, room_orient, room_width, room_depth]])).astype('float32')[0].tolist())

    def in_inverse(norm_sc_height, norm_sc_orient, norm_room_orient, norm_room_width, norm_room_depth):
        return tuple(in_scaler.inverse_transform(pd.DataFrame(
            [[norm_sc_height, norm_sc_orient, norm_room_orient, norm_room_width, norm_room_depth]])).astype('float32')[0].tolist())

    _, _, norm_room_orient, norm_room_width, norm_room_depth = in_norm(
        0, 0, room_orient, room_width, room_depth)

    # We use scipy's differential evolution algorithm for optimization on global minimal
    result = differential_evolution(objective, [(0, 100), (0, 100)], args=(
        norm_room_orient, norm_room_width, norm_room_depth, alpha), seed=1)
    norm_sc_height, norm_sc_orient = result.x
    sc_height, sc_orient, _, _, _ = in_inverse(
        norm_sc_height, norm_sc_orient, 0, 0, 0)
    return sc_height, round(sc_orient)


@hops.component(
    "/optimize",
    name="Optimize",
    description=("Optimize solar chimney design parameters given room orientation and dimensions, "
                 "minimizing objective = alpha * normalized_NS_OT_THDH + normalized_Volumn_Ratio"),
    icon="optimize.jpeg",
    inputs=[
        hs.HopsNumber("RoomOrient", "room_orient",
                      "Room orientation", default=0.0),
        hs.HopsNumber("RoomWidth", "room_width",
                      "Room width", default=10.0),
        hs.HopsNumber("RoomDepth", "room_depth",
                      "Room depth", default=10.0),
        hs.HopsNumber("Alpha", "alpha",
                      "Ratio of coefficient of NewScheduleOfficeTimeTooHotDiscomfortHours over VolumeRatio", default=1.0),
    ],
    outputs=[
        hs.HopsNumber("OptimizedSolarChimneyHeight", "op_sc_height",
                      "Optimized solar chimney height"),
        hs.HopsNumber("OptimizedSolarChimneyOrient", "op_sc_orient",
                      "Optimized solar chimney orientation"),
        hs.HopsNumber("FlowRate", "flow_rate", "Air flow rate, m3/hr"),
        hs.HopsNumber("TotalDiscomfortHours", "total_dh",
                      "Total discomfort hours all year, hr/yr"),
        hs.HopsNumber("TooHotDiscomfortHours", "too_hot_dh",
                      "Too hot discomfort hours all year, hr/yr"),
        hs.HopsNumber("OT_TotalDiscomfortHours", "ot_total_dh",
                      "Total discomfort hours during office time, hr/yr"),
        hs.HopsNumber("OT_TooHotDiscomfortHours", "ot_too_hot_dh",
                      "Too hot discomfort hours during office time, hr/yr"),
        hs.HopsNumber("NS_OT_TotalDiscomfortHours", "ns_ot_total_dh",
                      "New schedule (with cooling) total discomfort hours during office time , hr/yr"),
        hs.HopsNumber("NS_OT_TooHotDiscomfortHours", "ns_ot_toohot_dh",
                      "New schedule (with cooling) too hot discomfort hours during office time , hr/yr"),
        hs.HopsNumber("TotalEnergyUse", "total_eui",
                      "Total energy use intensity, kwh/yrm2"),
        hs.HopsNumber("CoolingEnergyUSe", "cooling_eui",
                      "Cooling energy use intensity (new schedule), kwh/yrm2"),
    ],
)
def optimize(room_orient, room_width, room_depth, alpha):
    sc_height, sc_orient = _optimize(
        room_orient, room_width, room_depth, alpha)
    return tuple([sc_height, sc_orient] + _predict(sc_height, sc_orient, room_orient, room_width, room_depth))


if __name__ == "__main__":
    app.run()
