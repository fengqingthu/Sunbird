from flask import Flask
import ghhops_server as hs
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler

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
model = Net()
model.load_state_dict(torch.load(
    "model/model_bs8_lr1.6e-05_wd0.01_epochs2000_tl6.15.pt", map_location=torch.device('cpu')))

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
    icon="predict.png",
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


if __name__ == "__main__":
    app.run()
