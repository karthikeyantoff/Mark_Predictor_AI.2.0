from flask import Flask, render_template, request
import torch
import torch.nn as nn

app = Flask(__name__)

#Load model config
config = torch.load("csv_folder/model_config.pth")
activation_map = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    "gelu": nn.GELU(),
    "leaky_relu": nn.LeakyReLU(),
    "silu": nn.SiLU(),
}
activation = activation_map[config["activation"]]

#Load model
model = nn.Sequential(
    nn.Linear(1, config["hidden1"]),
    activation,
    nn.Linear(config["hidden1"], config["hidden2"]),
    activation,
    nn.Linear(config["hidden2"], 2)
)
model.load_state_dict(torch.load("csv_folder/pytorch_traindata.pth"))
model.eval()


def predict(hours):
    x = torch.tensor([[hours]], dtype=torch.float32)
    with torch.no_grad():
        output = model(x)
    marks, grade = output[0]
    return round(marks.item(), 2), round(grade.item(), 2)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        hours = float(request.form["hours"])
        marks, grade = predict(hours)
        result = f"Marks: {marks}, Grade: {grade}"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)

