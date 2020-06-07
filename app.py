from flask import Flask, render_template, request, redirect, url_for
from wtforms import Form, validators, FloatField
import pandas as pd
import config
from predict import predict

app = Flask(__name__)


# validators.number_range


class data_form(Form):
    radius_mean = FloatField("radius_mean", validators=[validators.InputRequired()])
    texture_mean = FloatField("texture_mean", validators=[validators.InputRequired()])
    perimeter_mean = FloatField("perimeter_mean", validators=[validators.InputRequired()])
    area_mean = FloatField("area_mean", validators=[validators.InputRequired()])
    smoothness_mean = FloatField("smoothness_mean", validators=[validators.InputRequired()])
    compactness_mean = FloatField("compactness_mean", validators=[validators.InputRequired()])
    concavity_mean = FloatField("concavity_mean", validators=[validators.InputRequired()])
    concave_points_mean = FloatField("concave_points_mean", validators=[validators.InputRequired()])
    symmetry_mean = FloatField("symmetry_mean", validators=[validators.InputRequired()])
    fractal_dimension_mean = FloatField("fractal_dimension_mean", validators=[validators.InputRequired()])
    radius_se = FloatField("radius_se", validators=[validators.InputRequired()])
    texture_se = FloatField("texture_se", validators=[validators.InputRequired()])
    perimeter_se = FloatField("perimeter_se", validators=[validators.InputRequired()])
    area_se = FloatField("area_se", validators=[validators.InputRequired()])
    smoothness_se = FloatField("smoothness_se", validators=[validators.InputRequired()])
    compactness_se = FloatField("compactness_se", validators=[validators.InputRequired()])
    concavity_se = FloatField("concavity_se", validators=[validators.InputRequired()])
    concave_points_se = FloatField("concave_points_se", validators=[validators.InputRequired()])
    symmetry_se = FloatField("symmetry_se", validators=[validators.InputRequired()])
    fractal_dimension_se = FloatField("fractal_dimension_se", validators=[validators.InputRequired()])
    radius_worst = FloatField("radius_worst", validators=[validators.InputRequired()])
    texture_worst = FloatField("texture_worst", validators=[validators.InputRequired()])
    perimeter_worst = FloatField("perimeter_worst", validators=[validators.InputRequired()])
    area_worst = FloatField("area_worst", validators=[validators.InputRequired()])
    smoothness_worst = FloatField("smoothness_worst", validators=[validators.InputRequired()])
    compactness_worst = FloatField("compactness_worst", validators=[validators.InputRequired()])
    concavity_worst = FloatField("concavity_worst", validators=[validators.InputRequired()])
    concave_points_worst = FloatField("concave_points_worst", validators=[validators.InputRequired()])
    symmetry_worst = FloatField("symmetry_worst", validators=[validators.InputRequired()])
    fractal_dimension_worst = FloatField("fractal_dimension_worst", validators=[validators.InputRequired()])


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict', methods=["GET", "POST"])
def prediction():

    form = data_form(request.form)

    if request.method == "POST" and form.validate():
        radius_mean = form.radius_mean.data
        texture_mean = form.texture_mean.data
        perimeter_mean = form.perimeter_mean.data
        area_mean = form.area_mean.data
        smoothness_mean = form.smoothness_mean.data
        compactness_mean = form.compactness_mean.data
        concavity_mean = form.concavity_mean.data
        concave_points_mean = form.concave_points_mean.data
        symmetry_mean = form.symmetry_mean.data
        fractal_dimension_mean = form.fractal_dimension_mean.data
        radius_se = form.radius_se.data
        texture_se = form.texture_se.data
        perimeter_se = form.perimeter_se.data
        area_se = form.area_se.data
        smoothness_se = form.smoothness_se.data
        compactness_se = form.compactness_se.data
        concavity_se = form.concavity_se.data
        concave_points_se = form.concave_points_se.data
        symmetry_se = form.symmetry_se.data
        fractal_dimension_se = form.fractal_dimension_se.data
        radius_worst = form.radius_worst.data
        texture_worst = form.texture_worst.data
        perimeter_worst = form.perimeter_worst.data
        area_worst = form.area_worst.data
        smoothness_worst = form.smoothness_worst.data
        compactness_worst = form.compactness_worst.data
        concavity_worst = form.concavity_worst.data
        concave_points_worst = form.concave_points_worst.data
        symmetry_worst = form.symmetry_worst.data
        fractal_dimension_worst = form.fractal_dimension_worst.data

        data_list = [radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean,
                     concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean, radius_se,
                     texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se,
                     concave_points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst,
                     perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst,
                     concave_points_worst, symmetry_worst, fractal_dimension_worst]
        data = pd.DataFrame(columns=config.FEATURES)
        data.loc["one"] = data_list
        data["id"] = 1                          # I added manually id column because I did not use any database
        prediction = predict(data)              # call the model and prediction function
        pred_malignant = prediction[1][0][1]
        pred_benign = prediction[1][0][0]

        return render_template("prediction.html", pred_malignant="% {:.2f}".format(pred_malignant*100),
                               pred_benign="% {:.2f}".format(pred_benign*100), form=form)
    else:

        return render_template("prediction.html", form=form)


if __name__ == '__main__':
    app.run(debug=True)


