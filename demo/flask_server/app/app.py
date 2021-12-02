from flask import Flask, render_template, url_for, request, redirect
from flask_bootstrap import Bootstrap

import matplotlib
# matplotlib.use('TkAgg')

import os
import model

from pathlib import Path
import sys # added!


app = Flask(__name__, template_folder='template')
Bootstrap(app)
#<h1>Predicted Classes: {{ result.object_classes }}</h1>
"""
Routes
"""
@app.route('/', methods=['GET','POST'])
def index():
    sys.path.append("../..")  # added!
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            image_path = os.path.join('static', uploaded_file.filename)
            #uploaded_file.save(image_path)
            cls, plt = model.get_prediction(image_path)

            plt.savefig('../static/new_plot.png')
            #img_path = os.path.join('../', 'static', 'new_plot.png')
            img_path = Path('App/static/new_plot.png')
            print(img_path)
            result = {
                'object_classes': cls,
                'image_path': image_path,
                'img_path': img_path
            }

            return render_template('result.html', result = result, plot_name='new_plot', plot_url=img_path)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug = os.getenv('DEBUG'))