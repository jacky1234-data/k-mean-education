import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
import os
import base64
import io

#Initialize the flask App
app = Flask(__name__)

#Plot are always running in the backend
plt.switch_backend('Agg')

#Some global data and the dataset
k_values = [i for i in range(2,6)]
click = 0

data = pd.read_csv("datapoints.csv") #Use your own path

#Calculate the Euclidean distance
def cal_distance(pointA, pointB):
    return ((pointA[0] - pointB[0]) ** 2 + (pointA[1] - pointB[1]) ** 2) ** 0.5

#Find the group of closet centroids
def find_close_centroid(point, centroids, all_groups):
    group = all_groups[0]
    a_centroid = centroids[0]
    min_dist = cal_distance(point, a_centroid)
    for i,j in enumerate(centroids):
        if i == 0:
            continue
        dist = cal_distance(point,j)
        if dist < min_dist:
            min_dist = dist
            group = all_groups[i]
            a_centroid = j
    return group

#Assign groups to data points
def assign_group(df, centroids):
    df = df[[i for i in df.columns if i!="group"]]
    nrow = df.shape[0]
    df.loc[:,"group"] = "n/a"
    all_groups = ["A","B","C","D","E"]
    for i in range(nrow):
        cur_point = [df.loc[i,"x"], df.loc[i,"y"]]
        df.loc[i,"group"] = find_close_centroid(cur_point,centroids,all_groups)
    return df

#Recalculate centroids
def recal_centroid(df):
    all_groups = list(set(df.loc[:,"group"]))
    all_groups.sort()
    table = []
    for each in all_groups:
        df_each = df.loc[df["group"]==each]
        x_cent = int(df_each["x"].mean())
        y_cent = int(df_each["y"].mean())
        table.append([x_cent, y_cent])
    return table

#Check if any pair of centroids are too close
def check_distance(centroids):
    for i in range(len(centroids)-1):
        for j in range(i+1,len(centroids)):
            if cal_distance(centroids[i],centroids[j]) < 20:
                return "too short"
    return "okay"

#Plot the original graph
def plot_original(df):
    plot = sns.scatterplot(data = df , x = "x", y = "y", color = "#4f5661").set(title='Original Datapoints')

    # Convert the plot to SVG format
    svg_buffer = io.BytesIO()
    plt.savefig(svg_buffer, format='svg')
    svg_buffer.seek(0)
    svg_code = base64.b64encode(svg_buffer.read()).decode('utf-8')

    # Clear the Matplotlib figure to avoid overlapping plots
    plt.clf()

    return svg_code

#Plot the clustered graph
def plot_grouped(df):
    colors = ["red","blue","green","yellow","purple"]
    groups = ["A","B","C","D","E"]
    plot = sns.scatterplot(data = df , x = "x", y = "y", hue = "group", 
                           palette = {i:j for i,j in zip(groups,colors)}).set(title='Clustered Datapoints')
    plt.legend(bbox_to_anchor=(1.12, 1),loc='upper right',fontsize="8")

    # Convert the plot to SVG format
    svg_buffer = io.BytesIO()
    plt.savefig(svg_buffer, format='svg')
    svg_buffer.seek(0)
    svg_code = base64.b64encode(svg_buffer.read()).decode('utf-8')

    # Clear the Matplotlib figure to avoid overlapping plots
    plt.clf()
    return svg_code

@app.route('/')
def home():
    global click
    click = 0
    orig_graph = plot_original(data)
    return render_template('index.html', all_k_values = k_values, kselected = 2, plot_graph = orig_graph)


#Reset everything else while fixing the k value
@app.route('/reset', methods=['POST'])
def reset():
    global click
    k = int(request.form.get('kvalues'))
    click = 0
    orig_graph = plot_original(data)
    return render_template('index.html', all_k_values = k_values, kselected = k, plot_graph = orig_graph)

#Running iteration
@app.route('/predict', methods=['POST'])
def predict():
    global click
    global cur_data
    global all_centroids
    groups = ["A","B","C","D","E"]
    click = click + 1

    if click == 1: #First iteration
        all_centroids = []
        for each in groups:
            if not request.form.get("x" + each + '_val'):
                break
            val_x = int(request.form.get("x" + each + '_val'))
            val_y = int(request.form.get("y" + each + '_val'))
            all_centroids.append([val_x, val_y])

        accpeted_distance = check_distance(all_centroids)
        if accpeted_distance == "too short":
            orig_graph = plot_original(data)
            num__k = len(all_centroids)
            alert_message = "At least a pair of centroids have distance less than 20. Please reselect centroids."
            click = 0
            return render_template('index.html', all_k_values = k_values, kselected = num__k,
                           xA_val = all_centroids[0][0], yA_val = all_centroids[0][1],
                           xB_val = all_centroids[1][0], yB_val = all_centroids[1][1],
                           xC_val = all_centroids[2][0] if num__k >= 3 else "", yC_val = all_centroids[2][1] if num__k >= 3 else "",
                           xD_val = all_centroids[3][0] if num__k >= 4 else "", yD_val = all_centroids[3][1] if num__k >= 4 else "",
                           xE_val = all_centroids[4][0] if num__k == 5 else "", yE_val = all_centroids[4][1] if num__k == 5 else "",
                           plot_graph = orig_graph, distance_alert = alert_message) #Preserve changes made by users
        
        else:
            cur_data = assign_group(data, all_centroids)
            graph = plot_grouped(cur_data)

    elif click <= 40: #Continue iteration
        prev_centroids = all_centroids[:]
        all_centroids = recal_centroid(cur_data)
        cur_data = assign_group(cur_data, all_centroids)
        graph = plot_grouped(cur_data)
        convergence = "Convergence is reached!" if prev_centroids == all_centroids else ""

    else: #Stop adjusting centroids after 40 iterations
        graph = plot_grouped(cur_data)

    
    num_k = len(all_centroids)
    click_message = "You exceeded the maximum number (40) of iteration and there will be no more changes." \
          if click > 40 else "Number of iteration: {}".format(click)

    return render_template('index.html', all_k_values = k_values, kselected = num_k,
                           xA_val = all_centroids[0][0], yA_val = all_centroids[0][1],
                           xB_val = all_centroids[1][0], yB_val = all_centroids[1][1],
                           xC_val = all_centroids[2][0] if num_k >= 3 else "", yC_val = all_centroids[2][1] if num_k >= 3 else "",
                           xD_val = all_centroids[3][0] if num_k >= 4 else "", yD_val = all_centroids[3][1] if num_k >= 4 else "",
                           xE_val = all_centroids[4][0] if num_k == 5 else "", yE_val = all_centroids[4][1] if num_k == 5 else "",
                           num_click_msg = click_message, plot_graph = graph, convergent_alert = "" if click == 1 or click > 40 else convergence,
                          ) #Preserve changes made by users

#Clear everything
@app.route('/', methods=['POST'])
def clear():
    global click
    click = 0
    orig_graph = plot_original(data)
    return render_template('index.html', all_k_values = k_values, kselected = 2, plot_graph = orig_graph)


if __name__ == "__main__":
    app.run(debug=True)  
