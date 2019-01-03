from flask import Flask, render_template, url_for, request, redirect
import json

app = Flask(__name__)

@app.route("/")
def hello():
    return render_template('home.html')

@app.route("/settings",methods=['GET','POST'])
def settings():
    if "changed" in request.args:
        changed=request.args['changed']
    try:
        changed
    except NameError:
        changed=False
    if "type_0" in request.form:
        changed=True
        nbGestes=len(request.form)//2
        newSettings =list()
        geste='A'
        for i in range(nbGestes):
            d=dict()
            d['gesture']=geste
            d['type']=request.form[f"type_{i}"]
            d['value']=request.form[f"value_{i}"]
            newSettings.append(d)
            geste=chr(ord(geste)+1)

        with open('settings.json', 'w') as f:
            json.dump(newSettings,f)
        return redirect(url_for("settings",changed=True))

    with open('settings.json', 'r') as f:
        data = json.load(f)
    return render_template('settings.html',settings=data,changed=changed)
