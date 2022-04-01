from flask import Blueprint, render_template
from flask import Flask, render_template, request, jsonify
from start import db

#main flask app, responsible for the homepage.

main = Blueprint('main', __name__)

lock = 1;
over = 0;
mask = 0;


@main.route('/')
def index():
    return render_template('index.html')

@main.route('/profile')
def profile():
    return render_template('profile.html')

@main.route("/home", methods=["GET", "POST"])
def home():
    import globalVars

    global lock, over, mask
    
    if request.method == "GET":
        return render_template("home.html")

    print(request.json)
    if (request.json.get("maskModeVal") != None):
        mask = request.json.get("maskModeVal")
       
    if (request.json.get("lockModeVal") != None):
         lock = request.json.get("lockModeVal")
       
    if (request.json.get("overrideVal") != None):
         over = request.json.get("overrideVal")
        


    return jsonify({"data": {"maskMode":  mask, "lockMode":  lock, "override": over}})

   
@main.route('/_get_open')
def get_open():
    import globalVars
    global lock, over, mask

    print("lock is "+str(lock))
    print("mask is "+str(mask))
    print("over is "+str(over))

    if (over == 1):
        globalVars.doorLocked = False
    else:
        if (lock == 0):
            globalVars.doorLocked = True
        else:
            if (mask == 1):
                if (globalVars.prediction > 0.9):
                    globalVars.doorLocked = False
                else:
                    globalVars.doorLocked = True
            else:
                if (globalVars.prediction != -1):
                    globalVars.doorLocked = False
                else:
                    globalVars.doorLocked = True


    print("globalVars doorLocked is " + str(globalVars.doorLocked))

    if (lock):
        return jsonify(result = "True")
    else:
        return jsonify(result = "False")

@main.route('/_get_image')
def get_image():
    import globalVars
    # print("GLOBAL VARS doorLocked is " + str(globalVars.doorLocked))
    if (not globalVars.doorLocked):
        return jsonify(result = "True")
    else:
        return jsonify(result = "False")


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", debug=True, port = 80)
