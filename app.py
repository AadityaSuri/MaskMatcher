from flask import Flask, render_template, request, jsonify

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():        
    if request.method == "GET":
        return render_template("index.html")

    mask = request.json.get("maskModeVal")
    lock = request.json.get("lockModeVal")
    over = request.json.get("overrideVal")
    print(lock)
    print(mask)
    print(over)
    return jsonify({"data": {"maskMode": mask, "lockMode": lock, "override":over}})

# @app.route("/", methods=["GET", "POST"])
# def parse():
#     print("parsing")
#     maskChecked, lockChecked, overrideChecked = False, False, False
#     if request.form.get("maskMode"):
#         maskChecked = not maskChecked
#         print (maskChecked)
#     if request.form.get("lockMode"):
#         lockChecked = not lockChecked
#         print (lockChecked)
#     if request.form.get("override"):
#         overrideChecked = not overrideChecked
#         print(overrideChecked)
        
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=True)

#Joshua Testing


# @app.route('/test', methods=['GET', 'POST'])
# def testfn():
#     # GET request
#     if request.method == 'GET':
#         message = {'greeting':'Hello from Flask!'}
#         return jsonify(message)  # serialize and use JSON headers
#     # POST request
#     if request.method == 'POST':
#         print(request.get_json())  # parse as JSON
#         return 'Sucesss', 200


