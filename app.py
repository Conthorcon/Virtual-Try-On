import uuid
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static"

# coding=utf-8
from engine import test_gmm, test_tom
import torch

import os
import os.path as osp
import PIL.Image as Image
import time
import numpy as np

# from visualization import board_add_image, board_add_images, save_images

from model import build_model
from model.MVTON.MVTON import setup_mvton, run_mvton
import preprocess as pre
from preprocess.utils import tensor_to_pil_mask, tensor_to_pil
from model.MVTON.utils import get_args


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

model = None
sampler = None
opt = None
start_code = None

def get_model():
    global model, sampler, opt, start_code
    if model is None:
        model, sampler, opt, start_code = setup_mvton()
    return model, sampler, opt, start_code

def run_inference(person, cloth):
    model, sampler, opt, start_code = get_model()
    opt = get_args()

    opt.stage = 'GMM'
    data = pre.CPVTON(opt, person, cloth)
    

    gmm, _ = build_model(opt)
    gmm.eval()

    data = test_gmm(opt, data, gmm)

    data = pre.MVTON1(opt, person, cloth, data)

    inpaint = tensor_to_pil(data['inpaint_image'])
    
    result = run_mvton(model, sampler, data, opt, start_code)


    return inpaint, result


@app.route("/")
def index():
    return send_from_directory(".", "base.html")

@app.route("/tryon", methods=["POST"])
def tryon():
    if "person" not in request.files or "cloth" not in request.files:
        return jsonify({"error": "Missing input images"}), 400
    
    client_id = request.form.get("client_id")
    # Load images

    person = Image.open(request.files["person"]).convert("RGB")
    cloth  = Image.open(request.files["cloth"]).convert("RGB")

    try:
        cloth_seg, result = run_inference(person, cloth)
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

    # save_dir = f"/static/{client_id}"
    # os.makedirs(save_dir, exist_ok=True)
    save_dir = osp.join(app.config["UPLOAD_FOLDER"], f"result_{client_id}.png")
    cloth_seg_dir = osp.join(app.config["UPLOAD_FOLDER"], f"cloth_seg_{client_id}.png")
    result.save(save_dir)
    cloth_seg.save(cloth_seg_dir)

    return jsonify({
        "result_url": save_dir,
        "segmentation_url": cloth_seg_dir
    })

if __name__ == "__main__": 
    model, sampler, opt, start_code = setup_mvton() 
    app.run(debug=False)
