'''
Inference server.
In default, it works as an http server.
'''

import argparse
from pt.server import Server
from flask import request, jsonify
from flask import Flask
import os
from flask_cors import *
from tools.utils import mkdirs
from tools.const import INFER_TEMP_IMG

app = Flask(__name__)
CORS(app)


def parse_args():
    '''
    This inference gets started as an http server.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        default=None,
        help="Config yaml file path.")
    parser.add_argument(
        "--given_model",
        type=str,
        default=None,
        help="Saved trained model parameters file path.")
    parser.add_argument(
        "--host",
        type=str,
        default='0.0.0.0',
        help="host param for app.run().")
    parser.add_argument(
        "--port",
        type=str,
        default='5001',
        help="port param for app.run().")
    parser.add_argument(
        "--cache_folder",
        type=str,
        default='img_cache',
        help="Image cache folder.")
    args = parser.parse_args()
    return args


@app.route("/infer", methods=['POST'])
def index_infer():
    '''
    Receive uploaded image and give out predicted result.
    RETURN: {'result': [[cx, cy, cw, ch, degree, conf, class],...]}
    '''
    upload_file = request.files['file']
    if upload_file:
        file_path = os.path.join(FLAGS.cache_folder, INFER_TEMP_IMG)
        upload_file.save(file_path)
        res = server.single_infer(img_path=file_path).tolist()
        return jsonify(result=res)

    

if __name__ == '__main__':
    FLAGS = parse_args()
    mkdirs(FLAGS.cache_folder)
    server = Server(cfg_path=FLAGS.cfg,
                    infer_only=True,
                    infer_model=FLAGS.given_model)
    app.run(host=FLAGS.host, port=FLAGS.port)
