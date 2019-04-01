import numpy as np   
import tensorflow as tf
from PIL import Image
import ops as utils_ops
import label_map_util
import visualization_utils as vis_util
from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from pathlib import Path
import uvicorn, aiohttp, asyncio
import base64, sys, numpy as np
import time


path = Path(__file__).parent
model_file_url = 'https://www.dropbox.com/s/933b2dtccnmlfc3/frozen_inference_graph.pb?dl=1'
model_file_name = 'frozen_inference_graph'

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))
PATH_TO_LABELS = r'app/open.pbtxt'
MODEL_PATH = r'app/models/frozen_inference_graph.pb'
IMG_FILE_SRC = '/tmp/saved_image.png'
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=22, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_model():
    #UNCOMMENT HERE FOR CUSTOM TRAINED MODEL
    print("Downloading Model")
    print(MODEL_PATH)
    await download_file(model_file_url, MODEL_PATH)
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(MODEL_PATH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='') # Load your Custom trained model
    #model = ResNet50(weights='imagenet') # COMMENT, IF you have Custom trained model
    return detection_graph


# Asynchronous Steps
loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_model())]
model = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    #img_bytes = await (data["img"]).read()
    img_bytes = data["img"]
    bytes = base64.b64decode(img_bytes)
    with open(IMG_FILE_SRC, 'wb') as f: f.write(bytes)
    return model_predict(IMG_FILE_SRC, model)


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)
  
def run_inference_for_single_image(image, graph):
  time__=time.time()  
  with graph.as_default():
    with tf.Session() as sess:
      print("sess time",time.time()-time__)
      # Get handles to input and output tensors
      time__=time.time()  
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)

      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
      print("Tensors time",time.time()-time__)
      # Run inference
      print("Run inference")
      time__=time.time()
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})
      print("inf Time",time.time()-time__)
      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict


def model_predict(img_path, model):
    #print("img path",img_path)
    time_0 = time.time()
    image = Image.open(img_path)
    print("Open Image",time.time()-time_0)
    time_0 = time.time()
    image_np = load_image_into_numpy_array(image)
    print("Image np",time.time()-time_0)
    time_0 = time.time()
    output_dict = run_inference_for_single_image(image_np, model)
    print("Inference",time.time()-time_0)
    time_0 = time.time()
    vis_util.visualize_boxes_and_labels_on_image_array(
  image_np,
  output_dict['detection_boxes'],
  output_dict['detection_classes'],
  output_dict['detection_scores'],
  category_index,
  instance_masks=output_dict.get('detection_masks'),
  use_normalized_coordinates=True,
  min_score_thresh=.6,
  line_thickness=1)
    print("BBox Time",time.time()-time_0)
    img=Image.fromarray(image_np)
    img.save(IMG_FILE_SRC)
    listush = []
    classes = output_dict['detection_classes']
    for i in range(3):
        listush.append((str(round(output_dict['detection_scores'][i]*100,2))+' %',category_index[classes[i]]['name']))
    result_html1 = path/'static'/'result1.html'
    result_html2 = path/'static'/'result2.html'
    result_html = str(result_html1.open().read() +str(listush) + result_html2.open().read())
    return HTMLResponse(result_html)

@app.route("/")
def form(request):
    index_html = path/'static'/'index.html'
    return HTMLResponse(index_html.open().read())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
    #>if "serve" in sys.argv: uvicorn.run(app, host="0.0.0.0", port=8080)
