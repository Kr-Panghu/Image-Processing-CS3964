from PIL import Image
from ultralytics import YOLO

pic = ['pic1', 'pic2', 'pic3', 'pic4', 'pic5']

def predict_5_pics():
    for i in range(5):
        results = model.predict('inference/'+pic[i]+'.jpg', save=True, save_txt=True, device=1)

model_num = ['1', '2', '3', '4', '5', '6', '7', '8']
# metrics = model.val(data='VisDrone.yaml', split='test', device=0)

for num in model_num:
    model = YOLO('runs/detect/kr_train_' + num + '/weights/best.pt')
    predict_5_pics()

    # for i, (path, img, pred) in enumerate(results):
    #     img.save(f'predicted_image_{i}.jpg')
    # results.save(save_dir='before', save_json=True, json_name=f'results{i}.json')

    # with open(f'before/results{i}.json') as f:
    #     data = json.load(f)

    # for r in results:
    #     # Plot the prediction
    #     im_array = r.plot()
    #     im = Image.fromarray(im_array)
    #     im.show()
    #     im.save(f'before/results{i}.jpg')(base)
