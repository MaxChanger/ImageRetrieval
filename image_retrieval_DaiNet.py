import os
import cv2
import time
from datetime import timedelta
from retrieval.create_thumb_images import create_thumb_images
from flask import Flask, render_template, request, redirect, url_for, make_response,jsonify, flash
from retrieval.retrieval_DaiNet import load_model, load_data, extract_feature, load_query_image, sort_img, extract_feature_query

# Create thumb images.  创建缩略图
create_thumb_images(full_folder='./static/image_database/',
                    thumb_folder='./static/thumb_images/',
                    suffix='',
                    height=200,
                    del_former_thumb=True,
                    )

# Prepare data set.
data_loader = load_data(data_path='./static/image_database/',
                        batch_size=2,
                        shuffle=False,
                        transform='default',
                        )

# Prepare model. 加载预训练的model
model = load_model(pretrained_model=os.path.join('/home/SENSETIME/sunjiadai/WorkSpace/ImageRetrieval/DaiNet/checkpoint', 'DaiNet', 'ckpt.t7'), use_gpu=True)
print("Model load successfully!")

# Extract database features.
gallery_feature, image_paths = extract_feature(model=model, dataloaders=data_loader) # torch.Size([59, 2048])
print("extract_feature successfully!")

# Picture extension supported.
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp', 'jpeg', 'JPEG'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)
# Set static file cache expiration time
# app.send_file_max_age_default = timedelta(seconds=1)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)

@app.route('/', methods=['POST', 'GET'])  # add route
def image_retrieval():

    basepath = os.path.dirname(__file__)    # current path
    upload_path = os.path.join(basepath, 'static/upload_image','query.jpg')

    if request.method == 'POST':
        if request.form['submit'] == 'upload':
            if len(request.files) == 0:
                return render_template('upload_finish.html', message='Please select a picture file!',img_query='./static/upload_image/query.jpg?123456')
            else:
                f = request.files['picture']
         
                if not (f and allowed_file(f.filename)):
                    # return jsonify({"error": 1001, "msg": "Examine picture extension, only png, PNG, jpg, JPG, or bmp supported."})
                    return render_template('upload_finish.html', message='Examine picture extension, png、PNG、jpg、JPG、bmp support.',img_query='./static/upload_image/query.jpg')
                else:

                    f.save(upload_path)
             
                    # transform image format and name with opencv.
                    img = cv2.imread(upload_path)   # 从原来的读取img
                    cv2.imwrite(os.path.join(basepath, 'static/upload_image', 'query.jpg'), img) # 保存到 当前目录下
             
                    return render_template('upload_finish.html', message='Upload successfully!' ,img_query='./static/upload_image/query.jpg?123456')    # 点了upload之后的成功界面

        elif request.form['submit'] == 'retrieval':
            start_time = time.time()
            # Query.
            query_image = load_query_image('./static/upload_image/query.jpg')
            # Extract query features.
            query_feature = extract_feature_query(model=model, img=query_image) # [1,2048]
            # Sort.
            similarity, index = sort_img(query_feature, gallery_feature)
            sorted_paths = [image_paths[i] for i in index]

            print(sorted_paths) # 打印出查找之后根据相似度进行排序后的图片路径
            tmb_images = ['./static/thumb_images/' + os.path.split(sorted_path)[1] for sorted_path in sorted_paths]
            # sorted_files = [os.path.split(sorted_path)[1] for sorted_path in sorted_paths]

            return render_template('retrieval.html', message="Retrieval finished, cost {:3f} seconds.".format(time.time() - start_time),
            	sml1=similarity[0], sml2=similarity[1], sml3=similarity[2], sml4=similarity[3], sml5=similarity[4], sml6=similarity[5], sml7=similarity[6], sml8=similarity[7], sml9=similarity[8],
            	img1_tmb=tmb_images[0], img2_tmb=tmb_images[1],img3_tmb=tmb_images[2],img4_tmb=tmb_images[3],img5_tmb=tmb_images[4],img6_tmb=tmb_images[5],img7_tmb=tmb_images[6],img8_tmb=tmb_images[7],img9_tmb=tmb_images[8],img_query='./static/upload_image/query.jpg?123456')

    return render_template('upload.html')


if __name__ == '__main__':
    # app.debug = True
    app.run(host='127.0.0.1', port=8080, debug=True)