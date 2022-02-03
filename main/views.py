from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.core.files.storage import default_storage
from imageai.Detection.Custom import CustomObjectDetection
import tensorflow as tf
import multiprocessing


def run_ai(proc_num, file_name, count_fish, cnt):
	physical_devices = tf.config.experimental.list_physical_devices('GPU')
	assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
	tf.config.experimental.set_memory_growth(physical_devices[0], True)

	detector = CustomObjectDetection()
	detector.setModelTypeAsYOLOv3()
	detector.setModelPath("core_ai/detection_model-ex-034--loss-0016.692.h5")
	detector.setJsonPath("core_ai/detection_config.json")
	detector.loadModel()
	detections = detector.detectObjectsFromImage(input_image=file_name, output_image_path="pictures/detected.jpg", minimum_percentage_probability=31)
	for detection in detections:
		print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])
		count_fish += 1
	cnt[proc_num] = count_fish

def index(request):
	global proc_num
	data = {}
	proc_num = 0
	count_fish = 0
	cnt = multiprocessing.Manager().dict()
	print(cnt)
	if request.method == 'POST':
		file = request.FILES.get('fish')
		file_name = default_storage.save('pictures/' + file.name, file)
		p = multiprocessing.Process(target=run_ai, args=(proc_num, file_name, count_fish, cnt))
		p.start()
		p.join()

		data = {
		    'image': True,
		    'count': cnt[proc_num]
		}
		proc_num += 1

	return render(request, 'main/main_page.html', data)

def view(request):
    data = {
        'pictures': None
    }
    return render(request, 'main/view.html', data)