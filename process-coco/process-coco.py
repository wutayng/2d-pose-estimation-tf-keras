"""
Process COCO Data into h5 ML Training Format
"""
from cocoapi.PythonAPI.pycocotools.coco import COCO
import numpy as np
import pandas as pd
import skimage.io as io
from skimage.transform import resize
import matplotlib.pyplot as plt
import pylab, json, argparse, h5py, os
import time, datetime, random, string
from PIL import Image, ImageDraw
pylab.rcParams['figure.figsize'] = (7,7)

parser = argparse.ArgumentParser(
	description='process coco dataset into ml training data')
parser.add_argument(
	'output', type=str,
	help='the file where output is written')
parser.add_argument(
	'dataDir', type=str,
	help='coco data directory')
parser.add_argument(
	'cnt', type=int,
	help='number of images/anns to be processed')
parser.add_argument(
	'sz', type=int,
	help='output image resolution')
parser.add_argument(
	'--testing', default=True,
	help='test program or run process')
parser.add_argument(
	'--dataset', type=str, default='val',
	help='train or val')

args = parser.parse_args()


def load(annFile, coco, imgIds, dataType, count):
	"""
	Load Img + Anns
	return I, image
	"""
	if args.testing:
		img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
	else:
		img = coco.loadImgs(imgIds[count])[0]

	# load image
	I = io.imread('%s/%s/%s'%(args.dataDir,dataType,img['file_name']))

	return I, img


def keypoints_df(anns):
	"""
	convert keypoints to pd dataframe
	"""
	numHumans = len(anns)
	pd_initialize = True
	exists = False
	# For each Person in Image
	for i in range(numHumans):
		# Get x and y pixel vectors
		x = anns[i]['keypoints'][0::3]
		y = anns[i]['keypoints'][1::3]

		# Add to pd DataFrame if not 0
		if all(elem == 0 for elem in x):
			pass
		else:
			exists = True
			# Initialize Array if it does not exist yet
			if pd_initialize:
				pd_data = {('x'+str(i)) : x, ('y'+str(i)): y}
				df = pd.DataFrame(pd_data)
				pd_initialize = False
			else:
				df.insert((len(df.columns)), ('x'+str(i)), x, True)
				df.insert((len(df.columns)), ('y'+str(i)), y, True)

	# If no keypoints, return None
	if exists:
		return df
	else:
		return None


def scale(I, df):
	"""
	Rescale Img + Keypoints to Desired Output Size
	Add Black to Img Boundary to Preserve Aspect Ratio
	"""
	# add black to bottom or right to make square
	max_dim = max(I.shape[0],I.shape[1])
	new_im = Image.new('RGB', (max_dim, max_dim), (0, 0, 0))
	new_im.paste(Image.fromarray(I), box=None)

	# resize to desired
	im_rescaled = new_im.resize((args.sz, args.sz))

	# scale coordinates
	scale = args.sz / max(I.shape[0],I.shape[1])
	df_rescaled = df.mul(scale).round(2)

	return im_rescaled, df_rescaled


def disp_coco(I, coco, anns):
	"""
	display coco keypoints annotations
	"""
	plt.imshow(I); plt.axis('off')
	ax = plt.gca()
	coco.showAnns(anns)
	plt.show()
	return


def disp_resized(im_scaled, df_scaled):
	# draw new points on image
	draw = ImageDraw.Draw(im_scaled)
	# Size of ellipse to draw for each joint
	sz = args.sz/120
	# For all Joints
	for index, row in df_scaled.iterrows():
		# For Num People
		for elem in range(0,int(len(row)),2):
			# if nonzero
			if row[elem] != 0:
				draw.ellipse((row[elem]-sz, row[elem+1]-sz, row[elem]+sz, row[elem+1]+sz),
							 fill = 'white', outline ='red')
				draw.point((row[elem],row[elem+1]))
	# Show Plot
	plt.imshow(im_scaled); plt.axis('off')
	ax = plt.gca()
	plt.show()
	return


def coco_test_img(df, df_scaled, im_scaled, I, coco, anns):
	"""
	Test COCO Keypoints via Image Show
	"""
	print('-----Original Keypoints Dataframe-----')
	print(df)
	print('-----Scaled Keypoints Dataframe-----')
	print(df_scaled)
	disp_coco(I, coco, anns)
	disp_resized(im_scaled, df_scaled)

	return


def coco_test_h5():
	"""
	Test COCO h5 Export
	"""
	# Read h5 Output
	hf_analyze = h5py.File(args.output, 'r')
	print('--First Iteration key--')
	print(list(hf_analyze.keys())[0])
	name = list(hf_analyze.keys())[0]
	print('--img/keypoints keys for Each Iteration--')
	print(hf_analyze[name].keys())
	# Read in h5 keypoints
	if args.testing == True:
		print('-----h5 Scaled Keypoints Dataframe-----')
		print(hf_analyze[name]['keypoints'].shape)
		print(hf_analyze[name]['keypoints'][:])
	else:
		print('--Keypoints Shape--')
		print(hf_analyze[name]['keypoints'].shape)
	# Read in h5 Image
	print('--Img Shape--')
	print(hf_analyze[name]['img'].shape)
	print('Number of Iteration Keys (Samples): {}'.format(len(list(hf_analyze.keys()))))
	print('----- metadata -----')
	print(hf_analyze.attrs.keys())
	print(hf_analyze.attrs['jointnames'])

	return

def h5_write(img, h5f, df_scaled, im_scaled):
	"""
	Write Keypoints and Image to h5
	"""
	# Create Sub Group Keys
	rand_str = ''.join(random.choices(string.ascii_lowercase, k=4))
	jkey = '/' + rand_str + str(img['id']) + '/keypoints'
	ikey = '/' + rand_str + str(img['id']) + '/img'
	
	# pd keypoints to h5
	#df_scaled.to_hdf(args.output, key=jkey)
	h5f.create_dataset(jkey, data=df_scaled)
	
	# img array to h5
	data_img = np.asarray(im_scaled)
	
	h5f.create_dataset(ikey, data=data_img)

	return

def h5_attributes(keypoint_names):
	"""
	Write metadata in h5 file
	"""
	h5f = h5py.File(args.output, 'a')
	h5f.attrs["jointnames"] = keypoint_names
	return


def coco_process():
	"""
	Process COCO Images
	"""
	# Define COCO Data to Gather
	dataType = args.dataset + '2017'
	annFile = '{}/annotations/person_keypoints_{}.json'.format(args.dataDir,dataType)

	# Remove Current h5 File from Output
	if os.path.exists(args.output):
		os.remove(args.output)

	# initialize COCO api for person keypoints annotations
	coco=COCO(annFile)

	# get all images containing given categories, select one at random
	catIds = coco.getCatIds(catNms=['person']);
	imgIds = coco.getImgIds(catIds=catIds);

	# Joint names
	with open(annFile) as f:
		json_data = json.load(f)
	keypoint_names = json_data['categories'][0]['keypoints']

	# Open h5 File
	h5f = h5py.File(args.output, 'a')

	start_time = time.time()
	# for all iterations
	for count in range(min(args.cnt,len(imgIds))):

		# load image + keypoints
		I, img = load(annFile, coco, imgIds, dataType, count)

		# load annotations
		annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
		anns = coco.loadAnns(annIds)

		# Create Dataframe
		df = keypoints_df(anns)

		# If Img has any Keypoints
		if df is not None:
			# Scale Img and Keypoints
			im_scaled, df_scaled = scale(I, df)

			# Write to h5
			h5_write(img, h5f, df_scaled, im_scaled)

			# If testing, test all and exit
			if args.testing == True:
				h5f.close()

				coco_test_img(df, df_scaled, im_scaled, I, coco, anns)

				h5_attributes(keypoint_names)

				coco_test_h5()

				exit()

		else:
			# Disregard image with no keypoints
			pass

		if count % 100 == 0:
			print('{} Samples Done - {} Taken'.format(count,
				(str(datetime.timedelta(seconds=(time.time() - start_time))))))

	return keypoint_names


if __name__ == "__main__":

	keypoint_names = coco_process()

	h5_attributes(keypoint_names)

	coco_test_h5()


    
