import mechanize
import re
import json
import sys
import os
import tqdm
import pandas as pd
directory = sys.argv[1]
result_file_name = 'result.csv'
already_processed = set()

if os.path.exists(result_file_name):
	already_processed = set(pd.read_csv(result_file_name, sep=':')['name'])
	print (already_processed)
else:
	with open(result_file_name, 'w') as result_file:
		print >>result_file, "name:keypoints_x:keypoints_y"

with open(result_file_name, 'a') as result_file:
	
	for image_file_name in tqdm.tqdm(os.listdir(directory)):
		if image_file_name in already_processed:
			continue
		image_file_name_with_directory = os.path.join(directory, image_file_name)

		br = mechanize.Browser()

		res = br.open("http://zeus.robots.ox.ac.uk/keypoint/")

		br.select_form(id="query_form")

		br.form.add_file(open(image_file_name_with_directory), 'image/jpeg', image_file_name_with_directory)

		r = br.submit()

		image_name = r.read()
		image_name = re.match(r".*result-im=([A-Z0-9.a-z]*).*", image_name)
		image_name = image_name.group(1)

		text = "http://zeus.robots.ox.ac.uk/keypoint/result-im=%s" % (image_name, )
		br.open(text)
		r = br.follow_link(text_regex=r"json")

		result = json.loads(r.read())

		print >>result_file, ":".join([image_file_name, str(result['keypoint_x']), str(result['keypoint_y'])])
		result_file.flush()
